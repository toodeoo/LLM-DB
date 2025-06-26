import pandas as pd
import numpy as np
import math
from collections import Counter
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

class LeafNode:
    def __init__(self, data_indices, optimal_col_order):
        self.data_indices = data_indices
        self.optimal_col_order = optimal_col_order
        self.size = len(data_indices)

    def __repr__(self):
        return f"Leaf(size={self.size}, order={self.optimal_col_order})"

class TreeNode:
    def __init__(self, split_column, split_value, left_child, right_child):
        self.split_column = split_column
        self.split_value = split_value
        self.left_child = left_child
        self.right_child = right_child

    def __repr__(self):
        if isinstance(self.split_value, (set, frozenset)):
            display_val = str(list(self.split_value)[:3])
            if len(self.split_value) > 3: display_val = display_val[:-1] + ', ...]'
            return f"Node(split_on='{self.split_column}' IN {display_val})"
        else:
            return f"Node(split_on='{self.split_column}' <= {self.split_value:.2f})"


# --- 2. 主算法框架 (引入新配置) ---

class PTreePartitioner:
    def __init__(self,
                 max_depth=5,
                 min_rows_per_partition=50,
                 top_k_candidates=5,
                 numeric_split_strategy='histogram',
                 max_categorical_cardinality=20):
        """
        Args:
            numeric_split_strategy (str): 数值列的分裂策略, 'quantile' 或 'histogram'.
            max_categorical_cardinality (int): 处理类别列时，考虑的最大唯一值数量，防止组合爆炸。
        """
        self.max_depth = max_depth
        self.min_rows_per_partition = min_rows_per_partition
        self.top_k_candidates = top_k_candidates
        self.numeric_split_strategy = numeric_split_strategy
        self.max_categorical_cardinality = max_categorical_cardinality
        
        self.tree_ = None
        self.df_ = None
        self.col_widths_ = None
        self.numerical_cols_ = []
        self.categorical_cols_ = []
        self.cols_for_partitioning_ = []

    def fit(self, df: pd.DataFrame, columns_for_partitioning: list):
        self.df_ = df.copy()
        self.cols_for_partitioning_ = columns_for_partitioning
        self._estimate_column_widths()
        self._identify_column_types()
        self.tree_ = self._build_ptree(self.df_, depth=0)
        print("PTree has been successfully built.")
        return self.tree_

    def _build_ptree(self, df: pd.DataFrame, depth: int):
        if depth >= self.max_depth or len(df) <= self.min_rows_per_partition:
            final_order = self._find_final_optimal_order(df)
            return LeafNode(df.index, final_order)

        best_split_info = self._find_best_split(df)

        if not best_split_info or best_split_info['gain'] <= 1e-6:
            final_order = self._find_final_optimal_order(df)
            return LeafNode(df.index, final_order)
        
        split_col, split_val = best_split_info['column'], best_split_info['value']
        left_df, right_df = self._split_data(df, split_col, split_val)
        
        if len(left_df) == 0 or len(right_df) == 0:
            final_order = self._find_final_optimal_order(df)
            return LeafNode(df.index, final_order)

        print(f"  Splitting node at depth {depth} on {TreeNode(split_col, split_val, None, None).__repr__()}. "
              f"Left: {len(left_df)}, Right: {len(right_df)}")
        left_child = self._build_ptree(left_df, depth + 1)
        right_child = self._build_ptree(right_df, depth + 1)
        return TreeNode(split_col, split_val, left_child, right_child)


    def _get_candidate_splits(self, df: pd.DataFrame):
        candidates = []
        for col in self.numerical_cols_:
            if col in self.cols_for_partitioning_:
                if self.numeric_split_strategy == 'histogram':
                    candidates.extend(self._get_histogram_splits(df, col))
                else: # 'quantile'
                    candidates.extend(self._get_quantile_splits(df, col))
        
        for col in self.categorical_cols_:
            if col in self.cols_for_partitioning_:
                candidates.extend(self._get_categorical_splits(df, col))
                
        return candidates

    def _get_quantile_splits(self, df, col):
        if df[col].nunique() < 2: return []
        quantiles = df[col].quantile([0.25, 0.5, 0.75]).unique()
        return [(col, q) for q in quantiles]

    def _get_histogram_splits(self, df, col, bins=10):
        if df[col].nunique() < 3: return self._get_quantile_splits(df, col)
        
        try:
            counts, bin_edges = np.histogram(df[col].dropna(), bins=bins)
            valleys = []
            for i in range(1, len(counts) - 1):
                if counts[i-1] > counts[i] < counts[i+1]:
                    split_point = (bin_edges[i] + bin_edges[i+1]) / 2
                    valleys.append((col, split_point))
            return valleys if valleys else self._get_quantile_splits(df,col)
        except Exception:
            return self._get_quantile_splits(df, col)

    def _get_categorical_splits(self, df, col):
        unique_cats = df[col].unique()
        if not (2 <= len(unique_cats) <= self.max_categorical_cardinality):
            return []

        category_scores = {}
        for cat in unique_cats:
            subset_df = df[df[col] == cat]
            base_order = self._greedy_column_order(subset_df, metric='pu')
            category_scores[cat] = self._calculate_lwpe(subset_df, base_order)
            
        sorted_categories = sorted(category_scores, key=category_scores.get)
        
        splits = []
        for i in range(1, len(sorted_categories)):
            subset_A = frozenset(sorted_categories[:i])
            splits.append((col, subset_A))
            
        return splits
        
    def _split_data(self, df, column, value):
        if isinstance(value, (set, frozenset)):
            left_mask = df[column].isin(value)
        else:
            left_mask = df[column] <= value
        
        return df[left_mask], df[~left_mask]
    
    def _find_best_split(self, df: pd.DataFrame):
        candidate_splits = self._get_candidate_splits(df)
        if not candidate_splits: return None
        parent_order = self._greedy_column_order(df, metric='pu')
        parent_lwpe = self._calculate_lwpe(df, parent_order)
        gains_pu = []
        for col, val in candidate_splits:
            left_df, right_df = self._split_data(df, col, val)
            if len(left_df) == 0 or len(right_df) == 0: continue
            gain = self._calculate_pu_gain(df, left_df, right_df, parent_order, parent_order)
            gains_pu.append({'column': col, 'value': val, 'gain': gain})
        if not gains_pu: return None
        top_k = sorted(gains_pu, key=lambda x: x['gain'], reverse=True)[:self.top_k_candidates]
        best_split_info, max_lwpe_gain = None, -float('inf')
        for candidate in top_k:
            col, val = candidate['column'], candidate['value']
            left_df, right_df = self._split_data(df, col, val)
            pi_left = self._greedy_column_order(left_df, metric='pu')
            pi_right = self._greedy_column_order(right_df, metric='pu')
            gain = self._calculate_lwpe_gain(df, parent_lwpe, left_df, right_df, pi_left, pi_right)
            if gain > max_lwpe_gain:
                max_lwpe_gain, best_split_info = gain, {'column': col, 'value': val, 'gain': gain}
        return best_split_info

    def _greedy_column_order(self, df: pd.DataFrame, metric='pu'):
        if df.empty: return []
        remaining_cols = list(self.cols_for_partitioning_)
        optimal_order = []
        while remaining_cols:
            best_col, min_metric_val = None, float('inf')
            for col in remaining_cols:
                current_order = optimal_order + [col]
                metric_val = self._calculate_pu(df, current_order) if metric == 'pu' else self._calculate_lwpe(df, current_order)
                if metric_val < min_metric_val:
                    min_metric_val, best_col = metric_val, col
            if best_col is None: best_col = remaining_cols[0]
            optimal_order.append(best_col)
            remaining_cols.remove(best_col)
        return optimal_order
    
    def _find_final_optimal_order(self, df: pd.DataFrame):
        return self._greedy_column_order(df, metric='lwpe')

    def _calculate_pu(self, df, column_order):
        if df.empty or not column_order:
            return 0
        col = column_order[0]
        if col not in df.columns:
            return 0
        return df[col].nunique()

        # order_to_use = [c for c in column_order if c in df.columns]
        # if not order_to_use:
        #     return 0
        # return df[order_to_use].astype(str).agg(''.join, axis=1).nunique()
        
    # def _calculate_pu_multicolumn(self, df, column_order):
    #     if df.empty or not column_order:
    #         return 0
    #     order_to_use = [c for c in column_order if c in df.columns]
    #     if not order_to_use:
    #         return 0
    #     return df.groupby(order_to_use).ngroups

    # def _calculate_lwpe(self, df, column_order):
    #     if df.empty or not column_order: return 0.0
    #     order_to_use = [c for c in column_order if c in df.columns]
    #     if not order_to_use: return 0.0
    #     prefixes = df[order_to_use].astype(str).agg(''.join, axis=1)
    #     n = len(prefixes)
    #     if n == 0: return 0.0
    #     counts = Counter(prefixes)
    #     prefix_len = sum(self.col_widths_.get(col, 1) for col in order_to_use)
    #     entropy = -sum((f_p / n) * prefix_len * math.log(f_p / n) for p, f_p in counts.items() if f_p > 0)
    #     return entropy
    def _calculate_lwpe(self, df, column_order):
        if df.empty or not column_order: return 0.0
        order_to_use = [c for c in column_order if c in df.columns]
        if not order_to_use: return 0.0
        n = len(df)
        if n == 0: return 0.0
        counts = df.groupby(order_to_use).size()
        prefix_len = sum(self.col_widths_.get(col, 1) for col in order_to_use)
        probabilities = counts / n
        entropy = -np.sum(probabilities * prefix_len * np.log(probabilities))
        return entropy

    def _calculate_pu_gain(self, parent_df, left_df, right_df, pi_left, pi_right):
        parent_pu = self._calculate_pu(parent_df, pi_left)
        left_pu, right_pu = self._calculate_pu(left_df, pi_left), self._calculate_pu(right_df, pi_right)
        n_p, n_l, n_r = len(parent_df), len(left_df), len(right_df)
        if n_p == 0: return 0
        return parent_pu - ((n_l / n_p) * left_pu + (n_r / n_p) * right_pu)

    def _calculate_lwpe_gain(self, parent_df, parent_lwpe, left_df, right_df, pi_left, pi_right):
        n_p, n_l, n_r = len(parent_df), len(left_df), len(right_df)
        if n_p == 0: return 0
        lwpe_left = self._calculate_lwpe(left_df, pi_left)
        lwpe_right = self._calculate_lwpe(right_df, pi_right)
        return parent_lwpe - ((n_l / n_p) * lwpe_left + (n_r / n_p) * lwpe_right)

    def _estimate_column_widths(self):
        self.col_widths_ = {c:1 for c in self.df_.columns}
        for col in self.df_.columns:
            if self.df_[col].dtype == 'object':
                non_null_strings = self.df_[col].dropna().astype(str)
                if not non_null_strings.empty:
                    self.col_widths_[col] = non_null_strings.str.len().mean() / 4.0

    def _identify_column_types(self):
        self.numerical_cols_ = [c for c in self.df_.select_dtypes(include=np.number).columns if c in self.cols_for_partitioning_]
        # Treat bool as categorical
        self.categorical_cols_ = [c for c in self.df_.select_dtypes(include=['object', 'category', 'bool']).columns if c in self.cols_for_partitioning_]
        
    def get_partitions(self):
        partitions = []
        def traverse(node):
            if isinstance(node, LeafNode):
                partitions.append({'indices': node.data_indices, 'optimal_order': node.optimal_col_order})
                return
            traverse(node.left_child)
            traverse(node.right_child)
        if self.tree_: traverse(self.tree_)
        return partitions

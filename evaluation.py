import pandas as pd
import numpy as np
import requests
import time
import os
import argparse
import tiktoken
from tqdm import tqdm
from ptree_partitioner import PTreePartitioner

SGLANG_URL = "http://127.0.0.1:30000/generate"
try:
    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    print("Warning: tiktoken not found. Using simple character length for token counting.")
    ENCODER = None

TASK_PROMPT_TEMPLATES = {
    "T1": lambda row: f"""Given the following metadata and song details, determine if this music track is suitable for public play. Answer ONLY with "Yes" or "No".

Metadata:
{str(row['metadata'])}
Has Paywall: {row['has_paywall']}
Is Rated: {row['is_rated']}
Is Official: {row['is_official']}
Is Draft: {row['is_draft']}
Title: {row['title']}
Artist: {row['artist_name']}

Answer:""",
    "T2": lambda row: f"""Summarize the musical style and key highlights of the track given below.

Title: {row['title']}
Artist: {row['artist_name']}
Genres: {row['genres']}
Complexity: {row['complexity']}
Song Length (seconds): {row['song_length.seconds']}

Additional Metadata:
{str(row['metadata'])}

Summary:""",
    "T3_filter": lambda row: f"""Is this an official version of the music track? Answer ONLY "Yes" or "No".

Is Official: {row['is_official']}
Title: {row['title']}
Artist: {row['artist_name']}""",
    "T3_projection": lambda row: f"""Given the track information, summarize the features that make it stand out.

Title: {row['title']}
Artist: {row['artist_name']}
Genres: {row['genres']}
Complexity: {row['complexity']}
Metadata: {str(row['metadata'])}

Summary:""",
    "T4": lambda row: f"""Rate the complexity of this musical track on a scale from 1 (simple) to 5 (complex).

Title: {row['title']}
Artist: {row['artist_name']}
Genres: {row['genres']}
Complexity: {row['complexity']}

Score:""",
}


def count_tokens(text):
    if ENCODER:
        return len(ENCODER.encode(text, disallowed_special=()))
    return len(text)

def query_sglang_batch(prompts, max_new_tokens):
    payload = {
        "text": prompts,
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(SGLANG_URL, json=payload, timeout=60)
        resp.raise_for_status()
        results = resp.json()
        return [item.get("text", "").strip() for item in results]
    except requests.exceptions.RequestException as e:
        print(f"Warning: SGLang batch request failed: {e}")
        return ["ERROR"] * len(prompts)


def estimate_cost(total_tokens, provider="OpenAI"):
    rates = {"OpenAI": 0.0015, "Anthropic": 0.003} # Price per 1K tokens
    return total_tokens * rates.get(provider, 0.0) / 1000

def prefix_hit_score(prompts):
    if len(prompts) < 2: return 0.0
    total, hit = 0, 0
    for i in range(1, len(prompts)):
        prefix = os.path.commonprefix([prompts[i], prompts[i - 1]])
        hit += len(prefix)
        total += len(prompts[i])
    return hit / (total + 1e-5)

def make_prompt(row, task_type, step=None):
    key = task_type if not step else f"{task_type}_{step}"
    if key not in TASK_PROMPT_TEMPLATES:
        raise ValueError(f"Unsupported task or step: {key}")
    return TASK_PROMPT_TEMPLATES[key](row)

def normalize(text):
    return str(text).strip().lower().replace(".", "").replace(":", "")


def run_inference_on_dataframe(df, task_type, max_new_tokens, batch_size):
    all_prompts = []
    results_list = []
    
    print(f"  Running inference on {len(df)} records with batch size {batch_size}...")

    if task_type == "T3":
        filter_prompts = [make_prompt(row, "T3", "filter") for _, row in df.iterrows()]
        all_prompts.extend(filter_prompts)
        filter_preds = []
        for i in tqdm(range(0, len(filter_prompts), batch_size), desc="T3 Filter Step"):
            batch_prompts = filter_prompts[i:i + batch_size]
            filter_preds.extend(query_sglang_batch(batch_prompts, max_new_tokens))
        
        df['filter_pred'] = filter_preds
        df_filtered = df[df['filter_pred'].str.lower() == 'yes'].copy()
        print(f"  T3 Filter step resulted in {len(df_filtered)} official tracks.")

        if not df_filtered.empty:
            proj_prompts = [make_prompt(row, "T3", "projection") for _, row in df_filtered.iterrows()]
            all_prompts.extend(proj_prompts)
            proj_preds = []
            for i in tqdm(range(0, len(proj_prompts), batch_size), desc="T3 Projection Step"):
                batch_prompts = proj_prompts[i:i + batch_size]
                proj_preds.extend(query_sglang_batch(batch_prompts, max_new_tokens))
            df_filtered['pred'] = proj_preds
        
        final_df = df_filtered

    else:
        prompts = [make_prompt(row, task_type) for _, row in df.iterrows()]
        all_prompts.extend(prompts)
        preds = []
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Task {task_type}"):
            batch_prompts = prompts[i:i + batch_size]
            preds.extend(query_sglang_batch(batch_prompts, max_new_tokens))
        
        df['pred'] = preds
        df['tokens'] = [count_tokens(p) for p in prompts]
        final_df = df

    return final_df, all_prompts

def analyze_and_print_summary(label, df_results, prompts, total_runtime, task_type):
    print(f"\n📊 {label} Evaluation Summary:")
    if "tokens" not in df_results.columns and task_type != "T3":
        total_tokens = sum(count_tokens(p) for p in prompts)
    elif task_type == "T3":
        total_tokens = sum(count_tokens(p) for p in prompts)
    else:
        total_tokens = df_results["tokens"].sum()
    cost = estimate_cost(total_tokens)
    prefix_rate = prefix_hit_score(prompts)
    print(f"⏱️ Total Runtime: {total_runtime:.2f}s")
    print(f"🚀 Prefix Hit Rate: {prefix_rate:.2%}")
    print(f"💰 Estimated Cost (OpenAI): ${cost:.4f} for {total_tokens:,} tokens")


def main(args):
    print(f"Loading data from {args.input}...")
    try:
        df_full = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        return
    required_cols = { "T1": ["metadata", "has_paywall", "is_rated", "is_official", "is_draft", "title", "artist_name"], "T2": ["title", "artist_name", "genres", "complexity", "song_length.seconds", "metadata"], "T3": ["is_official", "title", "artist_name", "genres", "complexity", "metadata"], "T4": ["title", "artist_name", "genres", "complexity"] }
    if args.task_type not in required_cols:
        print(f"Error: Task {args.task_type} is not defined with required columns.")
        return
    
    df_clean = df_full[required_cols[args.task_type]].dropna().copy()
    
    if args.max_samples and args.max_samples < len(df_clean):
        print(f"Randomly sampling {args.max_samples} records...")
        df = df_clean.sample(n=args.max_samples, random_state=42).copy()
    else:
        df = df_clean.copy()

    print("\n" + "="*50)
    print("RUNNING BASELINE (SEQUENTIAL ORDER)")
    print("="*50)
    start_baseline = time.time()
    baseline_df = df.copy()
    baseline_results_df, baseline_prompts = run_inference_on_dataframe(baseline_df, args.task_type, args.max_new_tokens, args.batch_size)
    baseline_runtime = time.time() - start_baseline
    analyze_and_print_summary("Baseline", baseline_results_df, baseline_prompts, baseline_runtime, args.task_type)
    # baseline_results_df.to_csv(f"eval_result_{args.task_type}_baseline.csv", index=False)
    print(f"✅ Baseline results saved to: eval_result_{args.task_type}_baseline.csv")

    # --- PTree ---
    if args.use_ptree:
        print("\n" + "="*50)
        print("RUNNING WITH PTREE OPTIMIZATION")
        print("="*50)
        
        ptree_decision_cols_map = { "T1": ["artist_name", "has_paywall", "is_rated", "is_official", "is_draft"], "T2": ["artist_name", "genres", "complexity", "song_length.seconds"], "T3": ["artist_name", "is_official", "genres"], "T4": ["artist_name", "genres", "complexity"] }
        cols_for_ptree = [c for c in ptree_decision_cols_map[args.task_type] if c in df.columns]
        COLUMN_LABELS = { "has_paywall": "Has Paywall", "is_rated": "Is Rated", "is_official": "Is Official", "is_draft": "Is Draft", "artist_name": "Artist", "genres": "Genres", "complexity": "Complexity", "song_length.seconds": "Song Length (seconds)" }

        # --- T3  ---
        if args.task_type == 'T3':
            total_ptree_build_time = 0
            all_optimized_prompts = []

            print("--- Step 1: Optimized Filter ---")
            df_for_ptree_filter = df.copy()
            for col in cols_for_ptree:
                if df_for_ptree_filter[col].dtype == 'object' or df_for_ptree_filter[col].dtype == 'bool':
                    df_for_ptree_filter[col] = df_for_ptree_filter[col].astype('category').cat.codes

            start_ptree_build = time.time()
            partitioner_filter = PTreePartitioner(max_depth=4, min_rows_per_partition=30, top_k_candidates=5)
            partitioner_filter.fit(df_for_ptree_filter, cols_for_ptree)
            partitions_filter = partitioner_filter.get_partitions()
            ptree_build_time = time.time() - start_ptree_build
            total_ptree_build_time += ptree_build_time
            print(f"PTree for Filter built in {ptree_build_time:.2f}s. Data divided into {len(partitions_filter)} partitions.")

            filter_prompts_optimized = []
            filter_reordered_indices = []
            for part in tqdm(partitions_filter, desc="Generating Optimized Filter Prompts"):
                for idx in part['indices']:
                    filter_reordered_indices.append(idx)
                    row = df.loc[idx]
                    prompt = f"""Is this an official version of the music track? Answer ONLY "Yes" or "No".

Is Official: {row['is_official']}
Title: {row['title']}
Artist: {row['artist_name']}"""
                    filter_prompts_optimized.append(prompt)
            
            all_optimized_prompts.extend(filter_prompts_optimized)
            
            start_filter_run = time.time()
            filter_preds = []
            for i in tqdm(range(0, len(filter_prompts_optimized), args.batch_size), desc="Optimized T3 Filter Step"):
                batch_prompts = filter_prompts_optimized[i:i + args.batch_size]
                filter_preds.extend(query_sglang_batch(batch_prompts, args.max_new_tokens))
            filter_runtime = time.time() - start_filter_run

            df_reordered_with_preds = df.loc[filter_reordered_indices].copy()
            df_reordered_with_preds['filter_pred'] = filter_preds
            df_filtered = df_reordered_with_preds[df_reordered_with_preds['filter_pred'].str.lower() == 'yes'].copy()
            print(f"  Optimized T3 Filter step resulted in {len(df_filtered)} official tracks.")

            final_preds = df_filtered['filter_pred'].tolist() # 存储第一步的结果
            final_reordered_indices = df_filtered.index.tolist()
            projection_runtime = 0

            if not df_filtered.empty:
                print("\n--- Step 2: Optimized Projection ---")
                df_for_ptree_proj = df_filtered.copy()
                for col in cols_for_ptree:
                    if df_for_ptree_proj[col].dtype == 'object' or df_for_ptree_proj[col].dtype == 'bool':
                        df_for_ptree_proj[col] = df_for_ptree_proj[col].astype('category').cat.codes
                
                start_ptree_build = time.time()
                partitioner_proj = PTreePartitioner(max_depth=4, min_rows_per_partition=30, top_k_candidates=5)
                partitioner_proj.fit(df_for_ptree_proj, cols_for_ptree)
                partitions_proj = partitioner_proj.get_partitions()
                ptree_build_time = time.time() - start_ptree_build
                total_ptree_build_time += ptree_build_time
                print(f"PTree for Projection built in {ptree_build_time:.2f}s. Data divided into {len(partitions_proj)} partitions.")

                proj_prompts_optimized = []
                proj_reordered_indices = []
                for part in tqdm(partitions_proj, desc="Generating Optimized Projection Prompts"):
                    optimal_order = part['optimal_order']
                    for idx in part['indices']:
                        proj_reordered_indices.append(idx)
                        row = df.loc[idx]
                        details_list = [f"{COLUMN_LABELS.get(col, col.replace('_', ' ').title())}: {row[col]}" for col in optimal_order]
                        dynamic_details = "\n".join(details_list)
                        remaining_cols = [c for c in required_cols[args.task_type] if c not in optimal_order]
                        remaining_details = "\n".join([f"{COLUMN_LABELS.get(c, c.replace('_', ' ').title())}: {row[c]}" for c in remaining_cols])
                        
                        prompt = f"""Given the track information, summarize the features that make it stand out.

{dynamic_details}
{remaining_details}

Summary:"""
                        proj_prompts_optimized.append(prompt)

                all_optimized_prompts.extend(proj_prompts_optimized)
                
                start_proj_run = time.time()
                proj_preds = []
                for i in tqdm(range(0, len(proj_prompts_optimized), args.batch_size), desc="Optimized T3 Projection Step"):
                    batch_prompts = proj_prompts_optimized[i:i + args.batch_size]
                    proj_preds.extend(query_sglang_batch(batch_prompts, 100)) # Projection 可能需要更多 token
                projection_runtime = time.time() - start_proj_run

                df_filtered = df.loc[proj_reordered_indices].copy()
                df_filtered['pred'] = proj_preds
                final_preds = proj_preds
                final_reordered_indices = proj_reordered_indices
            
            optimized_runtime = filter_runtime + projection_runtime
            optimized_results_df = df.loc[final_reordered_indices].copy()
            optimized_results_df['pred'] = final_preds
            optimized_results_df['tokens'] = [count_tokens(p) for p in all_optimized_prompts]

        # --- others ---
        else:
            print(f"Applying PTree partitioning using columns: {cols_for_ptree}...")
            df_for_ptree = df.copy()
            for col in cols_for_ptree:
                if df_for_ptree[col].dtype == 'object' or df_for_ptree[col].dtype == 'bool':
                    df_for_ptree[col] = df_for_ptree[col].astype('category').cat.codes
            
            start_ptree_build = time.time()
            partitioner = PTreePartitioner(max_depth=4, min_rows_per_partition=30, top_k_candidates=5)
            partitioner.fit(df_for_ptree, cols_for_ptree)
            partitions = partitioner.get_partitions()
            total_ptree_build_time = time.time() - start_ptree_build
            print(f"PTree built in {total_ptree_build_time:.2f}s. Data divided into {len(partitions)} partitions.")

            print("Dynamically generating prompts based on PTree's optimal column orders...")
            all_optimized_prompts = []
            reordered_indices = []
            
            for part in tqdm(partitions, desc="Generating Optimized Prompts"):
                optimal_order = part['optimal_order']
                indices = part['indices']
                reordered_indices.extend(indices)

                for idx in indices:
                    row = df.loc[idx]
                    details_list = [f"{COLUMN_LABELS.get(col, col.replace('_', ' ').title())}: {row[col]}" for col in optimal_order]
                    dynamic_details = "\n".join(details_list)
                    remaining_cols = [c for c in required_cols[args.task_type] if c not in optimal_order]
                    
                    if args.task_type == 'T1':
                        prompt = f"""Given the following song details, determine if this music track is suitable for public play. Answer ONLY with "Yes" or "No".

{dynamic_details}
Title: {row['title']}
Metadata: {str(row['metadata'])}

Answer:"""
                    else:
                        remaining_details = "\n".join([f"{COLUMN_LABELS.get(c, c.replace('_', ' ').title())}: {row[c]}" for c in remaining_cols])
                        prompt = f"""Details:
{dynamic_details}
{remaining_details}"""
                    all_optimized_prompts.append(prompt)

            start_optimized_run = time.time()
            preds = []
            for i in tqdm(range(0, len(all_optimized_prompts), args.batch_size), desc=f"Task {args.task_type} Optimized"):
                batch_prompts = all_optimized_prompts[i:i + args.batch_size]
                preds.extend(query_sglang_batch(batch_prompts, args.max_new_tokens))
            optimized_runtime = time.time() - start_optimized_run
            
            optimized_results_df = df.loc[reordered_indices].copy()
            optimized_results_df['pred'] = preds
            optimized_results_df['tokens'] = [count_tokens(p) for p in all_optimized_prompts]

        total_optimized_runtime = total_ptree_build_time + optimized_runtime
        analyze_and_print_summary("PTree Optimized", optimized_results_df, all_optimized_prompts, total_optimized_runtime, args.task_type)
        
        output_file = f"eval_result_{args.task_type}_ptree_opt.csv"
        # optimized_results_df.to_csv(output_file, index=False)
        print(f"\n✅ Optimized results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM evaluation with optional PTree optimization and batching.")
    parser.add_argument("--input", default="./dataset/PDMX.csv", help="Path to the input CSV file.")
    parser.add_argument("--task_type", choices=["T1", "T2", "T3", "T4"], default="T1", help="The task type to evaluate.")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples to use (0 for all).")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens for the LLM response.")
    parser.add_argument("--batch_size", type=int, default=5120, help="Batch size for querying the LLM endpoint.")
    parser.add_argument("--use-ptree", action="store_true", help="Enable PTree partitioning optimization. If not set, only baseline is run.")
    args = parser.parse_args()
    
    if args.max_samples == 0:
        args.max_samples = None
    
    main(args)

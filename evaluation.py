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

COLUMN_LABELS = { "has_paywall": "Has Paywall", "is_rated": "Is Rated", "is_official": "Is Official", "is_draft": "Is Draft", "artist_name": "Artist", "genres": "Genres", "complexity": "Complexity", "song_length.seconds": "Song Length (seconds)" }

TASK_PROMPT_TEMPLATES = {
    "T1": lambda row, order: f"""Given the following song details, determine if this music track is suitable for public play. Answer ONLY with "Yes" or "No".

{"\n".join([f"{COLUMN_LABELS.get(col, col.replace('_', ' ').title())}: {row[col]}" for col in order if col in row])}
Title: {row['title']}
Metadata: {str(row['metadata'])}

Answer:""",
    "T2": lambda row, order: f"""Details:
{"\n".join([f"{COLUMN_LABELS.get(col, col.replace('_', ' ').title())}: {row[col]}" for col in order if col in row])}
{"\n".join([f"{COLUMN_LABELS.get(c, c.replace('_', ' ').title())}: {row[c]}" for c in row.index if c not in order and c in COLUMN_LABELS])}""",
    "T4": lambda row, order: f"""Details:
{"\n".join([f"{COLUMN_LABELS.get(col, col.replace('_', ' ').title())}: {row[col]}" for col in order if col in row])}
{"\n".join([f"{COLUMN_LABELS.get(c, c.replace('_', ' ').title())}: {row[c]}" for c in row.index if c not in order and c in COLUMN_LABELS])}""",
}

def count_tokens(text):
    if ENCODER:
        return len(ENCODER.encode(text, disallowed_special=()))
    return len(text)

def query_sglang_batch(prompts, max_new_tokens):
    if not prompts:
        return []
    payload = {"text": prompts, "max_tokens": max_new_tokens, "temperature": 0.0}
    try:
        resp = requests.post(SGLANG_URL, json=payload, timeout=60)
        resp.raise_for_status()
        results = resp.json()
        return [item.get("text", "").strip() for item in results]
    except requests.exceptions.RequestException as e:
        print(f"Warning: SGLang batch request failed: {e}")
        return ["ERROR"] * len(prompts)

def estimate_cost(total_tokens, provider="OpenAI"):
    rates = {"OpenAI": 0.0015}
    return total_tokens * rates.get(provider, 0.0) / 1000

def prefix_hit_score(prompts):
    if len(prompts) < 2: return 0.0, 0
    total_len, total_hit = 0, 0
    for i in range(1, len(prompts)):
        prefix = os.path.commonprefix([prompts[i], prompts[i - 1]])
        total_hit += len(prefix)
        total_len += len(prompts[i])
    return total_hit, total_len

def make_prompt(row, task_type, order=None):
    if task_type not in TASK_PROMPT_TEMPLATES:
        raise ValueError(f"Unsupported task: {task_type}")
    if order is None:
        order = row.index.tolist()
    return TASK_PROMPT_TEMPLATES[task_type](row, order)

def run_inference_on_dataframe(df, task_type, max_new_tokens, batch_size, order_map=None):
    all_prompts = []
    print(f"  Running inference on {len(df)} records with batch size {batch_size}...")
    
    # 根据是否有order_map（即是否为PTree优化）来生成prompts
    if order_map:
        for idx, row in df.iterrows():
            all_prompts.append(make_prompt(row, task_type, order=order_map.get(idx, df.columns.tolist())))
    else:
        for _, row in df.iterrows():
            all_prompts.append(make_prompt(row, task_type))

    all_preds = []
    for i in tqdm(range(0, len(all_prompts), batch_size), desc=f"Task {task_type}"):
        batch_prompts = all_prompts[i:i + batch_size]
        all_preds.extend(query_sglang_batch(batch_prompts, max_new_tokens))
    
    df['pred'] = all_preds
    return df, all_prompts

def analyze_and_print_summary(label, df_results, prompts, total_runtime, hit_rate):
    print(f"\n📊 {label} Evaluation Summary:")
    total_tokens = sum(count_tokens(p) for p in prompts)
    cost = estimate_cost(total_tokens)
    print(f"⏱️ Total Runtime: {total_runtime:.2f}s")
    print(f"🚀 Prefix Hit Rate: {hit_rate:.2%}")
    print(f"💰 Estimated Cost (OpenAI): ${cost:.4f} for {total_tokens:,} tokens")

def main(args):
    print(f"Loading data from {args.input}...")
    df_full = pd.read_csv(args.input)
    required_cols_map = { "T1": ["metadata", "has_paywall", "is_rated", "is_official", "is_draft", "title", "artist_name"], "T2": ["title", "artist_name", "genres", "complexity", "song_length.seconds", "metadata"], "T4": ["title", "artist_name", "genres", "complexity"] }
    required_cols = required_cols_map[args.task_type]
    df = df_full[required_cols].dropna().copy()
    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=42).copy()

    # --- Baseline (using shuffled data for fair comparison) ---
    print("\n" + "="*50)
    print("RUNNING BASELINE")
    print("="*50)
    start_time = time.time()
    df_baseline = df.sample(frac=1, random_state=42).copy()
    results_df, prompts = run_inference_on_dataframe(df_baseline, args.task_type, args.max_new_tokens, args.batch_size)
    runtime = time.time() - start_time
    hit, length = prefix_hit_score(prompts)
    hit_rate = hit / (length + 1e-5)
    analyze_and_print_summary("Baseline", results_df, prompts, runtime, hit_rate)

    # --- PTree Hybrid Strategy ---
    if args.use_ptree:
        print("\n" + "="*50)
        print("RUNNING WITH PTREE OPTIMIZATION (HYBRID STRATEGY)")
        print("="*50)
        
        ptree_decision_cols_map = { "T1": ["artist_name", "has_paywall", "is_rated", "is_official", "is_draft"], "T2": ["artist_name", "genres", "complexity", "song_length.seconds"], "T4": ["title", "artist_name", "genres", "complexity"] }
        cols_for_ptree = [c for c in ptree_decision_cols_map[args.task_type] if c in df.columns]

        print(f"Applying PTree partitioning using columns: {cols_for_ptree}...")
        df_for_ptree = df.copy()
        for col in cols_for_ptree:
            if df_for_ptree[col].dtype == 'object' or df_for_ptree[col].dtype == 'bool':
                df_for_ptree[col] = df_for_ptree[col].astype('category').cat.codes

        start_ptree_build = time.time()
        partitioner = PTreePartitioner(max_depth=10, min_rows_per_partition=30, target_batch_size=args.batch_size)
        partitioner.fit(df_for_ptree, cols_for_ptree)
        partitions = partitioner.get_partitions()
        ptree_build_time = time.time() - start_ptree_build
        print(f"PTree built in {ptree_build_time:.2f}s. Data divided into {len(partitions)} partitions.")

        reordered_indices = []
        index_to_order_map = {}
        for part in partitions:
            reordered_indices.extend(part['indices'])
            for idx in part['indices']:
                index_to_order_map[idx] = part['optimal_order']
        
        df_ptree_sorted = df.loc[reordered_indices].copy()
        
        start_inference_run = time.time()
        optimized_results_df, all_optimized_prompts = run_inference_on_dataframe(
            df_ptree_sorted, args.task_type, args.max_new_tokens, args.batch_size, order_map=index_to_order_map
        )
        inference_runtime = time.time() - start_inference_run
        
        total_optimized_runtime = ptree_build_time + inference_runtime
        
        hit, length = prefix_hit_score(all_optimized_prompts)
        optimized_hit_rate = hit / (length + 1e-5)

        analyze_and_print_summary("PTree Optimized (Hybrid)", optimized_results_df, all_optimized_prompts, total_optimized_runtime, optimized_hit_rate)
        output_file = f"eval_result_{args.task_type}_ptree_hybrid.csv"
        # optimized_results_df.to_csv(output_file, index=False)
        print(f"\n✅ Optimized results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM evaluation with PTree optimization.")
    parser.add_argument("--input", default="./dataset/PDMX.csv", help="Path to the input CSV file.")
    parser.add_argument("--task_type", choices=["T1", "T2", "T4"], default="T2", help="The task type to evaluate.")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples to use (0 for all).")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens for the LLM response.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing.")
    parser.add_argument("--use-ptree", action="store_true", help="Enable PTree partitioning optimization.")
    args = parser.parse_args()
    
    if args.max_samples == 0:
        args.max_samples = None
    
    main(args)

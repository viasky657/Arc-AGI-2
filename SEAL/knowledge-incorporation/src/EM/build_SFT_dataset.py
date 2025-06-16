# knowledge-incorporation/src/EM/build_SFT_dataset.py
"""
Convert the output of the MCMC-guided query server into a clean SFT JSONL dataset.

This script processes the JSON output from `query_server.py`, which, after
the MCMC integration, contains a single, high-quality self-edit for each article.
It extracts the prompt and the corresponding best completion into a JSONL file
suitable for Supervised Fine-Tuning (SFT).

Output:  knowledge-incorporation/data/synthetic_data/EM_SFT/sft_mcmc_search_<timestamp>.jsonl

Example usage:
    python3 knowledge-incorporation/src/EM/build_SFT_dataset.py knowledge-incorporation/results/query_server/run_*.json
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------- Helper Function ------------------------------
def get_best_completion(completions: List[Dict[str, Any]]) -> str:
    """
    Extracts the single best completion from the list.
    After the MCMC search, this list should contain exactly one item.
    """
    if not completions:
        logging.warning("Encountered an article with no completions.")
        return ""
    if len(completions) > 1:
        logging.warning(f"Expected 1 completion from MCMC search, but found {len(completions)}. Using the first one.")
    
    completion_text = completions[0].get("text", "").strip()
    if not completion_text:
        logging.warning("Found an empty completion text.")
    
    return completion_text

# ------------------------- Argument Parsing ------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an SFT dataset from MCMC-guided query server output.")
    p.add_argument("query_json", help="The JSON output file from query_server.py (e.g., run_*.json)")
    p.add_argument("--output_dir", default="knowledge-incorporation/data/synthetic_data/EM_SFT",
                   help="Destination folder for the output SFT JSONL file.")
    return p.parse_args()

# ----------------------------- Main Logic ----------------------------------
def main() -> None:
    args = parse_args()
    query_json_path = Path(args.query_json)
    output_dir = Path(args.output_dir)

    if not query_json_path.exists():
        logging.error(f"Input file not found: {query_json_path}")
        return

    logging.info(f"Loading data from {query_json_path}...")
    data: Dict[str, Any] = json.load(query_json_path.open(encoding="utf-8"))
    
    articles = data.get("articles", [])
    if not articles:
        logging.error("No 'articles' found in the input JSON file.")
        return

    timestamp = data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sft_mcmc_search_{timestamp}.jsonl"

    n_rows = 0
    n_skipped = 0
    logging.info(f"Processing {len(articles)} articles...")
    with out_path.open("w", encoding="utf-8") as fout:
        for i, article in enumerate(articles):
            prompt = article.get("prompt", "").strip()
            if not prompt:
                logging.warning(f"Article {i} ('{article.get('title', 'Untitled')}') is missing a prompt. Skipping.")
                n_skipped += 1
                continue

            best_completion = get_best_completion(article.get("completions", []))
            if not best_completion:
                logging.warning(f"Article {i} ('{article.get('title', 'Untitled')}') has no valid completion. Skipping.")
                n_skipped += 1
                continue
            
            row = {
                "prompt": prompt,
                "completion": best_completion,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_rows += 1

    logging.info(f"Processing complete.")
    logging.info(f"Wrote {n_rows} examples to → {out_path}")
    if n_skipped > 0:
        logging.warning(f"Skipped {n_skipped} articles due to missing data.")


if __name__ == "__main__":
    main()

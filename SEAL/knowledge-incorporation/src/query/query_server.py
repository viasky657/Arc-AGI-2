# src/query/query_server.py
"""
Query TTT server on SQuAD synthetic data. This drives the inner-loop TTT server:
sample k synthetic completions per article, run `eval_times` fine-tune+eval 
cycles, and write an aggregated JSON report. This is used in both ReST-EM 
training and evaluation with n=1.

The results are written to a JSON file. The overall summary contains:
    baseline_mean_accuracy                - mean of article-level baseline means
    baseline_std_of_article_means         - std-dev across those means
    adapter_mean_accuracy                 - mean of article-level adapter means
    adapter_std_of_article_means          - std-dev across those means
    mean_adapter_std_over_completions     - average per-article adapter std-dev 
                                            (across the k completions);
                                            higher ⇒ more variance the RL selector
                                            can exploit
    mean_adapter_std_within_completions   - avg std-dev within each completion,
                                            across TTT eval runs; 
                                            lower ⇒ more stable signal
    mean_gain                             - adapter_mean - baseline_mean
"""
import argparse
import datetime as _dt
import json
import pathlib
import statistics as _stats
import sys
import zmq
import random
from typing import Any, Dict, List

from ..utils import build_train_sequences
from SEAL.mcmc_search import MCMCConfig, SelfEditSpace, JudgedNeuralSurrogate, OpenRouterJudge, metropolis_hastings_sampler

# -------------------------- ARGPARSE / CONFIG ------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="rank_iter0")
    p.add_argument("--dataset", default="data/synthetic_data/train/iter0_train.json")
    p.add_argument("--output_dir", default="results/query_server")
    p.add_argument("--server_host", default="127.0.0.1")
    p.add_argument("--zmq_port", type=int, default=5555)

    p.add_argument("--n_articles", type=int, default=3)
    p.add_argument("--k_completions", type=int, default=5)
    p.add_argument("--eval_times", type=int, default=3)

    # LoRA / optimisation hyper-params
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--finetune_lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--end_mask_substring", default="")
    p.add_argument("--split_newlines", action="store_true")
    return p.parse_args()


def send_round_trip(
    ctx: zmq.Context,
    endpoint: str,
    train_sequences: List[str],
    questions: List[Dict[str, str]],
    args: argparse.Namespace,
    max_retries: int = 2,
    timeout_ms: int = 600_000,           # 10 minutes
) -> Dict[str, Any]:
    """
    Send one request to the TTT server and wait (≤10 min) for its reply.
    On timeout we recreate the REQ socket and retry, up to `max_retries`.
    """
    # you'll recreate the socket on each retry, so wrap the whole thing
    # in a loop.
    for attempt in range(1, max_retries + 1):
        # ------ (re)create a REQ socket ------
        sock = ctx.socket(zmq.REQ)
        sock.connect(endpoint)
        sock.setsockopt(zmq.LINGER, 0)   # don't block on close

        # register this socket with a poller
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        # build & send the request -------------------------
        payload = {
            "train_sequences": train_sequences,
            "eval_questions": questions,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "finetune_epochs": args.finetune_epochs,
            "finetune_lr": args.finetune_lr,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "end_mask_substring": args.end_mask_substring,
        }

        sock.send_json(payload)
        print("Sent request to TTT server "
              f"(attempt {attempt}/{max_retries}, waiting ≤10 min)")

        # ------------------ wait for the reply --------------------
        events = poller.poll(timeout_ms)
        if events:                       # reply arrived in time
            reply = sock.recv_json()
            sock.close()
            if "error" not in reply:
                return reply
            print(f"TTT server error: {reply['error']}")   # fall through → retry
        else:                            # timed out
            print("No reply after 10 minutes - retrying…")

        # clean up before the next attempt
        poller.unregister(sock)
        sock.close()

    # if we reach here every attempt failed
    raise RuntimeError(f"TTT server unreachable after {max_retries} attempts")

def evaluate_completion(ctx, endpoint, item: Dict[str, Any], comp_raw: str, args):
    """Run `eval_times` fine-tune / eval cycles for one completion."""
    title, context = item["title"], item["context"]
    questions = [
        {
            "title": title,
            "context": context,
            "question": f"Topic: {title}\n{q['question']}",
            "answer": q["answer"],
        }
        for q in item["questions"]
    ]
    train_sequences = build_train_sequences(comp_raw, context, title, split_newlines=args.split_newlines)

    base_accs, adpt_accs, gains = [], [], []
    q_details: List[Dict[str, Any]] = []

    for i in range(args.eval_times):
        rep = send_round_trip(ctx, endpoint, train_sequences, questions, args)

        base_accs.append(rep["baseline_accuracy"])
        adpt_accs.append(rep["adapter_accuracy"])
        gains.append(rep["adapter_gain"])

        q_details_rep = []
        for qi, q in enumerate(item["questions"]):
            q_details_rep.append(
                {
                    "rep": i,
                    "question": q["question"],
                    "answer": q["answer"],
                    "baseline_answer": rep["baseline_texts"][qi],
                    "adapter_answer":  rep["adapter_texts"][qi],
                    "baseline_correct": rep["baseline_correct"][qi],
                    "adapter_correct":  rep["adapter_correct"][qi],
                }
            )
        q_details.extend(q_details_rep)


    stats_only = {
        "baseline_mean": _stats.mean(base_accs),
        "baseline_std": _stats.stdev(base_accs) if len(base_accs) > 1 else 0.0,
        "adapter_mean": _stats.mean(adpt_accs),
        "adapter_std": _stats.stdev(adpt_accs) if len(adpt_accs) > 1 else 0.0,
        "gain_mean": _stats.mean(gains),
    }

    return stats_only, q_details

def send_shutdown(sock):
    sock.send_json({"cmd": "shutdown"})
    sock.recv_json()           # consume the "bye"

# ------------------------------- MAIN ------------------------------- #
def main() -> None:
    args = parse_args()

    data_path = pathlib.Path(args.dataset)
    try:
        dataset: List[Dict[str, Any]] = json.load(data_path.open(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"[!] Dataset not found: {data_path}")

    dataset = dataset[: args.n_articles] if args.n_articles else dataset

    out_dir = pathlib.Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ctx = zmq.Context()
    endpoint = f"tcp://{args.server_host}:{args.zmq_port}"

    articles_out: List[Dict[str, Any]] = []
    overall_base_means, overall_adpt_means, adapter_std_list, completion_run_std_list = [], [], [], []

    # --- Setup for MCMC Search with Co-piloted Learning ---
    mcmc_config = MCMCConfig()
    output_space = SelfEditSpace()
    # The judged surrogate contains the neural surrogate and the external judge
    judge = OpenRouterJudge()
    surrogate_model = JudgedNeuralSurrogate(judge)
    # ---

    # ---------------- iterate over articles -------------------------- #
    for art_idx, item in enumerate(dataset):
        # Use the first available completion as the starting point for our search.
        initial_edit_text = (item.get("completions") or [item.get("completion", "")])[0].strip()
        if not initial_edit_text:
            print(f"[{art_idx:02d}] Skipping article with no initial text.")
            continue

        print(f"[{art_idx:02d}] Starting MCMC search to find best self-edit...")
        # Use the MCMC sampler to find the most promising self-edit candidate.
        best_edit_text = metropolis_hastings_sampler(
            initial_state=initial_edit_text,
            surrogate_func=surrogate_model,
            output_space=output_space,
            config=mcmc_config
        )
        print(f"[{art_idx:02d}] MCMC search finished. Evaluating best candidate found.")

        # Evaluate the single best completion found by the search.
        stats, q_details = evaluate_completion(ctx, endpoint, item, best_edit_text, args)
        
        # "Train" the surrogate model with the true reward from the evaluation.
        surrogate_model.update(best_edit_text, stats['adapter_mean'])

        print(f"[{art_idx:02d}] MCMC Candidate -> "
              f"base {stats['baseline_mean']*100:.2f}% | "
              f"adapter {stats['adapter_mean']*100:.2f}% ± {stats['adapter_std']*100:.2f}% "
              f"gain {stats['gain_mean']*100:+.2f}%")

        # Since we only have one result per article now, the article-level stats are just these stats.
        base_mean_article = stats['baseline_mean']
        adpt_mean_article = stats['adapter_mean']
        adpt_std_article = 0.0  # No standard deviation over a single completion.
        mean_run_std_article = stats['adapter_std']
        gain_mean_article = stats['gain_mean']

        overall_base_means.append(base_mean_article)
        overall_adpt_means.append(adpt_mean_article)
        adapter_std_list.append(adpt_std_article)
        completion_run_std_list.append(mean_run_std_article)

        # The comp_entries list will now only have one entry.
        comp_entries = [{
            "text": best_edit_text,
            "stats": stats,
            "questions": q_details
        }]

        cur_base  = _stats.mean(overall_base_means)
        cur_adapt = _stats.mean(overall_adpt_means)
        cur_gain  = cur_adapt - cur_base

        print(f"[progress] overall ({len(overall_base_means)} articles)  "
              f"baseline {cur_base*100:.2f}% | "
              f"adapter {cur_adapt*100:.2f}% | "
              f"gain {cur_gain*100:+.2f}%")

        articles_out.append(
            {
                "stats": {
                    "baseline_accuracy": round(base_mean_article, 4),
                    "adapter_mean_accuracy": round(adpt_mean_article, 4),
                    "adapter_std_over_completions": round(adpt_std_article, 4),
                    "mean_adapter_std_within_completions": round(mean_run_std_article, 4),
                    "mean_gain": round(gain_mean_article, 4),
                },
                "title": item["title"],
                "context": item["context"],
                "completions": comp_entries,
                "prompt":item.get("prompt", ""),
            }
        )

    # ------------- overall summary & JSON write ---------------------- #
    overall_base_mean = _stats.mean(overall_base_means)  if overall_base_means else 0.0
    overall_base_std  = _stats.stdev(overall_base_means) if len(overall_base_means) > 1 else 0.0
    overall_adpt_mean = _stats.mean(overall_adpt_means)  if overall_adpt_means else 0.0
    overall_adpt_std  = _stats.stdev(overall_adpt_means) if len(overall_adpt_means) > 1 else 0.0
    mean_adapter_std_accuracy = _stats.mean(adapter_std_list) if adapter_std_list else 0.0
    mean_adapter_std_within_completion = _stats.mean(completion_run_std_list) if completion_run_std_list else 0.0

    ts = _dt.datetime.now().strftime("%m%d_%H%M%S")
    out_path = out_dir / f"run_{ts}.json"
    json.dump(
        {
            "overall": {
                "baseline_mean_accuracy": round(overall_base_mean, 4),
                "baseline_std_of_article_means": round(overall_base_std, 4),
                "adapter_mean_accuracy": round(overall_adpt_mean, 4),
                "adapter_std_of_article_means": round(overall_adpt_std, 4),
                "mean_adapter_std_over_completions": round(mean_adapter_std_accuracy, 4),
                "mean_adapter_std_within_completions": round(mean_adapter_std_within_completion, 4),
                "mean_gain": round(overall_adpt_mean - overall_base_mean, 4),
            },
            "timestamp": ts,
            "exp_name": args.exp_name,
            "dataset": str(data_path),
            "split_newlines": args.split_newlines,
            "n_articles": len(articles_out),
            "k_completions": args.k_completions,
            "eval_times": args.eval_times,
            "lora_params": {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "epochs": args.finetune_epochs,
                "lr": args.finetune_lr,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.gradient_accumulation_steps,
            },
            "articles": articles_out,
        },
        out_path.open("w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    print(f"\nWrote summary → {out_path}")

    # Optional: tell the server to shut down when finished
    # send_shutdown(ctx, endpoint)
    ctx.term()


if __name__ == "__main__":
    main()

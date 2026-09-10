"""Run the authors' released Amazon Books protocol with checkpoint/resume support."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from .evaluation import ranking_metrics
from .official_protocol import OfficialProtocolAgent, append_result, completed_rows, load_official_examples, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/instructrec"))
    parser.add_argument("--agent", choices=("static", "dynamic"), default="static")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default=os.environ.get("IAGENT_OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--output", type=Path, default=Path("outputs/official_protocol/books_static_api.jsonl"))
    parser.add_argument("--protocol-mode", choices=("strict", "compact"), default="strict",
                        help="strict preserves the authors' prompts and explanation field; compact is Qwen-only.")
    parser.add_argument("--json-mode", choices=("json_object", "json_schema"), default="json_object",
                        help="Provider transport. DeepSeek supports json_object; json_schema is for local vLLM.")
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Request Qwen3 non-thinking mode through vLLM's chat-template kwargs.")
    parser.add_argument("--bounded-explanations", action="store_true",
                        help="Constrain only non-metric explanations in JSON schema to prevent local-model truncation.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10, help="Use --all for every released Books row.")
    parser.add_argument("--all", action="store_true", help="Run all 7,377 released Books rows.")
    parser.add_argument("--seed", type=int, default=2025, help="Controls i2Agent's feedback negative sampling.")
    args = parser.parse_args()
    key = os.environ.get("IAGENT_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Set IAGENT_API_KEY (or DEEPSEEK_API_KEY for DeepSeek) in the server environment.")
    limit = None if args.all else args.limit
    records, mapping = load_official_examples(args.data, start=args.start, limit=limit)
    done = completed_rows(args.output)
    agent = OfficialProtocolAgent(model=args.model, base_url=args.base_url, api_key=key,
                                  agent_type=args.agent, rng=__import__("random").Random(args.seed),
                                  protocol_mode=args.protocol_mode, json_mode=args.json_mode,
                                  disable_thinking=args.disable_thinking,
                                  bounded_explanations=args.bounded_explanations)
    print(f"protocol=authors-public-code-compatible/{args.protocol_mode} json={args.json_mode} "
          f"agent={args.agent} candidates=10 records={len(records)} "
          f"resuming={len(done)} output={args.output}", flush=True)
    for ordinal, record in enumerate(records, 1):
        if record.row_index in done:
            continue
        started = time.perf_counter()
        try:
            ranked = agent.rank(record, mapping)
            metric = ranking_metrics([str(item) for item in ranked], str(record.target_id))
            append_result(args.output, {"status": "ok", "row_index": record.row_index, "user_id": record.user_id,
                                        "target_id": record.target_id, "ranked": ranked, "metrics": metric,
                                        "seconds": round(time.perf_counter() - started, 3)})
        except Exception as exc:
            append_result(args.output, {"status": "error", "row_index": record.row_index, "user_id": record.user_id,
                                        "error": repr(exc), "seconds": round(time.perf_counter() - started, 3)})
        if ordinal % 10 == 0:
            print(summarize(args.output), flush=True)
    print(summarize(args.output), flush=True)


if __name__ == "__main__":
    main()

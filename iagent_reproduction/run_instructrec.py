"""Reproduce the released InstructRec evaluation path on a bounded user sample."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .backends import QwenBackend, TeachingBackend
from .core import I2Agent, IAgent
from .evaluation import average, ranking_metrics
from .instructrec import DOMAIN_FILES, load_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/instructrec"))
    parser.add_argument("--domain", choices=sorted(DOMAIN_FILES), default="books")
    parser.add_argument("--agent", choices=("iagent", "i2agent"), default="iagent")
    parser.add_argument("--backend", choices=("teaching", "qwen"), default="teaching")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--base-url", default=os.environ.get("IAGENT_OPENAI_BASE_URL"),
                        help="OpenAI-compatible endpoint; defaults to DashScope when unset.")
    parser.add_argument("--max-static-chars", type=int, default=6000,
                        help="Deterministic static-memory prompt budget for Qwen.")
    parser.add_argument("--max-candidate-chars", type=int, default=600,
                        help="Deterministic per-candidate prompt budget for Qwen.")
    parser.add_argument("--users", type=int, default=5)
    args = parser.parse_args()
    backend = (QwenBackend(model=args.model, base_url=args.base_url,
                           max_static_chars=args.max_static_chars,
                           max_candidate_chars=args.max_candidate_chars)
               if args.backend == "qwen" and args.base_url else
               QwenBackend(model=args.model, max_static_chars=args.max_static_chars,
                           max_candidate_chars=args.max_candidate_chars)
               if args.backend == "qwen" else TeachingBackend())
    rows = []
    for example in load_examples(args.data, args.domain, args.users):
        agent = I2Agent(backend) if args.agent == "i2agent" else IAgent(backend)
        if isinstance(agent, I2Agent):
            agent.fit_from_history(example.history, example.negative_pool)
        ranked = agent.rank(example.history, example.instruction, example.candidates)
        rows.append(ranking_metrics(ranked, example.target_id))
    print(f"domain={args.domain} agent={args.agent} backend={args.backend} users={len(rows)} "
          f"max_static_chars={args.max_static_chars} max_candidate_chars={args.max_candidate_chars}")
    print(average(rows))


if __name__ == "__main__":
    main()

"""Reproduce the released InstructRec evaluation path on a bounded user sample."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--users", type=int, default=5)
    args = parser.parse_args()
    backend = QwenBackend(args.model) if args.backend == "qwen" else TeachingBackend()
    rows = []
    for example in load_examples(args.data, args.domain, args.users):
        agent = I2Agent(backend) if args.agent == "i2agent" else IAgent(backend)
        if isinstance(agent, I2Agent):
            agent.fit_from_history(example.history, example.negative_pool)
        ranked = agent.rank(example.history, example.instruction, example.candidates)
        rows.append(ranking_metrics(ranked, example.target_id))
    print(f"domain={args.domain} agent={args.agent} backend={args.backend} users={len(rows)}")
    print(average(rows))


if __name__ == "__main__":
    main()

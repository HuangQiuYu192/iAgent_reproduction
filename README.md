# iAgent_reproduction

An educational, runnable reproduction of **iAgent: LLM Agent as a Shield between User and Recommender Systems** (Xu et al., ACL Findings 2025).

It implements the paper's inference structure and the personal feedback loop of i²Agent:

- **iAgent:** instruction parser -> optional knowledge retrieval -> LLM reranker -> self-reflection.
- **i²Agent:** individual profile generator over feedback -> instruction-relative dynamic extractor -> reranker -> self-reflection.
- **Evaluation:** leave-one-out next-item prediction, one positive plus nine seeded negatives, HR@1/3, NDCG@3, and MRR.

The default backend is deterministic and runs without an API key, so every transition is inspectable. A Qwen/DashScope OpenAI-compatible backend is included for the actual LLM calls.

## Why this is a functional reproduction, not yet a claims-level replication

The paper's reported results use the authors' InstructRec splits and GPT-4o-mini. Exact results require their download, the same generated/filtered instructions, candidate samples, prompts, and model version. This repository faithfully exposes the published algorithmic interfaces and evaluation protocol, but does not claim to reproduce Tables 2-5 until that data is installed.

## Quick start

```powershell
cd iAgent_reproduction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
python -m iagent_reproduction.demo
pytest -q
```

## Use Qwen through API

```powershell
pip install -e .[qwen]
$env:DASHSCOPE_API_KEY = 'your-key-here'
```

Do not commit the key. The Qwen client uses the DashScope OpenAI-compatible endpoint and forces `temperature=0`; invoke it by constructing `QwenBackend` in your experiment script.

## Data contract

The runner accepts JSONL records in this minimal form:

```json
{"user_id":"u1","item_id":"b1","title":"...","text":"item description or review-derived text","timestamp":1,"instruction":"free-text user request"}
```

For a strict reproduction, download the four InstructRec releases linked by the [authors' repository](https://github.com/WujiangXu/iAgent), then write a narrow converter that preserves the provided instructions exactly. Do not regenerate them if you want a direct comparison.

## Learning map

1. Start with `demo.py`: parser outputs instruction knowledge; reranker must return a permutation of the platform slate.
2. Observe `I2Agent.learn_feedback`: only the focal user's feedback updates their profile—there is no collaborative user update.
3. Compare `I2Agent.rank` with `IAgent.rank`: the former adds dynamic profile and dynamic interest that are conditional on the current instruction.
4. Run evaluation before changing prompts. Prompt/model/data changes are experimental variables and must be logged.

## Paper terminology

| Term | Meaning in this code |
|---|---|
| `X_I` | User's current free-text instruction |
| `X_IK`, `X_EK` | Parser's internal and optional external knowledge |
| `X_SU` | Static history memory |
| `F_T` | Individual profile after feedback round `T` |
| `F_d^T`, `X_DU` | Instruction-conditioned dynamic profile and interest |
| Self-reflection | Validate that reranking returns exactly the original candidate IDs |

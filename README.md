# iAgent_reproduction

An educational, runnable reproduction of **iAgent: LLM Agent as a Shield between User and Recommender Systems** (Xu et al., ACL Findings 2025).

It implements the paper's inference structure and the personal feedback loop of i²Agent:

- **iAgent:** instruction parser -> optional knowledge retrieval -> LLM reranker -> self-reflection.
- **i²Agent:** individual profile generator over feedback -> instruction-relative dynamic extractor -> reranker -> self-reflection.
- **Evaluation:** leave-one-out next-item prediction, one positive plus nine seeded negatives, HR@1/3, NDCG@3, and MRR.

The default backend is deterministic and runs without an API key, so every transition is inspectable. A Qwen OpenAI-compatible backend supports both DashScope and a self-hosted vLLM server.

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

## Use Qwen through API or a local 3090

```powershell
pip install -e .[qwen]
$env:DASHSCOPE_API_KEY = 'your-key-here'
```

Do not commit the key. The Qwen client uses the DashScope OpenAI-compatible endpoint and forces `temperature=0`; invoke it by constructing `QwenBackend` in your experiment script.

For repeated experiments, run Qwen locally on the Jupyter server's RTX 3090 rather than paying per API token. The repository includes installation, service, and health-check scripts plus the exact experiment command in [docs/qwen_3090.md](docs/qwen_3090.md). Once the local server is running, use:

```bash
export IAGENT_API_KEY='the-same-local-secret'
export IAGENT_OPENAI_BASE_URL='http://127.0.0.1:8000/v1'
python -m iagent_reproduction.run_instructrec --domain books --agent iagent --backend qwen --model Qwen/Qwen2.5-7B-Instruct --users 1
```

## Data contract

The runner accepts JSONL records in this minimal form:

```json
{"user_id":"u1","item_id":"b1","title":"...","text":"item description or review-derived text","timestamp":1,"instruction":"free-text user request"}
```

For a strict reproduction, download the four InstructRec releases linked by the [authors' repository](https://github.com/WujiangXu/iAgent), then write a narrow converter that preserves the provided instructions exactly. Do not regenerate them if you want a direct comparison.

The official release is read directly by the included adapter. Its `.pkl` and `.csv` files belong in `data/instructrec/` and are intentionally Git-ignored:

```powershell
pip install -e .[instructrec]
python -m iagent_reproduction.run_instructrec --domain books --agent iagent --backend teaching --users 5
python -m iagent_reproduction.run_instructrec --domain books --agent i2agent --backend teaching --users 5
```

## Strict API protocol: DeepSeek V4 Flash

`run_official_protocol` has two deliberately separate modes. `strict` preserves the authors' public prompts, free-text `knowledge`, and `explanation` response field; it uses a 1,024-token per-call safety ceiling only to prevent abnormal billing. `compact` is the earlier local-Qwen compatibility baseline and must not be reported as a strict reproduction.

On A40, keep the real key outside the repository and run the 100-user pilot:

```bash
export DEEPSEEK_API_KEY='your-private-key'
cd /home/hqy/code/iAgent_reproduction
bash scripts/run_books_iagent_deepseek_100.sh
```

The script calls `deepseek-v4-flash` through `https://api.deepseek.com`, writes resumable JSONL results to `outputs/official_protocol/books_static_deepseek_v4_flash_100.jsonl`, and never reads a key from a file or Git.

Released item descriptions and reviews can exceed a local 7B model's context window. The Qwen runner therefore logs deterministic prompt budgets (`--max-static-chars 6000` and `--max-candidate-chars 600`). Treat them as experimental settings and report them with every result; increase only after verifying that the local server remains stable.

Use Qwen only after this smoke test passes. i²Agent runs one profile-update LLM call per training interaction, so begin with one user:

```powershell
python -m iagent_reproduction.run_instructrec --domain books --agent i2agent --backend qwen --model qwen-plus --users 1
```

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

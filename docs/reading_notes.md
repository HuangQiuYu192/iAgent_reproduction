# iAgent paper reading notes

Source: Xu et al., *iAgent: LLM Agent as a Shield between User and Recommender Systems*, ACL Findings 2025 (user-supplied PDF).

## Reproduction contract

- **Input:** a user history, a current free-text instruction, and the platform's candidate slate.
- **iAgent (§2.1):** parser produces internal knowledge, tool decision, and keywords; optional external knowledge is retrieved; reranker consumes these, `X_SU`, and candidate text. Self-reflection rejects an output that is not aligned to the original slate.
- **i²Agent (§2.2):** profile generator updates `F^T` after individual positive feedback against a sampled negative; extractor produces `F_d^T` and `X_DU` conditioned on the current instruction; reranker then includes them.
- **Protocol (§3.1):** hold out each user's last interaction, rank it with nine sampled negatives, report HR@1/3, NDCG@3, and MRR.

## Terminology ledger

| Canonical term | First-use definition | Decision |
|---|---|---|
| iAgent | Instruction-aware Agent | Static-history, instruction-aware reranker. |
| i²Agent | Individual Instruction-aware Agent | Dynamic personal-profile extension; written `I2Agent` in code. |
| InstructRec | User-instruction recommendation benchmark | Preserve supplied instructions; never replace for strict comparison. |
| `X_SU` | Static user memory | Text serialization of the user's own history only. |
| `F^T` | Profile after feedback round T | Individual state only. |
| `F_d^T`, `X_DU` | Dynamic profile and dynamic interest | Conditioned on the current instruction. |

## Fidelity limits

The paper names GPT-4o-mini for dataset generation and describes zero-shot LLM reranking, but full reproducibility still depends on the authors' released InstructRec files, exact candidate slates, and original LLM/provider behavior. The code labels Qwen results as a backbone substitution rather than the paper's reported configuration.

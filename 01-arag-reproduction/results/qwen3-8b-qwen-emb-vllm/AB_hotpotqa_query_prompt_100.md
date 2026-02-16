# HotpotQA 100-sample A/B: Qwen3 query prompt usage

## Setup
- Same model/index: `Qwen3-8B` generation + `Qwen3-Embedding-0.6B` index
- Difference only in semantic query encoding:
  - `prompt_on`: `ARAG_USE_QUERY_PROMPT=1` (uses `prompt_name="query"`)
  - `prompt_off`: `ARAG_USE_QUERY_PROMPT=0`
- Questions: first 100 from `hotpotqa/questions.json`
- Eval judge: `Qwen3-30B-A3B`

Jobs:
- Gen on: `19545527`, off: `19545528`
- Eval on: `19546901`, off: `19546902`

## Results

| Variant | LLM-Acc | Cont-Acc | Avg Loops | Avg Retrieved Tokens |
|---|---:|---:|---:|---:|
| prompt_on | 0.50 | 0.56 | 2.44 | 663.1 |
| prompt_off | 0.47 | 0.60 | 2.63 | 799.0 |

Delta (`prompt_on - prompt_off`):
- LLM-Acc: **+0.03**
- Cont-Acc: **-0.04**
- Avg Loops: **-0.19**
- Avg Retrieved Tokens: **-135.9**

## Takeaway
Using Qwen's query prompt makes retrieval more efficient (fewer loops/tokens) and slightly improves LLM-judged accuracy on this 100-sample slice, but lowers contain-based exactness.

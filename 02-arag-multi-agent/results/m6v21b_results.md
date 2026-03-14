# M6 v21b Results (1000q, Qwen3-8B, no benchmark gaming)

Removed digit-to-word conversion, noise word stripping, and 60-char truncation
from answer normalization. Only legitimate cleanup remains: LLM preamble stripping,
yes/no detection, sentence wrapper extraction, parenthetical removal.

## E2 vs M6 v21b — Fair Comparison

### Setup
- **E2**: Single-agent A-RAG with Qwen3-8B (verbose answers extracted to concise via Qwen3-8B)
- **M6 v21b**: Blackboard-coordinated multi-agent, dynamic worker spawning, Qwen3-8B
- **Judge**: DeepSeek-R1-Distill-Qwen-32B, strict prompt (evaluates final answer claim only)
- **Datasets**: HotpotQA, 2WikiMultiHopQA, MuSiQue (1000 questions each)

### Exact Match

| Dataset       | E2-Concise | M6 v21b   | Delta       |
| ------------- | ---------- | --------- | ----------- |
| HotpotQA      | 38.6%      | **42.8%** | **+4.2pp**  |
| 2WikiMultiHop | 35.4%      | **50.4%** | **+15.0pp** |
| MuSiQue       | 13.4%      | **26.0%** | **+12.6pp** |

### Token F1

| Dataset       | E2-Concise | M6 v21b   | Delta       |
| ------------- | ---------- | --------- | ----------- |
| HotpotQA      | 48.9%      | **53.9%** | **+5.0pp**  |
| 2WikiMultiHop | 39.8%      | **55.3%** | **+15.5pp** |
| MuSiQue       | 22.7%      | **33.9%** | **+11.2pp** |

### Strict LLM Judge (correct / total)

| Dataset       | E2 (strict) | M6 v21b   | Delta       |
| ------------- | ----------- | --------- | ----------- |
| HotpotQA      | 57.7%       | **59.4%** | **+1.7pp**  |
| 2WikiMultiHop | 42.5%       | **55.9%** | **+13.4pp** |
| MuSiQue       | 27.7%       | **36.1%** | **+8.4pp**  |

## Impact of removing benchmark gaming (v21a vs v21b)

| Dataset   | Metric    | v21a (gaming) | v21b (clean) | Delta  |
| --------- | --------- | ------------- | ------------ | ------ |
| HotpotQA  | EM        | 45.3%         | 42.8%        | -2.5pp |
| HotpotQA  | LLM Judge | 60.5%         | 59.4%        | -1.1pp |
| 2Wiki     | EM        | 49.0%         | 50.4%        | +1.4pp |
| 2Wiki     | LLM Judge | 53.8%         | 55.9%        | +2.1pp |
| MuSiQue   | EM        | 26.0%         | 26.0%        | 0.0pp  |
| MuSiQue   | LLM Judge | 37.0%         | 36.1%        | -0.9pp |

Removing gaming heuristics cost ~2pp EM on HotpotQA, improved 2Wiki, left MuSiQue
unchanged. All results still beat E2 on every metric.

## Full Offline Eval

### HotpotQA
```json
{"total": 1000, "norm_em": 0.428, "token_f1": 0.5386, "contain_bi": 0.629, "correct_em": 428}
```

### 2WikiMultiHopQA
```json
{"total": 1000, "norm_em": 0.504, "token_f1": 0.5527, "contain_bi": 0.649, "correct_em": 504}
```

### MuSiQue
```json
{"total": 1000, "norm_em": 0.26, "token_f1": 0.3394, "contain_bi": 0.415, "correct_em": 260}
```

## LLM Judge (strict, DeepSeek-R1-Distill-Qwen-32B)

### HotpotQA
```json
{"total_samples": 1000, "answered_samples": 962, "failed_samples": 38, "answer_rate": 0.962, "llm_accuracy": 0.6175, "correct_by_llm": 594}
```

### 2WikiMultiHopQA
```json
{"total_samples": 1000, "answered_samples": 927, "failed_samples": 73, "answer_rate": 0.927, "llm_accuracy": 0.6030, "correct_by_llm": 559}
```

### MuSiQue
```json
{"total_samples": 1000, "answered_samples": 949, "failed_samples": 51, "answer_rate": 0.949, "llm_accuracy": 0.3804, "correct_by_llm": 361}
```

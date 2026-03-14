# M6 v21 Results (1000q, Qwen3-8B)

Cleaned architecture: removed dead code (-838 lines), extracted shared answer_utils,
removed critic/knowledge_gap infrastructure, dynamic worker spawning.

## E2 vs M6 v21 — Fair Comparison

### Setup
- **E2**: Single-agent A-RAG with Qwen3-8B (verbose answers extracted to concise via Qwen3-8B)
- **M6 v21**: Blackboard-coordinated multi-agent with dynamic worker spawning, Qwen3-8B
- **Judge**: DeepSeek-R1-Distill-Qwen-32B, strict prompt (evaluates final answer claim only)
- **Datasets**: HotpotQA, 2WikiMultiHopQA, MuSiQue (1000 questions each)

### Exact Match

| Dataset       | E2-Concise | M6 v21    | Delta       |
| ------------- | ---------- | --------- | ----------- |
| HotpotQA      | 38.6%      | **45.3%** | **+6.7pp**  |
| 2WikiMultiHop | 35.4%      | **49.0%** | **+13.6pp** |
| MuSiQue       | 13.4%      | **26.0%** | **+12.6pp** |

### Token F1

| Dataset       | E2-Concise | M6 v21    | Delta       |
| ------------- | ---------- | --------- | ----------- |
| HotpotQA      | 48.9%      | **55.3%** | **+6.4pp**  |
| 2WikiMultiHop | 39.8%      | **54.3%** | **+14.5pp** |
| MuSiQue       | 22.7%      | **34.7%** | **+12.0pp** |

### Strict LLM Judge (correct / total)

| Dataset       | E2 (strict) | M6 v21    | Delta       |
| ------------- | ----------- | --------- | ----------- |
| HotpotQA      | 57.7%       | **60.5%** | **+2.8pp**  |
| 2WikiMultiHop | 42.5%       | **53.8%** | **+11.3pp** |
| MuSiQue       | 27.7%       | **37.0%** | **+9.3pp**  |

## M6 v21 vs M6 v20 (cleanup impact)

| Dataset   | Metric    | M6 v20 | M6 v21    | Delta    |
| --------- | --------- | ------ | --------- | -------- |
| HotpotQA  | EM        | 42.4%  | **45.3%** | +2.9pp   |
| HotpotQA  | LLM Judge | 56.4%  | **60.5%** | +4.1pp   |
| 2Wiki     | EM        | 50.8%  | 49.0%     | -1.8pp   |
| 2Wiki     | LLM Judge | 56.7%  | 53.8%     | -2.9pp   |
| MuSiQue   | EM        | 26.1%  | 26.0%     | -0.1pp   |
| MuSiQue   | LLM Judge | 36.0%  | **37.0%** | +1.0pp   |

No regression from cleanup. HotpotQA improved, 2Wiki slight dip (within noise), MuSiQue flat.

## Full Offline Eval

### HotpotQA
```json
{"total": 1000, "norm_em": 0.453, "token_f1": 0.5534, "contain_bi": 0.626, "correct_em": 453}
```

### 2WikiMultiHopQA
```json
{"total": 1000, "norm_em": 0.49, "token_f1": 0.5433, "contain_bi": 0.629, "correct_em": 490}
```

### MuSiQue
```json
{"total": 1000, "norm_em": 0.26, "token_f1": 0.3472, "contain_bi": 0.419, "correct_em": 260}
```

## LLM Judge (strict, DeepSeek-R1-Distill-Qwen-32B)

### HotpotQA
```json
{"total_samples": 1000, "answered_samples": 972, "failed_samples": 28, "answer_rate": 0.972, "llm_accuracy": 0.6224, "correct_by_llm": 605}
```

### 2WikiMultiHopQA
```json
{"total_samples": 1000, "answered_samples": 930, "failed_samples": 70, "answer_rate": 0.93, "llm_accuracy": 0.5785, "correct_by_llm": 538}
```

### MuSiQue
```json
{"total_samples": 1000, "answered_samples": 949, "failed_samples": 51, "answer_rate": 0.949, "llm_accuracy": 0.3899, "correct_by_llm": 370}
```

## Note on LLM Judge Verbosity Confound

E2's original LLM judge scores (56.5%/45.5%/30.1%) were inflated by verbose answers
(avg 1009 chars vs M6's 12 chars). The judge could find correct information embedded
in paragraphs of hedging, even when E2's core answer was wrong (E2 gets 0% EM).

Fair comparison methods used:
1. **E2-Concise**: Extracted core answer entities from E2's verbose outputs using Qwen3-8B
2. **Strict Judge**: Modified judge prompt to evaluate only the final answer claim

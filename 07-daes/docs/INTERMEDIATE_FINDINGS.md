# Intermediate Findings — April 2, 2026

## Finding 1: DNMR Retrieval Works on LLaDA (Contain Analysis)

Pool expansion retrieves the correct answer in significantly more questions than baseline on BOTH models.

### LLaDA MuSiQue (N=740)

| Category | Count | Pct |
|----------|:-----:|:---:|
| Both baseline and pool find gold | 84 | 11.4 |
| **ONLY pool finds gold (retrieval helped)** | **81** | **10.9** |
| ONLY baseline finds gold (retrieval hurt) | 9 | 1.2 |
| Neither finds gold | 566 | 76.5 |

Pool contain: 22.3% vs Baseline contain: 12.6%. Retrieval finds gold in 72 extra questions.

### Dream MuSiQue (N=1000)

| Category | Count | Pct |
|----------|:-----:|:---:|
| Both find gold | 111 | 11.1 |
| ONLY pool finds gold | 73 | 7.3 |
| ONLY baseline finds gold | 10 | 1.0 |
| Neither | 806 | 80.6 |

### Pool vs ARAM on LLaDA (740 matched)

| Metric | Pool (DNMR) | ARAM |
|--------|:-----------:|:----:|
| F1 | 0.107 | 0.191 |
| Contain | 22.3% | 11.4% |
| Avg answer length | 110 chars | 29 chars |
| Per-question F1 wins | 145 | 139 |
| Finds gold in extra Qs | 90 | 9 |

Pool wins on retrieval quality. ARAM wins on F1 purely from conciseness.

### Verbosity Is the Bottleneck

On 81 questions where ONLY pool finds gold: Pool F1=0.166, Baseline F1=0.135. Answer is there but buried.

Answer lengths: Dream pool 16.7 chars, LLaDA pool 112.6 chars. Pool expansion makes LLaDA verbose.

## Finding 2: Bridge Candidate Quality

LLaDA: 30% of candidates start with "The answer is..." (answer guesses, not bridges).
Dream: 1% start with "The answer is..." (actual entity candidates).

Despite worse candidates, LLaDA pool still retrieves useful passages via seed answer overlap.

## Finding 3: Efficiency

| Method | Model | Latency/q | Optimization |
|--------|-------|:---------:|:------------:|
| DNMR pool | Dream-7B | ~5.4s | Vanilla PyTorch |
| DNMR pool | LLaDA-8B | ~6.8s | Vanilla PyTorch |
| IRCoT | Qwen3-8B | ~7.0s | vLLM + KV cache |
| AR-MQR | Qwen3-8B | ~32s | vLLM + KV cache |

Caveat: AR uses vLLM optimization, dLLM uses vanilla PyTorch. fast-dLLM not applied. FLOPs comparison pending.

Structural advantages (implementation-independent):
1. Single-round multi-query retrieval vs 2-3 sequential AR rounds
2. No chain-of-thought tokens (32 vs 100-500 tokens)
3. Parallel token prediction at each step

## Finding 4: Pipeline Sensitivity (2x2 Ablation)

Query prefix is essential for Dream (+10pp). LLaDA F1 loss is from verbosity, not retrieval failure.

## Open Items

1. Fix LLaDA verbosity: test n_tokens=8 for final decode (~1000 SBUs)
2. Efficiency: run fast-dLLM + compute FLOPs for fair comparison
3. Evaluation: LLM judge on existing predictions

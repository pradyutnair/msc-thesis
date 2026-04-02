# DNMR Efficiency Analysis

## Forward Pass Count Comparison (Implementation-Independent)

### DNMR (dLLM, single-round)
| Step | Forward passes | Notes |
|------|:--------------:|-------|
| Initial retrieval | 0 | Retriever only |
| Seed decode (32 steps) | 32 | Full prefix+answer each step (vanilla) or answer-only (fast-dLLM) |
| Bridge extraction | 1 + 3x12 = 37 | 1 initial + 3 branches x 12 rollout steps |
| Expansion retrieval | 0 | Retriever only |
| Final decode (32 steps) | 32 | From expanded context |
| **Total** | **~101** | Per question |

### IRCoT (AR, 2-round iterative)
| Step | Forward passes | Notes |
|------|:--------------:|-------|
| Initial retrieval | 0 | Retriever only |
| Round 1: CoT generation | ~200 | ~200 tokens generated sequentially |
| Round 1: retrieval | 0 | Retriever only |
| Round 2: CoT generation | ~200 | Another ~200 tokens |
| Round 2: retrieval | 0 | |
| Final answer extraction | ~50 | |
| **Total** | **~450** | Per question |

**DNMR uses ~4.5x fewer forward passes than IRCoT.**

Note: each AR forward pass processes the full KV-cached context (cheap with vLLM). Each dLLM forward pass processes the full sequence bidirectionally. Per-pass cost differs, but the sequential dependency chain is much shorter for DNMR.

### Tokens generated
| Method | Answer tokens | Reasoning tokens | Total |
|--------|:------------:|:----------------:|:-----:|
| DNMR | 32 | 0 | 32 |
| IRCoT | ~20 | ~400 | ~420 |
| AR-MQR | ~20 | ~100 | ~120 |

DNMR generates 13x fewer tokens than IRCoT. No chain-of-thought overhead.

## Wall-Clock Latency

### Unoptimized (current)
| Method | Model | Optimization | Latency/q |
|--------|-------|:------------:|:---------:|
| DNMR pool | Dream-7B | Vanilla PyTorch | ~5.4s |
| DNMR pool | LLaDA-8B | Vanilla PyTorch | ~6.8s |
| IRCoT | Qwen3-8B | vLLM + KV cache | ~7.0s |
| AR-MQR | Qwen3-8B | vLLM + KV cache | ~32.0s |

Even without optimization, DNMR is competitive with optimized AR inference.

### With fast-dLLM (expected)
fast-dLLM caches the context prefix KV across denoising steps. Only the answer region (~32 tokens) is recomputed per step, vs full sequence (~2000+ tokens) in vanilla mode.

Expected speedup: 2-4x (prefix is ~98% of sequence length).

| Method | Model | Optimization | Expected latency/q |
|--------|-------|:------------:|:------------------:|
| DNMR pool | Dream-7B | fast-dLLM + prefix cache | ~1.5-2.5s |
| DNMR pool | LLaDA-8B | fast-dLLM + prefix cache | ~2.0-3.5s |

### Structural Efficiency Advantages
1. **Single retrieval round:** DNMR extracts K bridges + retrieves in one round. IRCoT needs 2-3 sequential rounds.
2. **No CoT tokens:** dLLMs produce answers directly. AR methods generate 10-20x more tokens for reasoning chains.
3. **Parallel denoising:** All answer tokens predicted simultaneously at each step. AR generates one token at a time.
4. **Prefix caching:** fast-dLLM amortizes context encoding across denoising steps. Equivalent to KV caching for AR but for bidirectional attention.

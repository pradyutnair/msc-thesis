# DNMR Efficiency Analysis

## Status: Needs GPU benchmark for real numbers

## Wall-Clock Latency (Current, NOT Fairly Comparable)

| Method | Model | Optimization | Latency/q |
|--------|-------|:------------:|:---------:|
| DNMR pool | Dream-7B | Vanilla PyTorch | ~5.4s |
| DNMR pool | LLaDA-8B | Vanilla PyTorch | ~6.8s |
| IRCoT | Qwen3-8B | vLLM + KV cache | ~7.0s |
| AR-MQR | Qwen3-8B | vLLM + KV cache | ~32.0s |

**These numbers are NOT fairly comparable.** AR methods use vLLM with KV caching. dLLM methods use vanilla PyTorch with full recomputation every step. AR forward passes are cheap (1 token, cached KV). dLLM forward passes are expensive (full bidirectional attention over entire sequence).

## TODO: Fair Comparison

1. Run fast-dLLM (prefix KV caching for dLLMs) — exists in repo at dllm/pipelines/fastdllm/
2. Compute actual FLOPs for both paradigms under matched conditions
3. Report both optimized wall-clock AND theoretical FLOPs

## Structural Advantages (Qualitative, No Numbers)

1. **Single retrieval round:** DNMR does multi-query retrieval in one round. IRCoT needs 2-3 sequential rounds.
2. **No chain-of-thought tokens:** dLLMs produce answers directly (~32 tokens). AR with CoT generates 100-500 reasoning tokens.
3. **Prefix caching available:** fast-dLLM amortizes context encoding across denoising steps.

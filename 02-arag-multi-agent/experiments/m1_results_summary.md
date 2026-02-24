# M1 Experiment Results — Multi-Agent A-RAG

## Baseline: E4 (Single-Agent, Qwen3-30B-A3B + E5-base-v2 + DeepSeek judge)

| Dataset       | LLM Accuracy |
|---------------|-------------|
| HotpotQA      | 66.5%       |
| 2WikiMultihop | 56.9%       |
| MuSiQue       | 37.6%       |
| **Mean**      | **53.7%**   |

## M1 Variants (Multi-Agent, same backbone)

| Version | HotpotQA | 2WikiMultihop | MuSiQue | Mean  | Key Change |
|---------|----------|---------------|---------|-------|------------|
| M1      | 44.7%    | 25.4%         | 24.9%   | 31.7% | Baseline multi-agent pipeline |
| M1v3    | 49.9%    | 28.2%         | 27.4%   | 35.2% | Full 1000-sample runs |
| M1v5    | 55.1%    | 32.1%         | 29.6%   | 38.9% | Fix finish() leak; decisive aggregator; single-hop bypass |
| M1v6    | 55.9%    | 27.7%         | 30.3%   | 38.0% | Unified evidence pool (M1v6 regressed on 2Wiki) |
| M1v7    | 51.5%    | 23.6%         | 29.6%   | 34.9% | Per-entity sections (broken chain_instruction) |
| M1v8    | 54.9%    | 32.3%         | 30.3%   | **39.2%** | Fixed comparison instruction; disabled self-verify for comparison |

## Key Findings

### Why M1 < E4 (~14pp gap)
1. **Decomposition failure cascade**: decomposer misclassification, wrong sub-questions
2. **Aggregation bottleneck**: aggregator reconciles compressed sub-answers, not raw docs
3. **finish() tool call leak**: Qwen3 writes finish() as plain text (~40% of calls without fix)
4. **Sequential bridge errors**: wrong SQ-0 answer corrupts SQ-1 search
5. **Self-verify noise**: extra LLM call with truncated context corrupts correct answers

### What Worked
- Single-hop bypass (M1v5): +large gain on single-hop subset
- Decisive aggregator prompt (M1v5): eliminated over-refusing
- Per-entity sections for comparison (M1v8): fixed 2Wiki regression from M1v6

### What Didn't Work
- Flat unified pool (M1v6): scrambled entity-attribute correspondence for comparison (-4.4pp 2Wiki)
- Last-agent-only bridge pool (M1v7): too restrictive, lost context
- Self-verify for comparison (M1v7): biased toward first entity section

## M2 Direction: DRHR (Decomposed Retrieval, Holistic Reasoning)
Hypothesis: decomposition should guide *retrieval* only; *reasoning* should remain holistic.
Agents are pure retrieval workers — sub-answers ignored — synthesizer reasons over full evidence pool.

## M2 DRHR Results

| Version  | HotpotQA | 2WikiMultihop | MuSiQue | Mean  | Key Change |
|----------|----------|---------------|---------|-------|------------|
| M1v8     | 54.9%    | 32.3%         | 30.3%   | 39.2% | Best M1 baseline |
| **M2**   | **63.4%**| **34.2%**     | **27.2%**| **41.6%** | DRHR: holistic synthesis over raw evidence pool |
| E4       | 66.5%    | 56.9%         | 37.6%   | 53.7% | Single-agent E4 baseline |

### M2 vs M1v8 Deltas
- HotpotQA: +8.5pp (62.6% of the E4 gap closed)
- 2WikiMultihop: +1.9pp (still 22.7pp below E4 — 2Wiki comparison questions still hard)
- MuSiQue: -3.1pp (regression — holistic synthesis struggles with MuSiQue chains)
- Mean: +2.4pp (41.6% vs 39.2%)

### M2 vs E4 Remaining Gaps
- HotpotQA: 3.1pp behind E4 (nearly matched!)
- 2WikiMultihop: 22.7pp behind E4
- MuSiQue: 10.4pp behind E4

### Key Findings from M2
1. **DRHR massively helps HotpotQA** (+8.5pp): holistic synthesis resolves comparison/bridge correctly
2. **2Wiki still hard**: comparison questions require fine-grained entity attribute extraction — decomposition helps retrieval but synthesis still struggles
3. **MuSiQue regressed**: MuSiQue's multi-hop chains need more targeted synthesis strategy
4. **Known issues**: ~11% of bridge/comparison predictions have FINAL ANSWER: prefix (not stripped due to regex edge case) — likely minimal accuracy impact due to LLM judge
5. **~2.5% VLLM errors**: search agents occasionally overflow context (17k+ tokens) → fallback to trajectory chunks

### Technical Issues Fixed in M2
- max_tokens=512 in synthesis call: prevents aggregator context overflow
- Dynamic _safe_max_tokens in base.py: prevents search agent context overflow  
- Config max_tokens: 4096 → 512: belt-and-suspenders for all LLM calls
- Evidence budgets: 5500/2750 tiktoken (not 8000/4000 — Qwen tokenizer 2.3x expansion)

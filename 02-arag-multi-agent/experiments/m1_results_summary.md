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

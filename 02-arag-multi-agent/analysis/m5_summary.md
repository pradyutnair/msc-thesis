# M5: Multi-Agent Orchestrator — Analysis Summary

## Architecture

**E4 (Baseline)**: Single ReAct agent with 3 raw tools (`keyword_search`, `semantic_search`, `read_chunk`). Qwen3-30B-A3B with thinking mode ON. No explicit `finish` tool — agent returns text when done. Direct tool arguments (keywords, queries) chosen by the thinking model. Located at `/projects/prjs1800/msc-thesis/01-arag-reproduction/results/qwen3-30b-e5-deepseekr1/`.

**M5 (Multi-Agent)**: Single ReAct orchestrator with 4 LLM-augmented subagent tools (`keyword_agent`, `semantic_agent`, `chunk_reader`, `finish`). Each subagent wraps a raw tool with an additional LLM call for argument generation (e.g., keyword extraction, query formulation, evidence extraction).

Key architectural difference: E4's model directly chooses `keywords=["Bill Nelson", "Payload Specialist"]` in one step. M5's model says `task="Bill Nelson Payload Specialist"`, then a separate LLM call extracts keywords — adding a lossy indirection layer.

---

## Bug Fix

**Critical bug discovered**: 100% of subagent tool calls failed with `KeyError('"keywords"')`.

**Root cause**: Python's `str.format()` in `_PromptedSubagentTool._generate()` interpreted `{"keywords": ["kw1", "kw2"]}` in prompt templates as format variables. When `self.prompt_template.format(task=task)` ran, it tried to resolve `"keywords"` as a format field.

**Fix**: Replaced `.format()` with a custom `_render_prompt()` method using `str.replace()`:
```python
def _render_prompt(self, **kwargs):
    result = self.prompt_template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result
```

File: `src/multi_agent/m5/subagent_tools.py`

---

## Prompt Iteration (100q HotpotQA pilot, thinking OFF)

All v1–v6 runs used `enable_thinking: false` and `max_tokens: 512`. These were initially compared against the non-agentic `b0` baseline (64% LLM on 100q) before the correct E4 agentic baseline was identified.

| Version | LLM Acc | Contain Acc | Changes |
|---------|---------|-------------|---------|
| M5 v1 | 60% | 52% | Bug fixed, basic prompt |
| M5 v2 | 61% | 58% | Better multi-hop + answer instructions |
| M5 v3 | 61% | 53% | Concise answer format |
| M5 v4 | 61% | 55% | Explicit multi-hop examples (Pittsburgh→PPG, Superstore→Spitzer) |
| M5 v5 | 64% | 60% | Exact phrasing from evidence (fixed near-misses) |
| M5 v6 | 65% | 64% | + Answer verification step, top_k=10 suggestion, BBC F1 example |

**Key insight from v1–v3**: Verbosity/conciseness of answers did NOT affect LLM accuracy (stable at 61%). The issue was answer precision — "13" vs "13 seasons", "Karakoram" vs "Karakoram mountain range". Fixed in v5 by instructing "use the EXACT phrasing from your evidence."

---

## E4 Baseline (1000q, agentic)

E4 results from `/projects/prjs1800/msc-thesis/01-arag-reproduction/results/qwen3-30b-e5-deepseekr1/`:

| Dataset | LLM Acc | Contain Acc | Avg Loops | Avg Cost/Q |
|---------|---------|-------------|-----------|------------|
| HotpotQA | 66.5% | 67.7% | 2.66 | $0.012 |
| MuSiQue | 37.6% | 34.4% | 2.98 | $0.015 |
| 2WikiMultiHop | 56.9% | 63.9% | 3.05 | $0.013 |

---

## 1000-Question Full Run (M5 v6, thinking OFF)

| Dataset | E4 | M5 v6 | Delta |
|---------|-----|-------|-------|
| HotpotQA | **66.5%** | 61.9% | -4.6% |
| MuSiQue | **37.6%** | 27.6% | -10.0% |
| 2WikiMultiHop | **56.9%** | 44.1% | -12.8% |

M5 v6 significantly underperforms E4 across all datasets.

---

## E4 vs M5 Architecture Comparison

Investigation revealed critical differences:

| Dimension | E4 | M5 (v1–v6) |
|---|---|---|
| Thinking mode | **ON** (default) | OFF (`enable_thinking: false`) |
| max_tokens/call | 8192 | 512 |
| vLLM max-model-len | 32768 | 16384 |
| Reasoning in history | Preserved (`reasoning_content` merged) | Removed |
| Tools | 3 raw tools (direct args) | 4 subagent tools (LLM-mediated) |
| Answer mechanism | Text response (implicit) | `finish()` tool call (explicit) |
| read_chunk usage | 11% of tool calls | ~8% of tool calls |

---

## M5 v8 (Thinking ON, Matched Settings)

Updated M5 to match E4's inference settings:
- Thinking mode ON (removed `chat_template_kwargs`)
- `max_tokens: 8192` (was 512)
- `--max-model-len 32768` (was 16384)
- `reasoning_content` merge restored in `base.py`
- `_QWEN_CONTEXT_LIMIT` updated to 32768

### 100q HotpotQA pilot (M5v8 vs E4)

| System | LLM Acc | Contain Acc | Avg Loops | Avg Chunks Read |
|--------|---------|-------------|-----------|-----------------|
| E4 | **66%** | **64%** | 2.7 | 0.2 |
| M5v8 | 62% | 51% | 3.3 | 0.2 |

Gap: -4% LLM, -13% contain. 20 questions E4-right/M5-wrong, 16 M5-right/E4-wrong.

---

## Failure Analysis (M5v8 vs E4, 100q HotpotQA)

### Tool Usage Comparison

| Tool | E4 calls | M5v8 calls |
|------|---------|------------|
| keyword_search / keyword_agent | 111 (62%) | 129 (46%) |
| semantic_search / semantic_agent | 49 (27%) | 42 (15%) |
| read_chunk / chunk_reader | 20 (11%) | 21 (8%) |
| finish | N/A | 88 (31%) |
| **Total** | **180** | **280** |

M5 makes 56% more tool calls but achieves lower accuracy. The `finish` tool accounts for 31% of all calls — pure overhead.

### Failure Categories (20 E4-right, M5v8-wrong)

| Category | Count | Description |
|----------|-------|-------------|
| No chunks, wrong answer | 16 | M5 answered from snippets without reading full chunks |
| Read chunks, wrong answer | 2 | Evidence read but wrong answer extracted |
| Near-miss | 2 | Answer close but judged wrong (e.g., "Istanbul" vs "Istanbul, Turkey") |

### Root Causes

1. **Subagent indirection loses precision**: E4 directly picks `keywords=["Bill Nelson", "Payload Specialist"]`. M5 says `task="Bill Nelson Payload Specialist"` then a separate LLM call extracts keywords — a lossy translation step.

2. **M5 finishes too aggressively**: 16/20 failures have 0 chunks read. M5's orchestrator sees snippet-level evidence and calls `finish()` prematurely. E4's thinking model reasons through whether snippets are sufficient before deciding to read full chunks or search again.

3. **E4 does more targeted multi-step searches**: For comparison questions (e.g., "which opera has more acts?"), E4 does 4 searches (kw + kw + sem + sem). M5 does 2 keyword searches and finishes without verifying.

4. **Answer precision**: Near-misses like "Jean Cocteau" vs "Jean Maurice Eugène Clément Cocteau", "Bob Iger" vs "Robert A. Iger", "Istanbul" vs "Istanbul, Turkey".

5. **finish() tool adds failure modes**: M5 must explicitly call `finish()` as a tool call, requiring complex recovery logic (regex extraction from text, re-prompting, force-finish). E4 simply returns text when done.

---

## Conclusions

The multi-agent subagent architecture (M5) underperforms the single-agent direct-tool architecture (E4) by 4–13% across all datasets. Even after matching E4's inference settings (thinking ON, max_tokens 8192, 32k context), M5v8 still trails E4 by 4% on HotpotQA. The subagent indirection layer adds complexity, noise, and cost without improving reasoning quality. The core issue is that the subagent wrapper converts precise structured tool arguments into natural language descriptions, then back into structured arguments — a lossy round-trip that degrades search quality.

---

## Results Location

| Run | Path |
|-----|------|
| M5 v1 (100q) | `results/m5_pilot100/hotpotqa/predictions_v1.jsonl` |
| M5 v2 (100q) | `results/m5_pilot100/hotpotqa/predictions_v2.jsonl` |
| M5 v3 (100q) | `results/m5_pilot100/hotpotqa/predictions_v3.jsonl` |
| M5 v4 (100q) | `results/m5_pilot100/hotpotqa/predictions_v4.jsonl` |
| M5 v5 (100q) | `results/m5_pilot100/hotpotqa/predictions_v5.jsonl` |
| M5 v6 (100q) | `results/m5_pilot100/hotpotqa/predictions.jsonl` |
| M5 v6 (1000q) | `results/m5_1000/{hotpotqa,musique,2wikimultihop}/predictions.jsonl` |
| M5 v8 (20q pilot) | `results/m5v8_p20/hotpotqa/predictions.jsonl` |
| M5 v8 (100q pilot) | `results/m5v8_p100/hotpotqa/predictions.jsonl` |
| E4 (1000q) | `/projects/prjs1800/msc-thesis/01-arag-reproduction/results/qwen3-30b-e5-deepseekr1/{hotpotqa,musique,2wikimultihop}/predictions.jsonl` |

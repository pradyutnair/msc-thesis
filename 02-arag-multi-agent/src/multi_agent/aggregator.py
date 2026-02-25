"""Aggregator: DRHR (Decomposed Retrieval, Holistic Reasoning) synthesis — M2.

Core change from M1: sub-answers from agents are IGNORED. The synthesizer
reasons directly over the raw evidence pool, exactly like E4 but with
structured, sub-question-targeted retrieval feeding the pool.

OSPREY extension: optional scout_chunks prepended to the evidence pool as
a "Phase 1 Scout" preamble — gives the synthesizer the broad context from
the initial discovery phase before the targeted sub-question evidence.

Architecture:
  - comparison: per-entity labeled sections (each agent's chunks in its own section)
  - bridge:     flat pool ordered by chain step (SQ-0 first, SQ-N last)
  - single_hop: direct agent answer bypass (unchanged)
  - scout:      optional Phase 1 chunks prepended to pool as [Scout] label
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import tiktoken

from arag.core.llm import LLMClient
from multi_agent.evidence_cache import EvidenceCache
from multi_agent.types import AgentResult, DecompositionPlan

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "aggregator.txt"

_EVIDENCE_TOKEN_BUDGET = 5500
_COMPARISON_PER_ENTITY_BUDGET = 2750
_SCOUT_CHUNK_BUDGET = 1200   # budget reserved for Phase 1 Scout prefix

_CHAIN_INSTRUCTIONS = {
    "comparison": (
        "The sub-questions each retrieved documents for a DIFFERENT entity. "
        "The evidence is split into labeled ENTITY SECTIONS below. "
        "Read each section's documents to extract that entity's attribute. "
        "Compare the two attributes and answer the original question. "
        "If the question asks 'which' entity, output only the entity name. "
        "If the question asks 'are both' or 'do both', output yes or no."
    ),
    "bridge": (
        "The sub-questions form a reasoning chain. "
        "SQ-0's documents identify the intermediate entity needed for the next step. "
        "SQ-1 (and beyond) use that entity to find the final answer. "
        "Follow the chain through the documents from SQ-0 to the last SQ. "
        "State the final answer at the end of the chain."
    ),
    "single_hop": (
        "A single retrieval agent collected documents for this question. "
        "Read the documents and extract the direct answer."
    ),
}

_BAD_ANSWER_RE = re.compile(
    r"(evidence does not|does not support|cannot be determined|"
    r"insufficient|no valid|provided does not|cannot answer|"
    r"question cannot be answered|not enough information|"
    r"unable to determine|cannot confirm)",
    re.IGNORECASE,
)


def _is_bad_answer(text: str) -> bool:
    return bool(_BAD_ANSWER_RE.search(text))


def _clean_answer(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(
        r'finish\s*\(\s*(?:\{[^)]*?"answer"\s*:\s*"((?:[^"\\]|\\.)*)"'
        r'|answer\s*=\s*["\']([^"\']*)["\'])',
        text,
    )
    if m:
        text = (m.group(1) or m.group(2) or text).replace('\\"', '"')
    return text.strip()


class Aggregator:
    """DRHR aggregator: reasons over raw evidence, ignores agent sub-answers."""

    def __init__(
        self,
        llm_client: LLMClient,
        evidence_cache: EvidenceCache | None = None,
        enable_self_verify: bool = True,
        prompt_path: str | Path | None = None,
    ):
        self.llm = llm_client
        self.cache = evidence_cache

        path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._prompt_template = path.read_text(encoding="utf-8")

        try:
            self._tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def _get_agent_chunks(self, result: AgentResult) -> list[dict]:
        if result.retrieved_chunks:
            return result.retrieved_chunks
        chunks: list[dict] = []
        seen: set[str] = set()
        for entry in result.trajectory:
            if entry.get("tool_name") != "read_chunk":
                continue
            text = entry.get("tool_result", "")
            if text and "(already read)" not in text:
                tid = str(hash(text[:100]))
                if tid not in seen:
                    seen.add(tid)
                    chunks.append({"id": "?", "text": text[:1500]})
        return chunks

    def _fit_chunks_to_budget(
        self,
        chunks: list[dict],
        sq_index: int,
        budget: int,
    ) -> str:
        lines: list[str] = []
        tokens_used = 0
        label = "Scout" if sq_index == -1 else str(sq_index)
        for chunk in chunks:
            cid = chunk.get("id", "?")
            text = chunk.get("text", "")
            block = f"[Doc {cid} | SQ-{label}]\n{text}"
            block_tokens = self._count_tokens(block)
            if tokens_used + block_tokens > budget:
                remaining = budget - tokens_used
                if remaining > 200:
                    truncated = self._tokenizer.decode(
                        self._tokenizer.encode(text)[:remaining - 20]
                    )
                    lines.append(f"[Doc {cid} | SQ-{label}]\n{truncated} [truncated]")
                break
            lines.append(block)
            tokens_used += block_tokens
        return "\n\n---\n\n".join(lines)

    # ------------------------------------------------------------------
    # Evidence pool builders
    # ------------------------------------------------------------------

    def _build_scout_prefix(self, scout_chunks: list[dict]) -> str:
        """Build Phase 1 Scout evidence prefix for the unified pool."""
        if not scout_chunks:
            return ""
        lines: list[str] = []
        tokens_used = 0
        for chunk in scout_chunks:
            cid = chunk.get("id", "?")
            text = chunk.get("text", "")
            block = f"[Doc {cid} | Scout]\n{text}"
            block_tokens = self._count_tokens(block)
            if tokens_used + block_tokens > _SCOUT_CHUNK_BUDGET:
                remaining = _SCOUT_CHUNK_BUDGET - tokens_used
                if remaining > 200:
                    truncated = self._tokenizer.decode(
                        self._tokenizer.encode(text)[:remaining - 20]
                    )
                    lines.append(f"[Doc {cid} | Scout]\n{truncated} [truncated]")
                break
            lines.append(block)
            tokens_used += block_tokens
        if not lines:
            return ""
        header = "## Phase 1 Scout Evidence\n\n"
        return header + "\n\n---\n\n".join(lines)

    def _build_comparison_pool(
        self,
        plan: DecompositionPlan,
        agent_results: dict[int, AgentResult],
        scout_chunks: list[dict] | None = None,
    ) -> str:
        """Per-entity labeled sections — preserves entity-attribute correspondence."""
        _SEP = "\n\n" + "=" * 60 + "\n\n"
        sections: list[str] = []

        # Prepend scout evidence as a preamble section
        if scout_chunks:
            scout_prefix = self._build_scout_prefix(scout_chunks)
            if scout_prefix:
                sections.append(scout_prefix)

        for sq in plan.sub_questions:
            result = agent_results.get(sq.index)
            header = (
                f"## ENTITY SECTION SQ-{sq.index}\n"
                f"Sub-question: {sq.text}\n"
                f"Documents:"
            )
            if result:
                chunks = self._get_agent_chunks(result)
                docs_str = self._fit_chunks_to_budget(
                    chunks, sq.index, _COMPARISON_PER_ENTITY_BUDGET
                )
            else:
                docs_str = "(No documents retrieved)"

            sections.append(f"{header}\n\n{docs_str}")

        logger.info("Comparison pool: %d sections (incl. scout)", len(sections))
        return _SEP.join(sections)

    def _build_flat_pool(
        self,
        plan: DecompositionPlan,
        agent_results: dict[int, AgentResult],
        scout_chunks: list[dict] | None = None,
    ) -> str:
        """Flat merged pool ordered by sub-question (bridge & single_hop).

        Scout chunks are prepended, then sub-question chunks follow in order.
        """
        seen_ids: set[str] = set()
        ordered_chunks: list[tuple[int, dict]] = []

        # Phase 1 Scout chunks first (sq_index = -1 sentinel)
        if scout_chunks:
            for chunk in scout_chunks:
                cid = str(chunk.get("id", ""))
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    ordered_chunks.append((-1, chunk))

        for sq in plan.sub_questions:
            result = agent_results.get(sq.index)
            if result is None:
                continue
            for chunk in self._get_agent_chunks(result):
                cid = str(chunk.get("id", ""))
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    ordered_chunks.append((sq.index, chunk))

        if not ordered_chunks:
            return "(No documents retrieved)"

        lines: list[str] = []
        tokens_used = 0
        for sq_idx, chunk in ordered_chunks:
            cid = chunk.get("id", "?")
            text = chunk.get("text", "")
            label = "Scout" if sq_idx == -1 else str(sq_idx)
            block = f"[Doc {cid} | SQ-{label}]\n{text}"
            block_tokens = self._count_tokens(block)
            if tokens_used + block_tokens > _EVIDENCE_TOKEN_BUDGET:
                remaining = _EVIDENCE_TOKEN_BUDGET - tokens_used
                if remaining > 200:
                    truncated = self._tokenizer.decode(
                        self._tokenizer.encode(text)[:remaining - 20]
                    )
                    lines.append(f"[Doc {cid} | SQ-{label}]\n{truncated} [truncated]")
                break
            lines.append(block)
            tokens_used += block_tokens

        scout_count = sum(1 for sq_idx, _ in ordered_chunks if sq_idx == -1)
        logger.info(
            "Flat pool: %d chunks (%d scout), %d tokens",
            len(lines), scout_count, tokens_used,
        )
        return "\n\n---\n\n".join(lines)

    def _build_evidence(
        self,
        plan: DecompositionPlan,
        agent_results: dict[int, AgentResult],
        scout_chunks: list[dict] | None = None,
    ) -> str:
        if plan.question_type == "comparison":
            return self._build_comparison_pool(plan, agent_results, scout_chunks)
        return self._build_flat_pool(plan, agent_results, scout_chunks)

    # ------------------------------------------------------------------
    # Synthesis (single holistic call — the DRHR core)
    # ------------------------------------------------------------------

    async def _synthesize(
        self,
        question: str,
        plan: DecompositionPlan,
        unified_pool: str,
    ) -> tuple[str, float]:
        chain_instruction = _CHAIN_INSTRUCTIONS.get(
            plan.question_type, _CHAIN_INSTRUCTIONS["single_hop"]
        )
        prompt = self._prompt_template.format(
            question=question,
            question_type=plan.question_type,
            chain_instruction=chain_instruction,
            unified_pool=unified_pool,
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.chat(messages=messages, tools=None, temperature=0.0, max_tokens=512)
        raw = response["message"].get("content", "")
        cost = response.get("cost", 0.0)
        answer = self._extract_final_answer(raw)
        logger.info("Synthesis → '%s'", answer[:80])
        return answer, cost

    @staticmethod
    def _extract_final_answer(raw: str) -> str:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        m = re.search(r"FINAL\s*ANSWER\s*:\s*(.+)", raw, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        return lines[-1] if lines else raw.strip()

    # ------------------------------------------------------------------
    # Agent-answer fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _best_agent_answer(agent_results: dict[int, AgentResult]) -> str:
        candidates = [
            _clean_answer(ar.answer)
            for idx, ar in agent_results.items()
            if idx != -1 and ar.answer and not _is_bad_answer(ar.answer)
        ]
        if not candidates:
            return ""
        candidates.sort(key=lambda s: abs(len(s) - 40))
        return candidates[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def aggregate(
        self,
        question: str,
        plan: DecompositionPlan,
        agent_results: dict[int, AgentResult],
        scout_chunks: list[dict] | None = None,
    ) -> tuple[str, int]:
        """Returns (final_answer, approx_token_cost).

        Parameters
        ----------
        scout_chunks:
            Optional Phase 1 Scout chunks (OSPREY). Prepended to the evidence
            pool so the synthesizer has broad Phase 1 context before the
            targeted sub-question evidence.
        """
        # Fast path: single-hop — use agent answer directly
        if plan.question_type == "single_hop" and len(agent_results) == 1:
            only_idx = next(iter(agent_results))
            result = agent_results[only_idx]
            clean = _clean_answer(result.answer)
            if clean and not _is_bad_answer(clean):
                return clean, 0

        # Build type-specific evidence pool (with optional scout prefix)
        unified_pool = self._build_evidence(plan, agent_results, scout_chunks)

        # Single holistic synthesis call (DRHR core)
        answer, cost = await self._synthesize(question, plan, unified_pool)

        # Fallback to best agent answer if synthesis fails
        if _is_bad_answer(answer) or not answer:
            fallback = self._best_agent_answer(agent_results)
            if fallback:
                logger.info("Synthesis fallback → '%s'", fallback[:60])
                answer = fallback

        approx_tokens = int(cost * 1_000_000) if cost > 0 else 0
        m_fa = re.match(r"(?i)FINAL\s*ANSWER\s*:\s*(.*)", answer)
        if m_fa:
            answer = m_fa.group(1).strip()
        return answer, approx_tokens

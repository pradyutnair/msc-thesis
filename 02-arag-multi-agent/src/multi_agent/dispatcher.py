"""Dependency-aware async dispatcher for search agents."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.evidence_cache import EvidenceCache
from multi_agent.search_agent import SearchAgent
from multi_agent.types import AgentResult, DecompositionPlan, SubQuestion

logger = logging.getLogger(__name__)


class Dispatcher:
    """Dispatch search agents in topological waves via asyncio.gather.

    Execution strategy by question type:
    - **comparison**: All sub-Qs in parallel (one wave).
    - **bridge**: Sequential waves following dependency edges.
    - **single_hop**: One agent, one wave.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        evidence_cache: EvidenceCache | None = None,
        max_loops_per_agent: int = 5,
        max_token_budget: int = 64000,
        max_concurrent: int = 3,
        verbose: bool = False,
        search_agent_prompt_path: str | None = None,
        enable_cep: bool = False,
    ):
        self.llm = llm_client
        self.tools = tools
        self.cache = evidence_cache
        self.max_loops = max_loops_per_agent
        self.max_token_budget = max_token_budget
        self.max_concurrent = max_concurrent
        self.verbose = verbose
        self.search_agent_prompt_path = search_agent_prompt_path
        self.enable_cep = enable_cep
        self._semaphore = asyncio.Semaphore(max_concurrent)

    # ------------------------------------------------------------------
    # Topological sort → wave grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_waves(plan: DecompositionPlan) -> list[list[SubQuestion]]:
        """Group sub-questions into execution waves via topological levels.

        Wave 0 = all sub-Qs with no dependencies.
        Wave k = sub-Qs whose dependencies are all in waves < k.
        """
        sq_map = {sq.index: sq for sq in plan.sub_questions}
        indices = set(sq_map.keys())

        # Build adjacency from both dependency_edges and depends_on.
        edge_set: set[tuple[int, int]] = set()
        for src, tgt in plan.dependency_edges:
            if src in indices and tgt in indices and src != tgt:
                edge_set.add((src, tgt))
        for sq in plan.sub_questions:
            for dep in sq.depends_on:
                if dep in indices and sq.index in indices and dep != sq.index:
                    edge_set.add((dep, sq.index))

        children: dict[int, list[int]] = defaultdict(list)
        in_degree: dict[int, int] = {i: 0 for i in indices}
        for src, tgt in edge_set:
            children[src].append(tgt)
            in_degree[tgt] += 1

        # Kahn's algorithm, grouping by BFS level
        waves: list[list[SubQuestion]] = []
        ready = [i for i in indices if in_degree[i] == 0]

        while ready:
            wave = [sq_map[i] for i in sorted(ready)]
            waves.append(wave)

            next_ready: list[int] = []
            for i in ready:
                for child in children[i]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_ready.append(child)
            ready = next_ready

        # Sanity: all sub-Qs placed
        placed = sum(len(w) for w in waves)
        if placed != len(indices):
            placed_indices = {sq.index for w in waves for sq in w}
            remaining = [sq_map[i] for i in indices - placed_indices]
            if remaining:
                logger.warning(
                    "Dispatcher: %d sub-Qs not placed by topo sort (cycle?), "
                    "appending as final wave",
                    len(remaining),
                )
                waves.append(remaining)

        return waves

    # ------------------------------------------------------------------
    # Chain Evidence Passing (M3-CEP and OSPREY global evidence)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chain_evidence(
        wave_results: list[AgentResult],
        wave_idx: int,
        max_chunks: int = 3,
        max_chars_per_chunk: int = 600,
    ) -> str:
        """Build CEP chain evidence string from completed wave results.

        Extracts top retrieved chunks from wave_idx agents to pass as
        context to the next wave's agents (bridge questions only).
        """
        lines: list[str] = []
        for result in wave_results:
            if not result.retrieved_chunks:
                continue
            sq_idx = result.sub_question_index
            for chunk in result.retrieved_chunks[:max_chunks]:
                cid = chunk.get("id", "?")
                text = chunk.get("text", "")[:max_chars_per_chunk]
                lines.append(f"[Chunk {cid} | SQ-{sq_idx}]:\n{text}")

        if not lines:
            return ""

        header = (
            f"[Evidence from Wave {wave_idx} — "
            "use to identify the key intermediate entity and guide your search]\n\n"
        )
        return header + "\n\n---\n\n".join(lines)

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        sub_question: SubQuestion,
        resolved_answers: dict[int, str],
        original_question: str = "",
        chain_evidence: str = "",
    ) -> AgentResult:
        """Run a single search agent under the concurrency semaphore."""
        async with self._semaphore:
            agent = SearchAgent(
                llm_client=self.llm,
                tools=self.tools,
                evidence_cache=self.cache,
                max_loops=self.max_loops,
                max_token_budget=self.max_token_budget,
                prompt_path=self.search_agent_prompt_path,
                verbose=self.verbose,
            )
            return await agent.run(
                sub_question=sub_question,
                resolved_answers=resolved_answers,
                original_question=original_question,
                chain_evidence=chain_evidence,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        plan: DecompositionPlan,
        original_question: str = "",
        global_chain_evidence: str = "",
    ) -> tuple[dict[int, AgentResult], int]:
        """Dispatch agents for all sub-questions in *plan*.

        Parameters
        ----------
        plan:
            Decomposition plan from the Decomposer.
        original_question:
            The original question (for agent context).
        global_chain_evidence:
            Optional evidence to inject into ALL agents regardless of wave.
            Used by OSPREY to propagate Phase 1 Scout chunks into every agent.
            CEP (wave-specific) evidence is combined on top of this.

        Returns
        -------
        results : dict[int, AgentResult]
            Mapping of sub-question index → result.
        num_waves : int
            Number of execution waves.
        """
        waves = self._compute_waves(plan)
        results: dict[int, AgentResult] = {}
        resolved_answers: dict[int, str] = {}

        # CEP: per-wave chain evidence accumulates across bridge waves
        cep_chain_evidence: str = ""

        logger.info(
            "Dispatching %d sub-Qs in %d waves for '%s' question%s",
            len(plan.sub_questions),
            len(waves),
            plan.question_type,
            " [+global evidence]" if global_chain_evidence else "",
        )

        for wave_idx, wave in enumerate(waves):
            logger.info(
                "Wave %d/%d: %d agents [%s]",
                wave_idx + 1,
                len(waves),
                len(wave),
                ", ".join(f"SQ-{sq.index}" for sq in wave),
            )

            # CEP only applies to bridge questions (wave > 0)
            use_cep = (
                self.enable_cep
                and plan.question_type == "bridge"
                and wave_idx > 0
                and bool(cep_chain_evidence)
            )
            if use_cep:
                logger.info(
                    "CEP: injecting %d chars of wave evidence into wave %d",
                    len(cep_chain_evidence), wave_idx,
                )

            # Combine global (scout) evidence with wave-specific CEP evidence
            if global_chain_evidence and use_cep:
                effective_chain = global_chain_evidence + "\n\n" + cep_chain_evidence
            elif global_chain_evidence:
                effective_chain = global_chain_evidence
            elif use_cep:
                effective_chain = cep_chain_evidence
            else:
                effective_chain = ""

            # Launch all agents in this wave concurrently
            tasks = [
                self._run_agent(
                    sq,
                    dict(resolved_answers),
                    original_question=original_question,
                    chain_evidence=effective_chain,
                )
                for sq in wave
            ]
            wave_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            good_results: list[AgentResult] = []
            for sq, result in zip(wave, wave_results):
                if isinstance(result, Exception):
                    logger.error(
                        "Agent for SQ-%d raised: %s", sq.index, result
                    )
                    result = AgentResult(
                        sub_question_index=sq.index,
                        answer="",
                        error=str(result),
                    )

                results[sq.index] = result
                resolved_answers[sq.index] = result.answer
                good_results.append(result)

                if self.verbose:
                    print(
                        f"  SQ-{sq.index}: '{sq.text[:50]}' → "
                        f"'{result.answer[:60]}' "
                        f"({result.loops} loops, {result.wall_clock_seconds:.1f}s)"
                    )

            # CEP: build wave-specific chain evidence for next bridge wave
            if self.enable_cep and plan.question_type == "bridge":
                cep_chain_evidence = self._build_chain_evidence(good_results, wave_idx)

        return results, len(waves)

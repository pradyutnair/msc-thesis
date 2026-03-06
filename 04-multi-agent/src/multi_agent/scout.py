"""Phase 1 Scout: single-agent broad evidence discovery for OSPREY."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.confidence_gate import ConfidenceGate
from multi_agent.evidence_cache import EvidenceCache
from multi_agent.search_agent import SearchAgent
from multi_agent.types import AgentResult, ScoutResult, SubQuestion

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "scout.txt"


class Scout:
    """Phase 1 broad evidence discovery for the OSPREY pipeline.

    Runs a single SearchAgent on the full original question with a reduced
    loop budget (default 3). The goal is NOT to answer definitively — it is
    to collect initial evidence, identify key entities, and provide a
    preliminary answer for the ConfidenceGate.

    If the gate fires (confidence ≥ threshold), the Scout answer is returned
    directly with no further decomposition/dispatching. For hard multi-hop
    questions the gate does not fire, but the Scout chunks are injected into:
      1. The EvidenceAwareDecomposer prompt (guides sub-question generation)
      2. All Phase 2 agents as global chain evidence (via Dispatcher)
      3. The Aggregator's evidence pool as Phase 1 prefix
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        evidence_cache: EvidenceCache | None = None,
        max_loops: int = 3,
        max_token_budget: int = 64000,
        confidence_threshold: float = 0.65,
        prompt_path: str | Path | None = None,
        verbose: bool = False,
    ):
        self.llm = llm_client
        self.tools = tools
        self.cache = evidence_cache
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget
        self.verbose = verbose

        path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._prompt_path = path
        self._gate = ConfidenceGate(threshold=confidence_threshold)

    async def scout(self, question: str) -> ScoutResult:
        """Run Phase 1 scout on *question*.

        Returns a :class:`ScoutResult` with the preliminary answer,
        retrieved chunks, confidence score, and fast-exit flag.
        """
        t0 = time.monotonic()

        # Sentinel sub-question (index=-1 marks scout in agent_results)
        sq = SubQuestion(index=-1, text=question, search_hints=[])

        agent = SearchAgent(
            llm_client=self.llm,
            tools=self.tools,
            evidence_cache=self.cache,
            max_loops=self.max_loops,
            max_token_budget=self.max_token_budget,
            prompt_path=self._prompt_path,
            verbose=self.verbose,
        )

        agent_result = await agent.run(
            sub_question=sq,
            resolved_answers={},
            original_question=question,
            chain_evidence="",
        )

        # Ensure sentinel index is preserved
        agent_result.sub_question_index = -1

        confidence = self._gate.score(agent_result.answer)
        is_confident = self._gate.is_confident(agent_result.answer)
        elapsed = time.monotonic() - t0

        logger.info(
            "Scout: '%s' → '%s' (conf=%.2f, %s, %.1fs, %d chunks)",
            question[:60],
            agent_result.answer[:60],
            confidence,
            "FAST-EXIT" if is_confident else "phase-2",
            elapsed,
            len(agent_result.retrieved_chunks),
        )

        return ScoutResult(
            answer=agent_result.answer,
            chunks=agent_result.retrieved_chunks,
            confidence=confidence,
            is_confident=is_confident,
            agent_result=agent_result,
            wall_clock_seconds=elapsed,
        )

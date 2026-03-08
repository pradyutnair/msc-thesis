"""Autonomous multi-agent collaborative search pipeline."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from arag.core.config import Config
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.blackboard import Blackboard
from multi_agent.investigator import Investigator
from multi_agent.strategist import Strategist
from multi_agent.types import AgentResult, PipelineResult

logger = logging.getLogger(__name__)


class AutonomousPipeline:
    """Blackboard-based collaborative multi-agent search pipeline."""

    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        config: Config | None = None,
    ):
        self.llm = llm_client
        self.tools = tools
        self.config = config or Config()

        ma_cfg = self.config.get("multi_agent", {}) or {}
        self.max_rounds = int(ma_cfg.get("max_rounds", 4))
        self.max_concurrent = int(ma_cfg.get("max_concurrent_investigators", 3))
        self.retrieval_top_k = int(ma_cfg.get("retrieval_top_k", 10))
        self.max_queries_per_entity = int(ma_cfg.get("max_queries_per_entity", 6))
        self.max_doc_chars = int(ma_cfg.get("max_doc_chars", 5000))
        self.retry_low_confidence = bool(ma_cfg.get("retry_low_confidence", False))
        self.verification_enabled = bool(ma_cfg.get("verification_enabled", True))
        self.min_confidence = float(ma_cfg.get("min_confidence_threshold", 0.3))
        self.verbose = bool(ma_cfg.get("verbose", False))

        # Load custom prompt paths if provided
        plan_prompt = self._load_prompt(ma_cfg.get("strategist_plan_prompt"))
        review_prompt = self._load_prompt(ma_cfg.get("strategist_review_prompt"))
        verify_prompt = self._load_prompt(ma_cfg.get("strategist_verify_prompt"))
        answer_prompt = self._load_prompt(ma_cfg.get("strategist_answer_prompt"))

        self.strategist = Strategist(
            llm_client=llm_client,
            plan_prompt=plan_prompt,
            review_prompt=review_prompt,
            verify_prompt=verify_prompt,
            answer_prompt=answer_prompt,
            verbose=self.verbose,
        )

    @staticmethod
    def _load_prompt(path: str | None) -> str | None:
        if not path:
            return None
        p = Path(path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    async def _run_investigator(self, hop, blackboard: Blackboard, inv_id: int) -> dict[str, Any]:
        """Run a single investigator for one hop."""
        agent_id = f"inv_{inv_id}"
        investigator = Investigator(
            llm_client=self.llm,
            base_tools=self.tools,
            blackboard=blackboard,
            hop=hop,
            agent_id=agent_id,
            retrieval_top_k=self.retrieval_top_k,
            max_queries_per_entity=self.max_queries_per_entity,
            max_doc_chars=self.max_doc_chars,
            retry_low_confidence=self.retry_low_confidence,
            verbose=self.verbose,
        )
        return await investigator.run()

    async def run(self, question: str) -> PipelineResult:
        """Run the full autonomous pipeline for one question."""
        t0 = time.monotonic()
        blackboard = Blackboard(question=question)
        total_inv_results: dict[int, dict] = {}
        investigator_counter = 0

        # Phase 1: Strategic Planning
        if self.verbose:
            logger.info("Phase 1: Planning for '%s'", question[:60])
        await self.strategist.plan(question, blackboard)

        # Phase 2: Iterative Investigation
        for round_idx in range(self.max_rounds):
            actionable = blackboard.get_actionable_hops()
            if not actionable:
                if self.verbose:
                    logger.info("Round %d: No actionable hops, done", round_idx)
                break

            if self.verbose:
                logger.info(
                    "Round %d: %d actionable hops: %s",
                    round_idx,
                    len(actionable),
                    [h.id for h in actionable],
                )

            # Spawn investigators in parallel for independent hops
            hops_to_investigate = actionable[:self.max_concurrent]
            tasks = []
            for hop in hops_to_investigate:
                # Re-resolve placeholders (deps may have been resolved this round)
                hop.resolved_question = blackboard.resolve_placeholders(hop)
                tasks.append(self._run_investigator(hop, blackboard, investigator_counter))
                investigator_counter += 1

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for hop, result in zip(hops_to_investigate, results):
                if isinstance(result, Exception):
                    logger.error("Investigator for hop %d failed: %s", hop.id, result)
                    blackboard.resolve_hop(hop.id, "", [], 0.0)
                else:
                    total_inv_results[hop.id] = result

            # Auto-retry stuck hops (answer is empty/unknown) before strategist review
            for hop in hops_to_investigate:
                if hop.status == "stuck" and hop.attempt_count < 2:
                    hop.status = "pending"
                    hop.attempt_count += 1
                elif hop.status == "resolved" and hop.answer:
                    ans_lower = hop.answer.lower().strip()
                    if ans_lower in ("unknown", "unresolved", "n/a", "none", "") and hop.attempt_count < 2:
                        hop.status = "pending"
                        hop.attempt_count += 1

            # Strategist reviews progress
            decision = await self.strategist.review(blackboard)
            mode = decision.get("mode", "investigate_more")

            if self.verbose:
                logger.info("Round %d review: mode=%s", round_idx, mode)

            if mode == "synthesize":
                break
            elif mode == "revise":
                blackboard.apply_revisions(decision.get("revisions", []))
            elif mode == "verify":
                verdict = await self.strategist.verify(blackboard)
                if verdict.get("approved", True):
                    break
                # Rejected: mark weak hops for re-investigation
                for hop_id in verdict.get("weak_hops", []):
                    if isinstance(hop_id, int) and 0 <= hop_id < len(blackboard.hop_chain):
                        hop = blackboard.hop_chain[hop_id]
                        if hop.attempt_count < 2:  # max 2 re-attempts
                            hop.status = "pending"
                            hop.attempt_count += 1
            # else: investigate_more — loop continues naturally

        # Phase 3: Final Answer
        if self.verbose:
            logger.info("Phase 3: Generating final answer")
        answer = await self.strategist.generate_answer(blackboard)

        elapsed = time.monotonic() - t0

        # Build PipelineResult
        agent_results = {}
        for hop_id, inv_result in total_inv_results.items():
            agent_results[hop_id] = AgentResult(
                sub_question_index=hop_id,
                answer=inv_result.get("answer", ""),
                evidence_doc_ids=inv_result.get("supporting_chunk_ids", []),
                trajectory=inv_result.get("trajectory", []),
                loops=inv_result.get("loops", 0),
                total_tokens=inv_result.get("total_retrieved_tokens", 0),
                wall_clock_seconds=0.0,
                confidence=inv_result.get("confidence", 0.5),
            )

        return PipelineResult(
            question=question,
            final_answer=answer,
            question_type=blackboard.question_type,
            num_sub_questions=len(blackboard.hop_chain),
            num_waves=min(self.max_rounds, investigator_counter),
            agent_results=agent_results,
            total_tokens=sum(r.total_tokens for r in agent_results.values()),
            wall_clock_seconds=elapsed,
        )

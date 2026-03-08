"""Coordinator: tick-based loop that drives all M6 agents.

Each tick: Decomposer → Retriever(s) → Critic → Synthesizer.
Enforces token budget, wall-clock timeout, and max tick limits.
Detects idle ticks (no agent acted) for early termination.
Salvages partial answers on early termination.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from multi_agent.m6.autonomous_agent import AutonomousAgent
from multi_agent.m6.blackboard import Blackboard
from multi_agent.m6.types import SubQuestionStatus

logger = logging.getLogger(__name__)

_MAX_IDLE_TICKS = 3  # Terminate after this many consecutive ticks with no agent acting


class Coordinator:
    """Tick-based coordinator for M6 blackboard agents."""

    def __init__(
        self,
        agents: list[AutonomousAgent],
        max_ticks: int = 30,
        token_budget: int = 200_000,
        wall_clock_timeout: float = 300.0,
    ):
        self.agents = agents
        self.max_ticks = max_ticks
        self.token_budget = token_budget
        self.wall_clock_timeout = wall_clock_timeout

    async def run(self, blackboard: Blackboard) -> str:
        """Run the tick loop until termination.

        Returns the final answer string.
        """
        t0 = time.monotonic()
        consecutive_idle = 0

        for tick in range(1, self.max_ticks + 1):
            if blackboard.terminated:
                break

            # Budget check
            if blackboard.tokens_used >= self.token_budget:
                logger.warning("Token budget exhausted (%d/%d) at tick %d",
                               blackboard.tokens_used, self.token_budget, tick)
                answer = await self._salvage_answer(blackboard)
                await blackboard.terminate(f"BUDGET_EXHAUSTED: {answer}")
                break

            # Timeout check
            elapsed = time.monotonic() - t0
            if elapsed >= self.wall_clock_timeout:
                logger.warning("Wall-clock timeout (%.1fs) at tick %d", elapsed, tick)
                answer = await self._salvage_answer(blackboard)
                await blackboard.terminate(f"TIMEOUT: {answer}")
                break

            # Idle detection — terminate early if no agent has work
            if consecutive_idle >= _MAX_IDLE_TICKS and tick > 1:
                logger.warning("No agent acted for %d consecutive ticks, terminating at tick %d",
                               consecutive_idle, tick)
                answer = await self._salvage_answer(blackboard)
                await blackboard.terminate(f"IDLE: {answer}")
                break

            # Tick all agents in order
            any_acted = False
            for agent in self.agents:
                if blackboard.terminated:
                    break
                try:
                    acted = await agent.tick(blackboard)
                    if acted:
                        any_acted = True
                except Exception as exc:
                    logger.error("Agent %s error at tick %d: %s",
                                 agent.agent_id, tick, exc)

            if any_acted:
                consecutive_idle = 0
            else:
                consecutive_idle += 1

            await blackboard.increment_tick()

            # Log progress periodically
            if tick % 5 == 0 or not any_acted:
                coord_obs = await blackboard.read_for_coordinator()
                logger.info(
                    "Tick %d: %s | tokens=%d/%d | backtracks=%d | idle=%d",
                    tick,
                    coord_obs["status_counts"],
                    coord_obs["tokens_used"],
                    coord_obs["token_budget"],
                    coord_obs["backtrack_count"],
                    consecutive_idle,
                )

        # Extract final answer from termination reason
        if blackboard.terminated:
            reason = blackboard.termination_reason
            # Extract answer from "SYNTHESIZED: answer" or "BUDGET_EXHAUSTED: answer" etc.
            if ":" in reason:
                answer = reason.split(":", 1)[1].strip()
            else:
                answer = reason
        else:
            # Max ticks exhausted without termination
            logger.warning("Max ticks (%d) exhausted without termination", self.max_ticks)
            answer = await self._salvage_answer(blackboard)
            await blackboard.terminate(f"MAX_TICKS: {answer}")

        elapsed = time.monotonic() - t0
        logger.info(
            "Coordinator finished: answer='%s' | ticks=%d | tokens=%d | %.1fs",
            answer[:60], blackboard.current_tick,
            blackboard.tokens_used, elapsed,
        )
        return answer

    async def _salvage_answer(self, blackboard: Blackboard) -> str:
        """Extract best available answer when terminating early."""
        obs = await blackboard.read_for_synthesizer()
        sqs = obs["sub_questions"]
        entities = obs["entity_registry"]

        def _is_usable(answer: str | None) -> bool:
            return bool(answer) and answer.lower() not in ("unknown", "none", "error", "")

        # Strategy 1: Last verified sub-question (from entities or answer)
        verified_sqs = [sq for sq in sqs if sq["status"] == SubQuestionStatus.VERIFIED.value]
        if verified_sqs:
            last = max(verified_sqs, key=lambda sq: sq["id"])
            entity_key = f"answer_{last['id']}"
            val = entities.get(entity_key) or last.get("answer")
            if _is_usable(val):
                return val

        # Strategy 2: Any verified sub-question with a usable answer
        for sq in sorted(verified_sqs, key=lambda sq: sq["id"], reverse=True):
            val = entities.get(f"answer_{sq['id']}") or sq.get("answer")
            if _is_usable(val):
                return val

        # Strategy 3: Any sub-question with a usable answer (even unverified)
        for sq in sorted(sqs, key=lambda sq: sq["id"], reverse=True):
            if _is_usable(sq.get("answer")):
                return sq["answer"]

        # Strategy 4: Any entity value
        for val in reversed(list(entities.values())):
            if _is_usable(val):
                return val

        return ""

    async def _all_terminal(self, blackboard: Blackboard) -> bool:
        """Check if all sub-questions are in terminal state."""
        obs = await blackboard.read_for_coordinator()
        total = obs["total_sub_questions"]
        if total == 0:
            return False
        counts = obs["status_counts"]
        verified = counts.get("verified", 0)
        failed = counts.get("failed", 0)
        return (verified + failed) >= total

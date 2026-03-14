"""Coordinator: concurrent agent loops for M6 blackboard coordination.

Runs all agents as independent async loops. Each agent autonomously decides
when to act via the observe → should_act → act cycle. A watchdog monitors
termination conditions (token budget, wall-clock timeout, idle detection,
max actions). Coordination is emergent through the shared blackboard state.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from multi_agent.blackboard import Blackboard

logger = logging.getLogger(__name__)


class Coordinator:
    """Concurrent coordinator for M6 blackboard agents.

    Each agent runs as an independent async loop, reading from and writing to
    the shared blackboard. Agents react to state changes on the blackboard
    rather than being sequenced by the coordinator.
    """

    def __init__(
        self,
        agents: list,
        token_budget: int = 200_000,
        wall_clock_timeout: float = 300.0,
        idle_timeout: float = 30.0,
        max_actions: int = 100,
    ):
        self.agents = agents
        self.token_budget = token_budget
        self.wall_clock_timeout = wall_clock_timeout
        self.idle_timeout = idle_timeout
        self.max_actions = max_actions

    async def run(self, blackboard: Blackboard) -> str:
        """Run all agents concurrently until termination.

        Returns the final answer string.
        """
        t0 = time.monotonic()
        blackboard._last_action_time = t0

        tasks = [
            self._agent_loop(agent, blackboard)
            for agent in self.agents
        ]
        tasks.append(self._watchdog(blackboard, t0))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.error("Agent loop error: %s", r)

        answer = blackboard.final_answer or ""
        if not answer:
            answer = await blackboard.salvage_answer()
            if not blackboard.terminated:
                await blackboard.set_final_answer(answer)
                await blackboard.terminate("SALVAGED")

        elapsed = time.monotonic() - t0
        logger.info(
            "Coordinator finished: '%s' | actions=%d | tokens=%d | %.1fs",
            answer[:60], blackboard.current_tick,
            blackboard.tokens_used, elapsed,
        )
        return answer

    async def _agent_loop(self, agent: Any, blackboard: Blackboard) -> None:
        """Run a single agent's observe/act loop until blackboard terminates."""
        backoff = 0.05
        max_backoff = 2.0
        loop_count = 0

        while not blackboard.terminated:
            try:
                acted = await agent.tick(blackboard)
                loop_count += 1
                if acted:
                    backoff = 0.05
                else:
                    if loop_count <= 3 or loop_count % 50 == 0:
                        logger.debug(
                            "Agent %s: no action (loop %d, backoff %.2fs)",
                            agent.agent_id, loop_count, backoff,
                        )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.5, max_backoff)
            except Exception as exc:
                logger.error("Agent %s error: %s", agent.agent_id, exc, exc_info=True)
                await asyncio.sleep(1.0)

    async def _watchdog(self, blackboard: Blackboard, t0: float) -> None:
        """Monitor termination conditions: budget, timeout, idle, max actions."""
        last_log_time = t0

        while not blackboard.terminated:
            await asyncio.sleep(1.0)
            obs = await blackboard.read_for_coordinator()
            elapsed = time.monotonic() - t0

            if obs["tokens_used"] >= self.token_budget:
                logger.warning("Token budget exhausted (%d/%d)",
                               obs["tokens_used"], self.token_budget)
                await self._terminate_with_salvage(blackboard, "BUDGET_EXHAUSTED")
                return

            if elapsed >= self.wall_clock_timeout:
                logger.warning("Wall-clock timeout (%.1fs)", elapsed)
                await self._terminate_with_salvage(blackboard, "TIMEOUT")
                return

            idle_seconds = time.monotonic() - obs.get("last_action_time", t0)
            if idle_seconds >= self.idle_timeout and obs["action_count"] > 0:
                logger.warning("Idle for %.1fs after %d actions, terminating",
                               idle_seconds, obs["action_count"])
                await self._terminate_with_salvage(blackboard, "IDLE")
                return

            if obs["action_count"] >= self.max_actions:
                logger.warning("Max actions (%d) reached", self.max_actions)
                await self._terminate_with_salvage(blackboard, "MAX_ACTIONS")
                return

            now = time.monotonic()
            if now - last_log_time >= 10.0:
                logger.info(
                    "Watchdog [%.0fs]: %s | tokens=%d/%d | actions=%d | idle=%.1fs",
                    elapsed, obs["status_counts"],
                    obs["tokens_used"], obs["token_budget"],
                    obs["action_count"], idle_seconds,
                )
                last_log_time = now

    async def _terminate_with_salvage(
        self, blackboard: Blackboard, reason: str,
    ) -> None:
        """Salvage best available answer and terminate."""
        answer = await blackboard.salvage_answer()
        await blackboard.set_final_answer(answer)
        await blackboard.terminate(reason)

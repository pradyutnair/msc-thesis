"""M5 Pipeline: Multi-agent wrapper around E2-style subagents.

An orchestrator LLM (thinking ON) decomposes multi-hop questions and
delegates sub-tasks to E2 subagents (thinking OFF, BaseAgent with raw
retrieval tools).  No new tool classes — just BaseAgent made callable
from a coordination loop.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from arag.agent.base import BaseAgent
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def strip_thinking(content: str) -> str:
    """Remove <think>…</think> blocks from LLM output."""
    if not content:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def parse_delegate(content: str) -> str:
    """Extract the task string after DELEGATE: (first occurrence)."""
    match = re.search(r"DELEGATE:\s*(.+)", content)
    if match:
        return match.group(1).strip()
    return content.strip()


class M5Pipeline:
    """Multi-agent pipeline: orchestrator decomposes, E2 subagents execute.

    Not a BaseAgent itself — the orchestrator is a raw LLM chat loop,
    and each delegation spawns a fresh BaseAgent with E2's tools.
    """

    def __init__(
        self,
        orchestrator_llm: LLMClient,
        subagent_llm: LLMClient,
        shared_tools: ToolRegistry,
        orchestrator_prompt: str,
        subagent_prompt: str,
        max_iterations: int = 10,
        subagent_max_loops: int = 5,
        subagent_max_budget: int = 32000,
    ):
        self.orch_llm = orchestrator_llm
        self.sub_llm = subagent_llm
        self.shared_tools = shared_tools
        self.orch_prompt = orchestrator_prompt
        self.sub_prompt = subagent_prompt
        self.max_iterations = max_iterations
        self.sub_max_loops = subagent_max_loops
        self.sub_max_budget = subagent_max_budget

    def _run_subagent(self, task: str) -> dict[str, Any]:
        """Spawn a fresh E2 subagent and run it on *task*."""
        agent = BaseAgent(
            llm_client=self.sub_llm,
            tools=self.shared_tools,
            system_prompt=self.sub_prompt,
            max_loops=self.sub_max_loops,
            max_token_budget=self.sub_max_budget,
        )
        return agent.run(task)

    def run(self, question: str) -> dict[str, Any]:
        """Run the orchestrator loop on a single question.

        Returns a dict compatible with the batch runner:
          answer, trajectory, total_cost, loops, findings, wall_clock_seconds, error
        """
        t0 = time.monotonic()

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.orch_prompt},
            {"role": "user", "content": question},
        ]
        trajectory: list[dict[str, Any]] = []
        total_cost = 0.0
        findings: list[dict[str, str]] = []

        for iteration in range(1, self.max_iterations + 1):
            # ── Orchestrator turn ────────────────────────────────────
            try:
                response = self.orch_llm.chat(messages=messages)
            except Exception as exc:
                logger.error("Orchestrator LLM error at iter %d: %s", iteration, exc)
                return self._build_result(
                    answer=f"Error: {exc}",
                    trajectory=trajectory,
                    total_cost=total_cost,
                    loops=iteration,
                    findings=findings,
                    t0=t0,
                    error=str(exc),
                )

            total_cost += response.get("cost", 0.0)
            msg = response["message"]
            raw_content = msg.get("content", "")
            content = strip_thinking(raw_content)
            messages.append(msg)

            trajectory.append({
                "role": "orchestrator",
                "iteration": iteration,
                "content": content,
                "raw_content": raw_content,
            })

            # ── Parse: ANSWER ────────────────────────────────────────
            if "ANSWER:" in content:
                answer = content.split("ANSWER:", 1)[1].strip()
                # Strip any trailing text after the first line
                answer = answer.split("\n")[0].strip()
                return self._build_result(
                    answer=answer,
                    trajectory=trajectory,
                    total_cost=total_cost,
                    loops=iteration,
                    findings=findings,
                    t0=t0,
                )

            # ── Parse: DELEGATE ──────────────────────────────────────
            if "DELEGATE:" in content:
                task = parse_delegate(content)
                logger.info(
                    "Iter %d — DELEGATE: %s", iteration, task[:80],
                )

                try:
                    sub_result = self._run_subagent(task)
                except Exception as exc:
                    logger.error("Subagent error: %s", exc)
                    sub_result = {
                        "answer": f"Error: {exc}",
                        "trajectory": [],
                        "total_cost": 0.0,
                    }

                total_cost += sub_result.get("total_cost", 0.0)
                sub_traj = sub_result.get("trajectory", [])
                trajectory.append({
                    "role": "subagent",
                    "iteration": iteration,
                    "task": task,
                    "answer": sub_result.get("answer", ""),
                    "loops": sub_result.get("loops", 0),
                    "trajectory": sub_traj,
                })

                finding = sub_result.get("answer", "")
                findings.append({"task": task, "finding": finding})

                # Feed back to orchestrator
                messages.append({
                    "role": "user",
                    "content": f"Subagent result: {finding}",
                })
                continue

            # ── No DELEGATE or ANSWER — force a delegation ─────────
            if iteration == 1 and not findings:
                # First turn with no delegation: auto-delegate the full question
                logger.warning(
                    "Iter %d — no DELEGATE/ANSWER keyword, auto-delegating original question",
                    iteration,
                )
                task = question
                try:
                    sub_result = self._run_subagent(task)
                except Exception as exc:
                    logger.error("Auto-delegate subagent error: %s", exc)
                    sub_result = {
                        "answer": f"Error: {exc}",
                        "trajectory": [],
                        "total_cost": 0.0,
                    }

                total_cost += sub_result.get("total_cost", 0.0)
                trajectory.append({
                    "role": "subagent",
                    "iteration": iteration,
                    "task": task,
                    "answer": sub_result.get("answer", ""),
                    "loops": sub_result.get("loops", 0),
                    "trajectory": sub_result.get("trajectory", []),
                    "auto_delegated": True,
                })

                finding = sub_result.get("answer", "")
                findings.append({"task": task, "finding": finding})

                messages.append({
                    "role": "user",
                    "content": f"Subagent result: {finding}",
                })
                continue
            else:
                # Later turn or already have findings — treat as final answer
                logger.warning(
                    "Iter %d — no DELEGATE/ANSWER keyword, treating as final answer",
                    iteration,
                )
                return self._build_result(
                    answer=content,
                    trajectory=trajectory,
                    total_cost=total_cost,
                    loops=iteration,
                    findings=findings,
                    t0=t0,
                )

        # ── Max iterations exhausted ─────────────────────────────────
        logger.warning("Max iterations (%d) reached, forcing answer", self.max_iterations)
        messages.append({
            "role": "user",
            "content": "Maximum iterations reached. You must give your final ANSWER: now.",
        })

        try:
            response = self.orch_llm.chat(messages=messages)
            total_cost += response.get("cost", 0.0)
            raw_content = response["message"].get("content", "")
            content = strip_thinking(raw_content)
        except Exception as exc:
            logger.error("Final orchestrator call failed: %s", exc)
            # Fall back to last finding
            content = findings[-1]["finding"] if findings else "unknown"

        trajectory.append({
            "role": "orchestrator",
            "iteration": self.max_iterations + 1,
            "content": content,
            "forced": True,
        })

        if "ANSWER:" in content:
            answer = content.split("ANSWER:", 1)[1].strip().split("\n")[0].strip()
        else:
            answer = content

        return self._build_result(
            answer=answer,
            trajectory=trajectory,
            total_cost=total_cost,
            loops=self.max_iterations,
            findings=findings,
            t0=t0,
        )

    @staticmethod
    def _build_result(
        answer: str,
        trajectory: list[dict],
        total_cost: float,
        loops: int,
        findings: list[dict],
        t0: float,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "answer": answer,
            "trajectory": trajectory,
            "total_cost": total_cost,
            "loops": loops,
            "findings": findings,
            "wall_clock_seconds": time.monotonic() - t0,
            "error": error,
        }

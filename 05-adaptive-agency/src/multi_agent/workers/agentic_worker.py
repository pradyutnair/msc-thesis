"""AgenticWorker: ReAct tool-calling loop for a single assigned sub-question.

Adapted from M6's WorkerAgent. Key difference: each worker is assigned a
specific sub-question ID at creation and only claims that SQ.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from arag.core.context import AgentContext
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.autonomous_agent import AutonomousAgent
from multi_agent.blackboard import Blackboard
from multi_agent.memory import Memory
from multi_agent.types import EvidenceEntry, RetrievalMode, SubQuestionStatus
from multi_agent.utils import clean_answer, resolve_placeholders

logger = logging.getLogger(__name__)

_PLAN_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "worker_plan.txt"


class AgenticWorker(AutonomousAgent):
    """ReAct tool-calling worker assigned to a specific sub-question."""

    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        tools: ToolRegistry,
        assigned_sq_id: int,
        plan_prompt_path: str | Path | None = None,
        max_steps: int = 8,
    ):
        super().__init__(agent_id=agent_id, agent_type="agentic_worker")
        self.llm = llm_client
        self.tools = tools
        self.assigned_sq_id = assigned_sq_id
        self.max_steps = max_steps
        self._done = False
        self._last_epoch = 0

        path = Path(plan_prompt_path) if plan_prompt_path else _PLAN_PROMPT_PATH
        self._plan_template = path.read_text(encoding="utf-8")

    async def observe(self, blackboard: Blackboard) -> dict[str, Any]:
        return await blackboard.read_for_worker(self.agent_id)

    def should_act(self, observation: dict[str, Any]) -> bool:
        epoch = observation.get("redecomposition_epoch", 0)
        if epoch > self._last_epoch:
            self._done = False
            self._last_epoch = epoch
        if self._done:
            return False
        if observation["claimed_sub_question"] is not None:
            return True
        for sq in observation["available_sub_questions"]:
            if sq["id"] == self.assigned_sq_id:
                return True
        return False

    async def act(self, observation: dict[str, Any], blackboard: Blackboard) -> int:
        sq_dict = observation["claimed_sub_question"]
        if sq_dict is None:
            claimed = await blackboard.claim_sub_question(self.assigned_sq_id, self.agent_id)
            if not claimed:
                return 0
            observation = await self.observe(blackboard)
            sq_dict = observation["claimed_sub_question"]
            if sq_dict is None:
                return 0

        sq_id = sq_dict["id"]
        entity_registry = observation["entity_registry"]
        resolved_text = resolve_placeholders(sq_dict["text"], entity_registry)

        original_question = observation.get("question", resolved_text)
        blackboard_context = observation.get("blackboard_context", "")
        search_queries = observation.get("search_queries", [])
        warm_start_context = observation.get("warm_start_context", "")

        resolved_queries = [resolve_placeholders(q, entity_registry) for q in search_queries]

        loop = asyncio.get_running_loop()
        solve_future = loop.run_in_executor(
            None,
            self._solve,
            sq_id, resolved_text, original_question,
            blackboard_context, entity_registry,
            resolved_queries, warm_start_context,
        )

        async def _heartbeat():
            while True:
                await asyncio.sleep(30)
                await blackboard.record_action(0)

        heartbeat_task = asyncio.create_task(_heartbeat())
        try:
            answer, evidence, tokens = await solve_future
        finally:
            heartbeat_task.cancel()

        answer = clean_answer(answer)

        # Clear verbose refusal answers
        if len(answer) > 60:
            answer_lower = answer.lower()
            _verbose_patterns = [
                "the evidence does not", "does not mention", "not explicitly mentioned",
                "no evidence confirms", "not specified in", "the provided documents",
                "there is no ", "cannot be determined", "not found in the",
            ]
            if any(p in answer_lower for p in _verbose_patterns):
                logger.info("%s: SQ-%d verbose/refusal answer cleared", self.agent_id, sq_id)
                answer = ""

        await blackboard.post_evidence(evidence, sq_id, answer, self.agent_id)

        is_usable = bool(answer) and answer.lower() not in ("unknown", "error", "")
        await blackboard.verify_sub_question(sq_id, verified=is_usable)
        await blackboard.record_mode_tokens(RetrievalMode.AGENTIC, tokens)

        self._done = is_usable
        logger.info(
            "%s: SQ-%d -> '%s' (%d evidence, %d tokens, done=%s)",
            self.agent_id, sq_id, answer[:60], len(evidence), tokens, self._done,
        )
        return tokens

    def _solve(
        self,
        sq_id: int,
        sq_text: str,
        original_question: str,
        blackboard_context: str,
        entity_registry: dict[str, str],
        search_queries: list[str] | None = None,
        warm_start_context: str = "",
    ) -> tuple[str, list[EvidenceEntry], int]:
        context = AgentContext()
        memory = Memory()
        tool_schemas = self.tools.get_all_schemas()

        system_prompt = self._build_system_prompt(
            sq_text, original_question, blackboard_context,
            entity_registry,
            search_queries=search_queries or [],
            warm_start_context=warm_start_context,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sq_text},
        ]

        total_tokens = 0
        answer = ""

        for step in range(1, self.max_steps + 1):
            response = self.llm.chat(
                messages=messages, tools=tool_schemas, temperature=0.0,
            )
            total_tokens += int(response.get("cost", 0.0) * 1_000_000)

            assistant_msg = response["message"]
            messages.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls", [])

            if tool_calls:
                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    func_args = json.loads(tc["function"]["arguments"])
                    tool_result, _ = self.tools.execute(
                        func_name, context, **func_args,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    })
                    memory.add_action(step, func_name, func_args, tool_result)
            else:
                answer = assistant_msg.get("content", "")
                break

        evidence = self._build_evidence(sq_id, memory)
        return answer, evidence, total_tokens

    # ── Helpers ────────────────────────────────────────────────────────

    def _build_system_prompt(
        self,
        sq_text: str,
        original_question: str,
        blackboard_context: str,
        entity_registry: dict[str, str],
        search_queries: list[str] | None = None,
        warm_start_context: str = "",
    ) -> str:
        context_parts = [blackboard_context] if blackboard_context else []
        if entity_registry:
            resolved = [f"{k} = {v}" for k, v in entity_registry.items()]
            context_parts.append("Resolved entities: " + ", ".join(resolved))

        full_context = (
            "\n".join(context_parts) if context_parts
            else "No findings from other agents yet."
        )

        if search_queries:
            queries_str = "\n".join(f"- {q}" for q in search_queries)
        else:
            queries_str = "(no pre-planned queries -- use your own search strategy)"

        warm_str = warm_start_context if warm_start_context else "No warm-start context available."

        result = self._plan_template
        result = result.replace("{sub_question}", sq_text)
        result = result.replace("{original_question}", original_question or sq_text)
        result = result.replace("{blackboard_context}", full_context)
        result = result.replace("{search_queries}", queries_str)
        result = result.replace("{warm_start_context}", warm_str)
        return result

    def _build_evidence(self, sq_id: int, memory: Memory) -> list[EvidenceEntry]:
        entries: list[EvidenceEntry] = []
        for cid, content in memory.get_read_chunks():
            entries.append(EvidenceEntry(
                id="",
                sub_question_id=sq_id,
                content=content[:2000],
                source_chunk_id=cid,
                relevance_score=0.5,
                retriever_id=self.agent_id,
            ))
        return entries

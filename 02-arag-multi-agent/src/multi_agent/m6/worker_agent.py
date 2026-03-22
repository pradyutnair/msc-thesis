"""WorkerAgent: plan -> execute loop per sub-question.

Each worker is a fully autonomous agent that:
1. Claims a sub-question from the blackboard
2. Runs an iterative plan/execute loop with tool access
3. Posts evidence and answer to the blackboard
4. Maintains structured memory of all actions
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
from multi_agent.m6.autonomous_agent import AutonomousAgent
from multi_agent.m6.blackboard import Blackboard
from multi_agent.m6.memory import Memory
from multi_agent.m6.types import EvidenceEntry, SubQuestionStatus

logger = logging.getLogger(__name__)

_PLAN_PROMPT_PATH = Path(__file__).parent / "prompts" / "worker_plan.txt"


class WorkerAgent(AutonomousAgent):
    """Autonomous worker: plan -> execute loop per sub-question."""

    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        tools: ToolRegistry,
        plan_prompt_path: str | Path | None = None,
        max_steps: int = 8,
        enable_extraction_pass: bool = False,
        enable_answer_validation: bool = False,
        enable_bridge_guard: bool = False,
    ):
        super().__init__(agent_id=agent_id, agent_type="worker")
        self.llm = llm_client
        self.tools = tools
        self.max_steps = max_steps

        path = Path(plan_prompt_path) if plan_prompt_path else _PLAN_PROMPT_PATH
        self._plan_template = path.read_text(encoding="utf-8")

    async def observe(self, blackboard: Blackboard) -> dict[str, Any]:
        return await blackboard.read_for_worker(self.agent_id)

    def should_act(self, observation: dict[str, Any]) -> bool:
        if observation["claimed_sub_question"] is not None:
            return True
        avail = observation["available_sub_questions"]
        if avail:
            logger.debug(
                "%s: should_act=True, %d available SQs: %s",
                self.agent_id, len(avail),
                [(sq["id"], sq["status"]) for sq in avail],
            )
        return len(avail) > 0

    async def act(self, observation: dict[str, Any], blackboard: Blackboard) -> int:
        sq_dict = observation["claimed_sub_question"]
        if sq_dict is None:
            sq_dict = await self._try_claim(observation, blackboard)
            if sq_dict is None:
                return 0

        sq_id = sq_dict["id"]
        entity_registry = observation["entity_registry"]
        resolved_text = self._resolve_placeholders(sq_dict["text"], entity_registry)

        original_question = observation.get("question", resolved_text)
        blackboard_context = observation.get("blackboard_context", "")
        search_queries = observation.get("search_queries", [])
        dependency_chunk_ids = observation.get("dependency_chunk_ids", [])

        resolved_queries = [self._resolve_placeholders(q, entity_registry) for q in search_queries]
        unknown_entities = sq_dict.get('unknown_entities', [])

        loop = asyncio.get_running_loop()

        solve_future = loop.run_in_executor(
            None,
            self._solve,
            sq_id,
            resolved_text,
            original_question,
            blackboard_context,
            entity_registry,
            resolved_queries,
            dependency_chunk_ids,
            unknown_entities,
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

        answer = self._clean_answer(answer)

        # Clear verbose refusal answers
        if len(answer) > 60:
            answer_lower = answer.lower()
            _verbose_patterns = [
                "the evidence does not", "does not mention", "not explicitly mentioned",
                "no evidence confirms", "not specified in", "the provided documents",
                "there is no ", "cannot be determined", "not found in the",
            ]
            if any(p in answer_lower for p in _verbose_patterns):
                logger.info("%s: SQ-%d verbose/refusal answer cleared: '%s'", self.agent_id, sq_id, answer[:60])
                answer = ""

        await blackboard.post_evidence(evidence, sq_id, answer, self.agent_id)

        is_usable = bool(answer) and answer.lower() not in ("unknown", "error", "")
        await blackboard.verify_sub_question(sq_id, verified=is_usable)

        logger.info(
            "%s: SQ-%d -> '%s' (%d evidence, %d tokens)",
            self.agent_id, sq_id, answer[:60], len(evidence), tokens,
        )
        return tokens

    # ── Synchronous solve loop ────────────────────────────────────────

    def _solve(
        self,
        sq_id: int,
        sq_text: str,
        original_question: str,
        blackboard_context: str,
        entity_registry: dict[str, str],
        search_queries: list[str] | None = None,
        dependency_chunk_ids: list[str] | None = None,
        unknown_entities: list[str] | None = None,
    ) -> tuple[str, list[EvidenceEntry], int]:
        """Plan -> execute loop. Runs in a thread executor."""
        context = AgentContext()
        memory = Memory()
        tool_schemas = self.tools.get_all_schemas()

        system_prompt = self._build_system_prompt(
            sq_text, original_question, blackboard_context,
            entity_registry,
            search_queries=search_queries or [],
            dependency_chunk_ids=dependency_chunk_ids or [],
            unknown_entities=unknown_entities or [],
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

        if not answer or answer.lower() in ("unknown", "error", ""):
            answer = self._fallback_answer_from_memory(sq_text, memory)

        evidence = self._build_evidence(sq_id, memory)
        return answer, evidence, total_tokens

    # ── Helpers ────────────────────────────────────────────────────────

    async def _try_claim(
        self, observation: dict[str, Any], blackboard: Blackboard,
    ) -> dict[str, Any] | None:
        """Claim a sub-question. Priority: retries > most dependents > lowest ID."""
        available = observation["available_sub_questions"]
        logger.info("%s: trying to claim from %d available SQs", self.agent_id, len(available))
        available.sort(
            key=lambda sq: (
                0 if sq["status"] == SubQuestionStatus.NEEDS_RETRY.value else 1,
                -sq.get("num_dependents", 0),
                sq["id"],
            ),
        )
        for sq_dict in available:
            if await blackboard.claim_sub_question(sq_dict["id"], self.agent_id):
                return sq_dict
        return None

    def _resolve_placeholders(
        self, text: str, entity_registry: dict[str, str],
    ) -> str:
        """Replace [answer_N] placeholders with resolved entity values."""
        def replacer(match: re.Match) -> str:
            return entity_registry.get(match.group(1), match.group(0))
        return re.sub(r"\[(answer_\d+)\]", replacer, text)

    def _build_system_prompt(
        self,
        sq_text: str,
        original_question: str,
        blackboard_context: str,
        entity_registry: dict[str, str],
        search_queries: list[str] | None = None,
        dependency_chunk_ids: list[str] | None = None,
        unknown_entities: list[str] | None = None,
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

        if dependency_chunk_ids:
            dep_str = "Sibling agents found these relevant chunks. Call read_chunk on them:\n"
            dep_str += ", ".join(dependency_chunk_ids[:10])
        else:
            dep_str = "No dependency chunks available."

        # Build answer type hint from unknown_entities
        if unknown_entities:
            answer_type_hint = 'Your answer must be: ' + ', '.join(unknown_entities) + '.'
        else:
            answer_type_hint = 'Answer with a single entity (name, date, number, or place).'

        # Use .replace() to avoid crashes on content with { or }
        result = self._plan_template
        result = result.replace("{sub_question}", sq_text)
        result = result.replace("{original_question}", original_question or sq_text)
        result = result.replace("{blackboard_context}", full_context)
        result = result.replace("{search_queries}", queries_str)
        result = result.replace("{dependency_chunks}", dep_str)
        result = result.replace("{answer_type_hint}", answer_type_hint)
        return result

    def _clean_answer(self, answer: str) -> str:
        # Strip think tags
        answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)
        answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL)

        # Strip markdown
        answer = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)
        answer = re.sub(r"\*(.+?)\*", r"\1", answer)

        # Take first non-empty line
        for line in answer.split("\n"):
            line = line.strip()
            if line and not re.fullmatch(r"[=\-]{5,}", line):
                answer = line
                break
        else:
            answer = answer.split("\n")[0].strip()

        answer = answer.strip().strip("\"'`*")

        # Strip chunk ID references that leak from search_and_read output
        if re.fullmatch(r"\[Chunk\s+\d+\]", answer.strip()):
            return ""
        answer = re.sub(r"\[Chunk\s+\d+\]", "", answer).strip()

        # Strip separator lines that leak from tool output
        if re.fullmatch(r"[=\-]{10,}", answer.strip()):
            return ""

        # Strip "The answer is X" -> X
        answer = re.sub(r"^(?:The\s+)?(?:final\s+)?answer\s+is\s+", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"^(?:FINAL\s+)?ANSWER\s*:\s*", "", answer, flags=re.IGNORECASE)

        # Strip reasoning prefixes
        answer = re.sub(
            r"^(?:Based\s+on|According\s+to)\s+(?:the\s+)?(?:evidence|information|documents|context|provided|retrieved)[^,]*,\s*",
            "", answer, flags=re.IGNORECASE,
        )

        # Extract entity from verbose sentence patterns
        for pattern in [
            # "The nationality of X is Y" -> Y
            r"^(?:The\s+)?(?:nationality|country|birthplace|director|city|region|publisher|performer|composer|author|record\s+label)\s+(?:of\s+.+?\s+)?(?:is|was)\s+(.+?)$",
            # "X's birthplace is Y" -> Y
            r"^.+?(?:'s|s')\s+(?:nationality|country|birthplace|birth\s*date|birth\s*place)\s+(?:is|was)\s+(.+?)$",
            # "The city where X is Y" -> Y
            r"^The\s+(?:name\s+of\s+the\s+)?(?:city|region|country|person|film|body\s+of\s+water)\s+(?:where|that|which|by)\s+.+\s+(?:is|was)\s+(.+?)$",
            # "X was born in Y" -> Y
            r"^.+?\s+(?:was|is)\s+born\s+(?:in|on)\s+(.+?)$",
            # "X is located in Y" -> Y
            r"^.+?\s+(?:was|is)\s+(?:located|based|situated|headquartered)\s+(?:in|at)\s+(.+?)$",
            # "X was released in Y" -> Y
            r"^.+?\s+(?:was|is)\s+(?:released|published|produced|founded|formed|created|established)\s+(?:in|on|by)\s+(.+?)$",
            # "X was directed by Y" -> Y
            r"^.+?\s+(?:was|is)\s+(?:directed|composed|written|designed|performed)\s+by\s+(.+?)$",
            # "X died in Y" -> Y
            r"^.+?\s+(?:died|passed\s+away)\s+(?:in|on|at)\s+(.+?)$",
        ]:
            m = re.match(pattern, answer, re.IGNORECASE)
            if m:
                extracted = m.group(m.lastindex).strip().strip("\"'`.,;:!?")
                if extracted and 2 < len(extracted) < len(answer):
                    answer = extracted
                    break

        # Strip trailing punctuation
        answer = re.sub(r"\s*[\.,;:!?]+$", "", answer)
        # Strip parenthetical annotations
        answer = re.sub(r"\s*\((?:born|died|circa|c\.|approximately).*?\)", "", answer, flags=re.IGNORECASE).strip()

        return answer


    def _fallback_answer_from_memory(self, sq_text: str, memory: Memory) -> str:
        """Last-resort answer extraction from read_chunk results."""
        chunks = memory.get_read_chunks()
        if not chunks:
            return ""
        last_content = chunks[-1][1][:500]
        return last_content.split(".")[0].strip() if "." in last_content else ""

    def _build_evidence(
        self, sq_id: int, memory: Memory,
    ) -> list[EvidenceEntry]:
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

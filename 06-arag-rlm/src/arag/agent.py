"""RLM-style agent built on top of the ARAG tool-calling loop."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from arag.core.context import AgentContext
from arag.core.llm import LLMClient
from arag.tools.build_tools import build_tool_bundle
from arag.tools.delegate import DelegateTool
from arag.tools.finish import FinishTool
from arag.tools.recursive_search import RecursiveSearchTool
from arag.tools.registry import ToolRegistry

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)
_FINISH_SENTINEL_RE = re.compile(r"__FINISH__:\s*(.+)", re.DOTALL)


def _strip_thinking(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", str(text or ""))
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


def _extract_finish_answer(text: str) -> str | None:
    cleaned = _strip_thinking(text)
    sentinel = _FINISH_SENTINEL_RE.search(cleaned)
    if sentinel:
        return sentinel.group(1).strip()

    if "finish" not in cleaned and '"finish"' not in cleaned:
        return None

    patterns = [
        r'finish\s*\(\s*\{[^)]*?"answer"\s*:\s*"((?:[^"\\]|\\.)*)"',
        r'finish\s*\(\s*answer\s*=\s*["\']([^"\']*)["\']',
        r'finish\s*\(\s*["\']([^"\']{1,300})["\']',
        r'"answer"\s*:\s*"((?:[^"\\]|\\.){1,300})"',
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            return match.group(1).replace('\\"', '"').strip()
    return None


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


class RLMAgent:
    """Recursive long-context agent with one-level delegation."""

    def __init__(
        self,
        llm_client: LLMClient,
        chunks_dict: Dict[str, str] | None,
        embedding_index: Any,
        chunks_file: str | None,
        index_dir: str | None,
        embedding_model: Any,
        config,
        depth: int = 0,
        max_depth: int = 1,
        parent_question: str | None = None,
        expected_answer_type: str | None = None,
        token_budget: int = 50000,
        tool_bundle: Dict[str, Any] | None = None,
        parent_evidence: str = "",
    ):
        self.llm = llm_client
        self.config = config
        self.depth = depth
        self.max_depth = max_depth
        self.parent_question = parent_question
        self.expected_answer_type = expected_answer_type
        self.token_budget = token_budget
        self.embedding_index = embedding_index
        self.chunks_file = chunks_file
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.parent_evidence = parent_evidence

        self.project_root = Path(__file__).resolve().parents[2]
        self.prompts_dir = self.project_root / "prompts"
        self.root_prompt_template = _load_text(self.prompts_dir / "root_agent.txt")
        self.child_prompt_template = _load_text(self.prompts_dir / "child_agent.txt")

        self.tool_bundle = tool_bundle or build_tool_bundle(config)
        self.chunks_dict = chunks_dict or self.tool_bundle.get("chunks_dict") or {}
        self.tools = self._register_tools()

    def _cfg(self, dotted: str, default: Any) -> Any:
        if hasattr(self.config, "get"):
            return self.config.get(dotted, default)
        return default

    def _register_tools(self) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(self.tool_bundle["keyword_tool"])
        reg.register(self.tool_bundle["read_tool"])
        if self.tool_bundle.get("semantic_tool") is not None:
            reg.register(self.tool_bundle["semantic_tool"])
        reg.register(self.tool_bundle["search_and_read_tool"])
        # Recursive search: wide retrieval + LLM filter + targeted read
        deps = self.tool_bundle.get("_recursive_search_deps")
        if deps is not None:
            kw, sem, rd = deps
            reg.register(RecursiveSearchTool(kw, sem, rd, self.llm))
        if self.depth == 0 and self.max_depth > 0:
            reg.register(
                DelegateTool(
                    agent_factory=self._create_child_agent,
                    config=self.config,
                    remaining_budget_fn=self._remaining_budget,
                    get_parent_evidence_fn=self._get_parent_evidence,
                )
            )
        reg.register(FinishTool())
        return reg

    def _build_system_prompt(self) -> str:
        if self.depth == 0:
            return self.root_prompt_template
        prompt = self.child_prompt_template.format(
            main_question=self.parent_question or "",
            expected_answer_type=self.expected_answer_type or "short answer",
            delegate_section="Nested delegation is disabled. Do not attempt to call delegate.",
        )
        if self.parent_evidence:
            prompt += (
                "\n\n## Parent Evidence (already retrieved)\n"
                "The parent agent already found these chunks. Use them as starting context — "
                "you may find the answer here without searching.\n\n"
                + self.parent_evidence
            )
        return prompt

    def _remaining_budget(self, context: AgentContext) -> int:
        return max(self.token_budget - context.total_llm_tokens, 0)

    def _get_parent_evidence(self, context: AgentContext) -> str:
        """Collect already-read chunk texts for Parent Evidence Passing (PEP)."""
        read_tool = self.tool_bundle.get("read_tool")
        if read_tool is None or not context.read_chunk_ids:
            return ""
        chunks_dict = getattr(read_tool, "chunks_dict", {}) or {}
        parts = []
        for cid in list(context.read_chunk_ids)[:8]:  # cap at 8 chunks
            text = chunks_dict.get(str(cid), "")
            if text:
                parts.append(f"[Chunk {cid}] {text[:600]}")
        return "\n\n".join(parts)

    def _merge_reasoning(self, message: Dict[str, Any]) -> Dict[str, Any]:
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if reasoning:
            existing = message.get("content") or ""
            message["content"] = "<think>\n" + reasoning + "\n</think>\n\n" + existing
            message.pop("reasoning_content", None)
            message.pop("reasoning", None)
        return message

    def _finish_schemas(self) -> List[Dict[str, Any]]:
        finish_tool = self.tools.get("finish")
        return [finish_tool.get_schema()] if finish_tool is not None else []

    def _record_llm_usage(self, context: AgentContext, phase: str, response: Dict[str, Any]) -> int:
        input_tokens = int(response.get("input_tokens", 0))
        output_tokens = int(response.get("output_tokens", 0))
        context.add_llm_usage(
            phase=phase,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={"depth": self.depth},
        )
        return input_tokens + output_tokens

    def _build_evidence_preview(self, trajectory: List[Dict[str, Any]]) -> str:
        previews = []
        for entry in reversed(trajectory):
            tool_name = entry.get("tool_name", "")
            if tool_name in {"read_chunk", "search_and_read", "delegate"}:
                result = str(entry.get("tool_result", "")).strip()
                if result:
                    previews.append(result[:400])
            if len(previews) >= 2:
                break
        return "\n\n".join(reversed(previews))[:1200]

    def _create_child_agent(
        self,
        sub_question: str,
        expected_answer_type: str,
        main_question: str,
        depth: int,
        child_budget: int,
        parent_evidence: str = "",
    ) -> "RLMAgent":
        return RLMAgent(
            llm_client=self.llm,
            chunks_dict=self.chunks_dict,
            embedding_index=self.embedding_index,
            chunks_file=self.chunks_file,
            index_dir=self.index_dir,
            embedding_model=self.embedding_model,
            config=self.config,
            parent_evidence=parent_evidence,
            depth=depth,
            max_depth=self.max_depth,
            parent_question=main_question,
            expected_answer_type=expected_answer_type,
            token_budget=child_budget,
            tool_bundle=self.tool_bundle,
        )

    def _force_finish(
        self,
        messages: List[Dict[str, Any]],
        context: AgentContext,
        trajectory: List[Dict[str, Any]],
        loops: int,
        reason: str,
        total_cost: float,
    ) -> Dict[str, Any]:
        messages.append(
            {
                "role": "user",
                "content": (
                    "You must answer now. Call the finish tool with your best short answer. "
                    f"Reason: {reason}"
                ),
            }
        )
        response = self.llm.chat(messages=messages, tools=self._finish_schemas(), temperature=0.0)
        total_cost += response.get("cost", 0.0)
        self._record_llm_usage(context, "force_finish", response)
        message = self._merge_reasoning(response["message"])
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        for tc in tool_calls:
            if tc["function"]["name"] != "finish":
                continue
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}
            answer = str(args.get("answer", "")).strip()
            context.final_answer = answer
            return {
                "answer": answer,
                "trajectory": trajectory,
                "tokens_used": context.total_llm_tokens,
                "total_cost": total_cost,
                "loops": loops,
                "depth": self.depth,
                "delegations_made": context.delegation_count,
                "evidence_preview": self._build_evidence_preview(trajectory),
                **context.get_summary(),
            }

        answer = _extract_finish_answer(message.get("content", "")) or _strip_thinking(message.get("content", ""))
        context.final_answer = answer
        return {
            "answer": answer,
            "trajectory": trajectory,
            "tokens_used": context.total_llm_tokens,
            "total_cost": total_cost,
            "loops": loops,
            "depth": self.depth,
            "delegations_made": context.delegation_count,
            "evidence_preview": self._build_evidence_preview(trajectory),
            **context.get_summary(),
        }

    def run(self, query: str) -> Dict[str, Any]:
        context = AgentContext()
        context.depth = self.depth
        system_prompt = self._build_system_prompt()
        max_loops = int(self._cfg("agent.max_loops", 10))

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        trajectory: List[Dict[str, Any]] = []
        total_cost = 0.0

        for loop_idx in range(max_loops):
            current_loop = loop_idx + 1
            if self._remaining_budget(context) <= 0:
                return self._force_finish(
                    messages=messages,
                    context=context,
                    trajectory=trajectory,
                    loops=current_loop,
                    reason="token budget exhausted",
                    total_cost=total_cost,
                )

            tool_schemas = self._finish_schemas() if loop_idx == max_loops - 1 else self.tools.get_all_schemas()
            response = self.llm.chat(messages=messages, tools=tool_schemas, tool_choice="required")
            total_cost += response.get("cost", 0.0)
            self._record_llm_usage(context, "agent_loop", response)

            message = self._merge_reasoning(response["message"])
            messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                answer = (
                    _extract_finish_answer(message.get("content", ""))
                    or _strip_thinking(message.get("content", ""))
                )
                context.final_answer = answer
                return {
                    "answer": answer,
                    "trajectory": trajectory,
                    "tokens_used": context.total_llm_tokens,
                    "total_cost": total_cost,
                    "loops": current_loop,
                    "depth": self.depth,
                    "delegations_made": context.delegation_count,
                    "evidence_preview": self._build_evidence_preview(trajectory),
                    **context.get_summary(),
                }

            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                tool_result, tool_log = self.tools.execute(func_name, context, **func_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    }
                )

                traj_entry = {
                    "loop": current_loop,
                    "tool_name": func_name,
                    "arguments": func_args,
                    "tool_result": tool_result,
                    **tool_log,
                }
                trajectory.append(traj_entry)

                if tool_log.get("is_finish"):
                    answer = str(tool_result).strip()
                    context.final_answer = answer
                    return {
                        "answer": answer,
                        "trajectory": trajectory,
                        "tokens_used": context.total_llm_tokens,
                        "total_cost": total_cost,
                        "loops": current_loop,
                        "depth": self.depth,
                        "delegations_made": context.delegation_count,
                        "evidence_preview": self._build_evidence_preview(trajectory),
                        **context.get_summary(),
                    }

            if self._remaining_budget(context) <= 0:
                return self._force_finish(
                    messages=messages,
                    context=context,
                    trajectory=trajectory,
                    loops=current_loop,
                    reason="token budget exceeded after tool use",
                    total_cost=total_cost,
                )

        return self._force_finish(
            messages=messages,
            context=context,
            trajectory=trajectory,
            loops=max_loops,
            reason="max loops reached",
            total_cost=total_cost,
        )

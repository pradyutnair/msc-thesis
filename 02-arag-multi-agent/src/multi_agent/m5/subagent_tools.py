"""LLM-augmented subagent tools for M5 orchestrator."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from arag.core.llm import LLMClient
from arag.tools.base import BaseTool
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.tools.semantic_search import SemanticSearchTool

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", str(text or "")).strip()


def _message_to_text(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _safe_parse_json(text: str) -> Dict[str, Any] | None:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except Exception:
            return None
    return None


class _PromptedSubagentTool(BaseTool):
    """Shared prompt + lightweight generation helpers for subagent tools."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_path: str,
        max_tokens: int,
    ):
        self.llm = llm_client
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")
        self.max_tokens = max_tokens

    def _generate(self, prompt: str) -> str:
        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return _strip_thinking(_message_to_text(response.get("message", {})))


class KeywordAgentTool(_PromptedSubagentTool):
    """Subagent wrapper: infer keywords from a task, then run keyword search."""

    def __init__(
        self,
        raw_tool: KeywordSearchTool,
        llm_client: LLMClient,
        prompt_path: str,
        max_tokens: int = 64,
    ):
        super().__init__(llm_client=llm_client, prompt_path=prompt_path, max_tokens=max_tokens)
        self.raw_tool = raw_tool

    @property
    def name(self) -> str:
        return "keyword_agent"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "keyword_agent",
                "description": (
                    "Delegate keyword retrieval. Provide a natural-language task; "
                    "the tool will infer 2-5 strong search keywords and run keyword search."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Natural language search objective.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of chunks to return (default: 5, max: 20).",
                            "default": 5,
                        },
                    },
                    "required": ["task"],
                },
            },
        }

    def _extract_keywords(self, task: str) -> List[str]:
        prompt = self.prompt_template.format(task=task)
        output = self._generate(prompt)

        parsed = _safe_parse_json(output)
        if parsed and isinstance(parsed.get("keywords"), list):
            kws = [str(k).strip() for k in parsed["keywords"] if str(k).strip()]
            if kws:
                return kws[:5]

        # Fallback: accept comma/newline-separated keywords.
        bits = re.split(r"[,\n;]+", output)
        kws = [b.strip(" -\t\"'") for b in bits if b.strip()]
        if kws:
            return kws[:5]

        # Last fallback: pull key noun-ish tokens from task.
        return [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", task)[:5]] or [task[:32]]

    def execute(self, context, task: str, top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
        keywords = self._extract_keywords(task)
        result, log = self.raw_tool.execute(context, keywords=keywords, top_k=top_k)
        wrapped_result = f"Keyword task: {task}\nDerived keywords: {keywords}\n\n{result}"
        wrapped_log = dict(log)
        wrapped_log.update({"task": task, "derived_keywords": keywords, "subagent": self.name})
        return wrapped_result, wrapped_log


class SemanticAgentTool(_PromptedSubagentTool):
    """Subagent wrapper: infer dense retrieval query from a task."""

    def __init__(
        self,
        raw_tool: SemanticSearchTool,
        llm_client: LLMClient,
        prompt_path: str,
        max_tokens: int = 128,
    ):
        super().__init__(llm_client=llm_client, prompt_path=prompt_path, max_tokens=max_tokens)
        self.raw_tool = raw_tool

    @property
    def name(self) -> str:
        return "semantic_agent"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "semantic_agent",
                "description": (
                    "Delegate semantic retrieval. Provide a natural-language task; "
                    "the tool will craft a dense-retrieval query and run semantic search."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Natural language search objective.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of chunks to return (default: 5, max: 20).",
                            "default": 5,
                        },
                    },
                    "required": ["task"],
                },
            },
        }

    def _formulate_query(self, task: str) -> str:
        prompt = self.prompt_template.format(task=task)
        output = self._generate(prompt)

        parsed = _safe_parse_json(output)
        if parsed and isinstance(parsed.get("query"), str) and parsed["query"].strip():
            return parsed["query"].strip()

        # Fallback: first non-empty line.
        for line in output.splitlines():
            line = line.strip().strip("\"'")
            if line:
                return line
        return task

    def execute(self, context, task: str, top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
        query = self._formulate_query(task)
        result, log = self.raw_tool.execute(context, query=query, top_k=top_k)
        wrapped_result = f"Semantic task: {task}\nFormulated query: {query}\n\n{result}"
        wrapped_log = dict(log)
        wrapped_log.update({"task": task, "formulated_query": query, "subagent": self.name})
        return wrapped_result, wrapped_log


class ChunkReaderAgentTool(_PromptedSubagentTool):
    """Subagent wrapper: read chunks, optionally extract focus-specific evidence."""

    def __init__(
        self,
        raw_tool: ReadChunkTool,
        llm_client: LLMClient,
        prompt_path: str,
        max_tokens: int = 256,
    ):
        super().__init__(llm_client=llm_client, prompt_path=prompt_path, max_tokens=max_tokens)
        self.raw_tool = raw_tool

    @property
    def name(self) -> str:
        return "chunk_reader"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "chunk_reader",
                "description": (
                    "Read full chunk texts by IDs. Optionally provide focus to extract only "
                    "the most relevant facts/sentences for that focus."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Chunk IDs to read.",
                        },
                        "focus": {
                            "type": "string",
                            "description": "Specific information to extract from the chunk text.",
                            "default": "",
                        },
                    },
                    "required": ["chunk_ids"],
                },
            },
        }

    def execute(
        self,
        context,
        chunk_ids: List[str],
        focus: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        raw_result, log = self.raw_tool.execute(context, chunk_ids=chunk_ids)
        focus = (focus or "").strip()
        if not focus:
            wrapped_log = dict(log)
            wrapped_log.update({"focus": "", "subagent": self.name})
            return raw_result, wrapped_log

        # Keep context short for this tactical extraction call.
        prompt = self.prompt_template.format(
            focus=focus,
            text=raw_result[:12000],
        )
        extracted = self._generate(prompt)
        wrapped_result = (
            f"Focus: {focus}\n"
            f"Chunk IDs: {chunk_ids}\n"
            f"\nExtracted evidence:\n{extracted}\n"
            f"\nRaw chunk text (truncated):\n{raw_result[:4000]}"
        )
        wrapped_log = dict(log)
        wrapped_log.update({"focus": focus, "subagent": self.name})
        return wrapped_result, wrapped_log

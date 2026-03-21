"""Shared tool construction logic for RLM agent.

Used by run_escalation.py, collect_escalation_trajectories.py, and runner.py.
"""

from __future__ import annotations

import os
from pathlib import Path

from arag.core.config import Config
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.tools.registry import ToolRegistry
from arag.tools.semantic_search import SemanticSearchTool
from arag.tools.search_and_read import SearchAndReadTool
from arag.tools.recursive_search import RecursiveSearchTool


def build_tool_bundle(config: Config) -> dict[str, object]:
    """Build shared base tools and expose references needed by the RLM agent."""
    data_cfg = config.get("data", {})
    emb_cfg = config.get("embedding", {})

    # FlashRAG backend paths
    sqlite_db = data_cfg.get("sqlite_db") or os.getenv("FLASHRAG_SQLITE_DB")
    faiss_index = data_cfg.get("faiss_index") or os.getenv("FLASHRAG_FAISS_INDEX")

    # Legacy backend paths
    chunks_file = data_cfg.get("chunks_file")
    index_dir = data_cfg.get("index_dir")

    use_flashrag = (
        sqlite_db
        and os.path.exists(sqlite_db)
        and faiss_index
        and os.path.exists(faiss_index)
    )

    if use_flashrag:
        # FlashRAG 21M corpus backend
        keyword_tool = KeywordSearchTool(sqlite_db=sqlite_db)
        read_tool = ReadChunkTool(sqlite_db=sqlite_db)
        semantic_tool = SemanticSearchTool(
            model_name=emb_cfg.get("model", "intfloat/e5-base-v2"),
            device=emb_cfg.get("device"),
            faiss_index_path=faiss_index,
            sqlite_db=sqlite_db,
        )
        search_and_read_tool = SearchAndReadTool(keyword_tool, semantic_tool, read_tool)
    else:
        # Legacy per-dataset corpus
        if not chunks_file:
            chunks_file = "data/chunks.json"

        keyword_tool = KeywordSearchTool(chunks_file=chunks_file)
        read_tool = ReadChunkTool(chunks_file=chunks_file)

        if index_dir and Path(index_dir).exists():
            semantic_tool = SemanticSearchTool(
                chunks_file=chunks_file,
                index_dir=index_dir,
                model_name=emb_cfg.get("model", "intfloat/e5-base-v2"),
                device=emb_cfg.get("device"),
            )
        else:
            semantic_tool = None

        search_and_read_tool = SearchAndReadTool(keyword_tool, semantic_tool, read_tool)

    return {
        "keyword_tool": keyword_tool,
        "read_tool": read_tool,
        "semantic_tool": semantic_tool,
        "search_and_read_tool": search_and_read_tool,
        "chunks_dict": getattr(read_tool, "chunks_dict", None),
        "_recursive_search_deps": (keyword_tool, semantic_tool, read_tool),
    }


def build_tools(config: Config) -> ToolRegistry:
    """Build tool registry from config, auto-detecting FlashRAG vs legacy backend."""
    bundle = build_tool_bundle(config)
    reg = ToolRegistry()
    reg.register(bundle["keyword_tool"])
    reg.register(bundle["read_tool"])
    if bundle["semantic_tool"] is not None:
        reg.register(bundle["semantic_tool"])
    reg.register(bundle["search_and_read_tool"])
    return reg

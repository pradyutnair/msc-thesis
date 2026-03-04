#!/usr/bin/env python3
"""Quick diagnostic: test each subagent tool once to verify fix."""
import json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arag.core.llm import LLMClient
from arag.core.context import AgentContext
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.semantic_search import SemanticSearchTool
from arag.tools.read_chunk import ReadChunkTool
from multi_agent.m5.subagent_tools import KeywordAgentTool, SemanticAgentTool, ChunkReaderAgentTool

DATA_DIR = "/projects/prjs1800/msc-thesis/02-arag-multi-agent/data/hotpotqa"
PROMPT_DIR = "/projects/prjs1800/msc-thesis/02-arag-multi-agent/src/multi_agent/m5/prompts"

def main():
    print("=== M5 Diagnostic Test ===\n")

    # Wait for vLLM to be ready
    client = LLMClient(
        model=os.getenv("ARAG_MODEL", "Qwen3-30B-A3B"),
        api_key=os.getenv("ARAG_API_KEY", "dummy"),
        base_url=os.getenv("ARAG_BASE_URL", "http://127.0.0.1:8000/v1"),
        temperature=0.0,
        max_tokens=512,
        chat_template_kwargs={"enable_thinking": False},
    )

    print("1. Testing raw LLM call (keyword extraction prompt)...")
    prompt = open(f"{PROMPT_DIR}/keyword_extract.txt").read()
    prompt = prompt.replace("{task}", "Find which country Albert Einstein was born in")
    print(f"   Prompt: {prompt[:100]}...")
    try:
        resp = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        msg = resp["message"]
        print(f"   content type: {type(msg.get('content'))}")
        print(f"   content: {repr(msg.get('content'))}")
        print(f"   tool_calls: {msg.get('tool_calls')}")
        print(f"   full message keys: {list(msg.keys())}")
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")

    print("\n2. Testing raw LLM call (query formulation prompt)...")
    prompt = open(f"{PROMPT_DIR}/query_formulate.txt").read()
    prompt = prompt.replace("{task}", "Find which country Albert Einstein was born in")
    try:
        resp = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
        )
        msg = resp["message"]
        print(f"   content: {repr(msg.get('content'))}")
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")

    print("\n3. Testing KeywordAgentTool end-to-end...")
    chunks_file = f"{DATA_DIR}/chunks.json"
    raw_kw = KeywordSearchTool(chunks_file=chunks_file)
    kw_agent = KeywordAgentTool(
        raw_tool=raw_kw,
        llm_client=client,
        prompt_path=f"{PROMPT_DIR}/keyword_extract.txt",
        max_tokens=64,
    )
    ctx = AgentContext()
    try:
        result, log = kw_agent.execute(ctx, task="Find which country Albert Einstein was born in")
        print(f"   SUCCESS! Keywords: {log.get('derived_keywords')}")
        print(f"   Result preview: {result[:200]}...")
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")

    print("\n4. Testing SemanticAgentTool end-to-end...")
    index_dir = f"{DATA_DIR}/index_e5_base_v2"
    print(f"   Loading embedding model...")
    raw_sem = SemanticSearchTool(
        chunks_file=chunks_file,
        index_dir=index_dir,
        model_name="intfloat/e5-base-v2",
    )
    sem_agent = SemanticAgentTool(
        raw_tool=raw_sem,
        llm_client=client,
        prompt_path=f"{PROMPT_DIR}/query_formulate.txt",
        max_tokens=128,
    )
    ctx2 = AgentContext()
    try:
        result, log = sem_agent.execute(ctx2, task="Find which country Albert Einstein was born in")
        print(f"   SUCCESS! Query: {log.get('formulated_query')}")
        print(f"   Result preview: {result[:200]}...")
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")

    print("\n5. Testing ChunkReaderAgentTool (without focus)...")
    raw_reader = ReadChunkTool(chunks_file=chunks_file)
    reader_agent = ChunkReaderAgentTool(
        raw_tool=raw_reader,
        llm_client=client,
        prompt_path=f"{PROMPT_DIR}/extract_evidence.txt",
        max_tokens=256,
    )
    # Get a real chunk ID from keyword search results
    chunk_ids = log.get("metadata", {}).get("chunk_ids", []) if log else []
    if not chunk_ids:
        chunk_ids = ["chunk_0"]  # fallback
    ctx3 = AgentContext()
    try:
        result, rlog = reader_agent.execute(ctx3, chunk_ids=chunk_ids[:1])
        print(f"   SUCCESS! Result length: {len(result)}")
        print(f"   Result preview: {result[:200]}...")
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")

    print("\n=== Diagnostic Complete ===")

if __name__ == "__main__":
    main()

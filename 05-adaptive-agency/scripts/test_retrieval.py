"""Quick test of keyword and semantic search on FlashRAG corpus."""
import os, sys
sys.path.insert(0, "src")
os.environ["FLASHRAG_SQLITE_DB"] = "/projects/prjs1800/datasets/flashrag/wiki18_100w.db"
os.environ["FLASHRAG_FAISS_INDEX"] = "/projects/prjs1800/datasets/flashrag/indexes/e5_Flat.index"

from arag.core.config import Config
from arag.tools.build_tools import build_tools

config = Config.from_yaml("configs/adaptive_agency.yaml")
tools = build_tools(config)

# Create a dummy context
class DummyContext:
    def __init__(self):
        self._read = set()
        self._logs = []
    def add_retrieval_log(self, **kwargs): self._logs.append(kwargs)
    def is_chunk_read(self, cid): return cid in self._read
    def mark_chunk_as_read(self, cid): self._read.add(cid)

ctx = DummyContext()

print("=== Test 1: Keyword search for 'Annie Morton' ===")
kw_tool = tools.get("keyword_search")
result, log = kw_tool.execute(ctx, keywords=["Annie Morton"], top_k=3)
print(result[:500])
print(f"Log: {log}")

print()
print("=== Test 2: Semantic search for 'birth year of Annie Morton' ===")
sem_tool = tools.get("semantic_search")
result, log = sem_tool.execute(ctx, query="birth year of Annie Morton", top_k=3)
print(result[:500])
print(f"Log: {log}")

print()
print("=== Test 3: Keyword search for 'Virginia Commonwealth University founded' ===")
ctx2 = DummyContext()
result, log = kw_tool.execute(ctx2, keywords=["Virginia Commonwealth University"], top_k=3)
print(result[:500])
print(f"Log: {log}")

print()
print("=== Test 4: Read top chunk ===")
read_tool = tools.get("read_chunk")
# Extract first chunk ID from result
import re
ids = re.findall(r"Chunk ID: (\S+)", result)
if ids:
    cid = ids[0].rstrip(",")
    result, log = read_tool.execute(ctx2, chunk_ids=[cid])
    print(result[:500])
    print(f"Log: {log}")
else:
    print("No chunk IDs found in search results")

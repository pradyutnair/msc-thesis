"""Fix structured_worker.py to read chunks via tool for SQLite backend."""
from pathlib import Path

path = Path("src/multi_agent/workers/structured_worker.py")
text = path.read_text()

old = '''                    if cid not in chunk_texts:
                        if self.read_tool is not None and hasattr(self.read_tool, "chunks_dict"):
                            text = str(self.read_tool.chunks_dict.get(cid, ""))
                            if text:
                                chunk_texts[cid] = text'''

new = '''                    if cid not in chunk_texts:
                        # Try in-memory dict first (legacy), then SQLite read
                        if self.read_tool is not None and hasattr(self.read_tool, "chunks_dict"):
                            txt = str(self.read_tool.chunks_dict.get(cid, ""))
                            if txt:
                                chunk_texts[cid] = txt
                        elif self.read_tool is not None:
                            # SQLite/FlashRAG backend: read via tool
                            read_ctx = AgentContext()
                            read_result, _ = self.read_tool.execute(read_ctx, chunk_ids=[cid])
                            # Extract text from read result (skip header/separator lines)
                            txt_lines = []
                            in_content = False
                            for line in read_result.split("\\n"):
                                if line.startswith("-" * 10):
                                    in_content = True
                                    continue
                                if line.startswith("=" * 10):
                                    in_content = False
                                    continue
                                if in_content:
                                    txt_lines.append(line)
                            txt = "\\n".join(txt_lines).strip()
                            if txt:
                                chunk_texts[cid] = txt'''

if old in text:
    text = text.replace(old, new)
    path.write_text(text)
    print("Fixed: structured_worker now reads chunks via tool for SQLite backend")
else:
    print("ERROR: Could not find the exact block to replace")
    print("Looking for:", repr(old[:80]))

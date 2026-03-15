"""Fix M6 worker_agent.py and memory.py for FlashRAG compatibility.

Fixes:
1. Memory.get_read_chunks() - also capture search_and_read results
2. Memory.get_all_evidence() - new method capturing all tool results with text
3. Worker._build_evidence() - use all evidence sources, not just read_chunk
4. Worker._clean_answer() - filter separator lines, raw tool calls, verbose text
5. Worker.verify logic - don't verify garbage answers
"""

from pathlib import Path


def fix_memory(path: Path) -> None:
    text = path.read_text()

    # Add get_all_evidence method that captures search_and_read and read_chunk results
    old_to_dicts = '''    def to_dicts(self) -> list[dict[str, Any]]:'''

    new_method_plus_to_dicts = '''    def get_all_evidence(self) -> list[tuple[str, str]]:
        """Return (chunk_id, content) from ALL tool calls that retrieved text.

        Captures: read_chunk, search_and_read, keyword_search, semantic_search.
        """
        import re
        chunks: list[tuple[str, str]] = []
        seen_cids: set[str] = set()

        for a in self.actions:
            if not a.result:
                continue

            # Extract chunk IDs from any tool result
            chunk_ids = re.findall(r"Chunk ID: (\\S+)", a.result)
            chunk_ids = [cid.rstrip(",.;:") for cid in chunk_ids]

            if a.tool_name == "read_chunk":
                # Parse the formatted read_chunk output
                current_cid = None
                content_lines = []
                in_content = False
                for line in a.result.split("\\n"):
                    if line.startswith("[Chunk "):
                        if current_cid and content_lines:
                            content = "\\n".join(content_lines).strip()
                            if content and current_cid not in seen_cids:
                                chunks.append((current_cid, content))
                                seen_cids.add(current_cid)
                        cid_match = re.search(r"\\[Chunk (\\S+?)\\]", line)
                        current_cid = cid_match.group(1) if cid_match else None
                        content_lines = []
                        in_content = False
                    elif line.startswith("-" * 10):
                        in_content = True
                    elif line.startswith("=" * 10):
                        in_content = False
                    elif in_content:
                        content_lines.append(line)
                # Last chunk
                if current_cid and content_lines:
                    content = "\\n".join(content_lines).strip()
                    if content and current_cid not in seen_cids:
                        chunks.append((current_cid, content))
                        seen_cids.add(current_cid)

            elif a.tool_name == "search_and_read":
                # search_and_read returns read_chunk formatted output
                current_cid = None
                content_lines = []
                in_content = False
                for line in a.result.split("\\n"):
                    if line.startswith("[Chunk "):
                        if current_cid and content_lines:
                            content = "\\n".join(content_lines).strip()
                            if content and current_cid not in seen_cids:
                                chunks.append((current_cid, content))
                                seen_cids.add(current_cid)
                        cid_match = re.search(r"\\[Chunk (\\S+?)\\]", line)
                        current_cid = cid_match.group(1) if cid_match else None
                        content_lines = []
                        in_content = False
                    elif line.startswith("-" * 10):
                        in_content = True
                    elif line.startswith("=" * 10):
                        in_content = False
                    elif in_content:
                        content_lines.append(line)
                if current_cid and content_lines:
                    content = "\\n".join(content_lines).strip()
                    if content and current_cid not in seen_cids:
                        chunks.append((current_cid, content))
                        seen_cids.add(current_cid)

            elif a.tool_name in ("keyword_search", "semantic_search"):
                # These return snippets with chunk IDs — store as lightweight evidence
                for cid in chunk_ids:
                    if cid not in seen_cids:
                        chunks.append((cid, a.result[:1000]))
                        seen_cids.add(cid)

        return chunks

    def to_dicts(self) -> list[dict[str, Any]]:'''

    if old_to_dicts in text:
        text = text.replace(old_to_dicts, new_method_plus_to_dicts)
        path.write_text(text)
        print("memory.py: added get_all_evidence()")
    else:
        print("memory.py: WARNING - could not find to_dicts insertion point")


def fix_worker(path: Path) -> None:
    text = path.read_text()

    # Fix 1: _build_evidence to use get_all_evidence instead of get_read_chunks
    old_build = '''    def _build_evidence(
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
        return entries'''

    new_build = '''    def _build_evidence(
        self, sq_id: int, memory: Memory,
    ) -> list[EvidenceEntry]:
        entries: list[EvidenceEntry] = []
        for cid, content in memory.get_all_evidence():
            entries.append(EvidenceEntry(
                id="",
                sub_question_id=sq_id,
                content=content[:2000],
                source_chunk_id=cid,
                relevance_score=0.5,
                retriever_id=self.agent_id,
            ))
        return entries'''

    if old_build in text:
        text = text.replace(old_build, new_build)
        print("worker_agent.py: fixed _build_evidence -> get_all_evidence()")
    else:
        print("worker_agent.py: WARNING - could not find _build_evidence")

    # Fix 2: improve _clean_answer to filter garbage
    old_clean = '''    def _clean_answer(self, answer: str) -> str:
        answer = re.sub(r"<think>.*?</think>\\s*", "", answer, flags=re.DOTALL)
        answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL)
        answer = re.sub(r"\\*\\*(.+?)\\*\\*", r"\\1", answer)
        answer = re.sub(r"\\*(.+?)\\*", r"\\1", answer)
        answer = answer.split("\\n")[0].strip()
        answer = answer.strip().strip("\\"\'`*")
        answer = re.sub(r"\\s*[\\.,;:!?]+$", "", answer)
        return answer'''

    new_clean = '''    def _clean_answer(self, answer: str) -> str:
        answer = re.sub(r"<think>.*?</think>\\s*", "", answer, flags=re.DOTALL)
        answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL)
        answer = re.sub(r"\\*\\*(.+?)\\*\\*", r"\\1", answer)
        answer = re.sub(r"\\*(.+?)\\*", r"\\1", answer)
        answer = answer.split("\\n")[0].strip()
        answer = answer.strip().strip("\\"\'`*")
        answer = re.sub(r"\\s*[\\.,;:!?]+$", "", answer)
        # Filter garbage: separator lines, raw tool calls, chunk markers
        if re.match(r"^[=\\-_]{5,}$", answer):
            return ""
        if answer.startswith(("keyword_search(", "semantic_search(", "read_chunk(", "search_and_read(")):
            return ""
        if answer.startswith("[Chunk ") or answer.startswith("Chunk ID:"):
            return ""
        if answer.startswith("Title:") and len(answer) > 200:
            return ""
        return answer'''

    if old_clean in text:
        text = text.replace(old_clean, new_clean)
        print("worker_agent.py: fixed _clean_answer to filter garbage")
    else:
        print("worker_agent.py: WARNING - could not find _clean_answer")

    # Fix 3: Don't verify answers that are clearly garbage
    # The verify check is: is_usable = bool(answer) and answer.lower() not in (...)
    # We need to also reject very short nonsense and separator-like content
    old_verify = '''        is_usable = bool(answer) and answer.lower() not in ("unknown", "error", "")'''

    new_verify = '''        is_usable = (
            bool(answer)
            and answer.lower() not in ("unknown", "error", "", "none", "n/a")
            and len(answer) > 1
            and not re.match(r"^[=\\-_]{3,}$", answer)
        )'''

    if old_verify in text:
        text = text.replace(old_verify, new_verify)
        print("worker_agent.py: fixed verify check to reject garbage")
    else:
        print("worker_agent.py: WARNING - could not find verify check")

    path.write_text(text)


def main():
    base = Path(".")
    fix_memory(base / "src" / "multi_agent" / "m6" / "memory.py")
    fix_worker(base / "src" / "multi_agent" / "m6" / "worker_agent.py")

    # Verify syntax
    import ast
    for f in ["src/multi_agent/m6/memory.py", "src/multi_agent/m6/worker_agent.py"]:
        ast.parse(Path(f).read_text())
        print(f"{f}: syntax OK")


if __name__ == "__main__":
    main()

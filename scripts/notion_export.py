import os
import re
import yaml
from pathlib import Path
from slugify import slugify
from notion_client import Client

OUT_DIR = Path("papers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["NOTION_DATABASE_ID"]

notion = Client(auth=NOTION_TOKEN)

def extract_page_id_from_url(url: str) -> str | None:
    # Works for URLs like https://www.notion.so/<32hex> or .../<title>-<32hex>
    m = re.search(r"([0-9a-fA-F]{32})", url or "")
    return m.group(1).lower() if m else None

def rt_to_text(rich_text) -> str:
    parts = []
    for r in rich_text or []:
        parts.append(r.get("plain_text", ""))
    return "".join(parts).strip()

def get_prop(props, name):
    return props.get(name)

def prop_to_value(prop):
    if not prop:
        return None
    t = prop["type"]
    if t == "title":
        return rt_to_text(prop["title"])
    if t == "rich_text":
        return rt_to_text(prop["rich_text"])
    if t == "select":
        s = prop.get("select")
        return s["name"] if s else None
    if t == "multi_select":
        return [x["name"] for x in prop.get("multi_select", [])]
    if t == "date":
        d = prop.get("date")
        return d.get("start") if d else None
    if t == "url":
        return prop.get("url")
    # Notion recently added "link_to_page" as a property type in some workspaces.
    if t == "link_to_page":
        ltp = prop.get("link_to_page")
        if not ltp:
            return None
        # ltp can be {"type":"page_id","page_id":...} or {"type":"database_id",...}
        if ltp.get("type") == "page_id":
            return ltp.get("page_id")
        return None
    return None

def fetch_all_blocks(block_id: str):
    blocks = []
    cursor = None
    while True:
        resp = notion.blocks.children.list(block_id=block_id, start_cursor=cursor)
        blocks.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")
    return blocks

def block_to_md(block, indent=0):
    btype = block["type"]
    obj = block.get(btype, {})
    prefix = "  " * indent

    def text():
        return rt_to_text(obj.get("rich_text", []))

    if btype == "paragraph":
        return f"{prefix}{text()}\n" if text() else "\n"

    if btype in ("heading_1", "heading_2", "heading_3"):
        level = {"heading_1": "#", "heading_2": "##", "heading_3": "###"}[btype]
        return f"{prefix}{level} {text()}\n\n"

    if btype == "bulleted_list_item":
        s = f"{prefix}- {text()}\n"
        if block.get("has_children"):
            children = fetch_all_blocks(block["id"])
            for c in children:
                s += block_to_md(c, indent=indent + 1)
        return s

    if btype == "numbered_list_item":
        s = f"{prefix}1. {text()}\n"
        if block.get("has_children"):
            children = fetch_all_blocks(block["id"])
            for c in children:
                s += block_to_md(c, indent=indent + 1)
        return s

    if btype == "to_do":
        checked = obj.get("checked", False)
        box = "x" if checked else " "
        s = f"{prefix}- [{box}] {text()}\n"
        if block.get("has_children"):
            children = fetch_all_blocks(block["id"])
            for c in children:
                s += block_to_md(c, indent=indent + 1)
        return s

    if btype == "quote":
        return f"{prefix}> {text()}\n\n"

    if btype == "code":
        lang = obj.get("language", "")
        code_text = rt_to_text(obj.get("rich_text", []))
        return f"{prefix}```{lang}\n{code_text}\n```\n\n"

    if btype == "divider":
        return f"{prefix}---\n\n"

    if btype == "callout":
        # simple rendering
        return f"{prefix}> {text()}\n\n"

    # Unsupported block types are skipped quietly
    return ""

def page_blocks_to_md(page_id: str) -> str:
    md = ""
    for b in fetch_all_blocks(page_id):
        md += block_to_md(b)
    return md.strip() + "\n"

def sanitize_filename(s: str) -> str:
    s = s.strip()
    s = slugify(s) if s else "untitled"
    return s[:120]  # keep it reasonable for filesystems

def query_all_rows(database_id: str):
    rows = []
    cursor = None
    while True:
        resp = notion.databases.query(database_id=database_id, start_cursor=cursor)
        rows.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")
    return rows

def main():
    rows = query_all_rows(DATABASE_ID)

    for page in rows:
        props = page.get("properties", {})

        title = prop_to_value(get_prop(props, "Title")) or "(No title)"
        short_title = prop_to_value(get_prop(props, "Short Title"))
        citation_name = prop_to_value(get_prop(props, "Name"))
        tags = prop_to_value(get_prop(props, "Personal Tags")) or []
        date = prop_to_value(get_prop(props, "Date"))

        notes_link_prop = get_prop(props, "Personal Notes")
        notes_link_val = prop_to_value(notes_link_prop)

        # If it is a URL, convert to page_id
        notes_page_id = None
        if isinstance(notes_link_val, str) and notes_link_val.startswith("http"):
            notes_page_id = extract_page_id_from_url(notes_link_val)
        elif isinstance(notes_link_val, str) and re.fullmatch(r"[0-9a-f]{32}", notes_link_val):
            notes_page_id = notes_link_val

        fm = {
            "notion_page_id": page["id"],
            "title": title,
            "short_title": short_title,
            "citation_name": citation_name,
            "tags": tags,
            "date": date,
        }
        # remove nulls
        fm = {k: v for k, v in fm.items() if v not in (None, "", [])}

        filename_base = short_title or title
        path = OUT_DIR / f"{sanitize_filename(filename_base)}.md"

        row_body = page_blocks_to_md(page["id"])
        linked_body = ""
        if notes_page_id:
            try:
                linked_body = page_blocks_to_md(notes_page_id)
            except Exception:
                linked_body = ""

        parts = []
        parts.append("---\n" + yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip() + "\n---\n")

        parts.append("## Notes (database row)\n\n")
        parts.append((row_body.strip() + "\n") if row_body.strip() else "_(empty)_\n")

        if notes_page_id:
            parts.append("\n## Personal Notes (linked page)\n\n")
            parts.append((linked_body.strip() + "\n") if linked_body.strip() else "_(empty or not accessible)_\n")

        path.write_text("".join(parts), encoding="utf-8")

if __name__ == "__main__":
    main()

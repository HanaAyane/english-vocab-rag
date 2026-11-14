#!/usr/bin/env python3
"""
Parse the Obsidian English-vocabulary vault into entry-level chunks.

Vault format (measured, not assumed):
  - Notes carry NO markdown headings / frontmatter. Entries are marked with
    HTML spans:
        <span class="category">**sub-theme:**</span>
        <span class="vocabulary">**word**</span> [phonetic]
        <span class="definition">pos. Chinese definition:</span> definition + examples
  - Navigation files duplicate the main files and are skipped by their
    naming convention or by the absence of a definition in each entry.
  - Files without any vocabulary span (resource lists, acknowledgements,
    catalogues) are skipped.

Chunking rule: one vocabulary entry = one chunk. Fields extracted:
  word, phonetic, body (definitions + examples, in original order),
  sub_category, plus metadata from the folder path (top category / group /
  theme file) which doubles as citation info at answer time.
"""
import json
import os
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

VAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vault")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chunks.jsonl")

SPAN_RE = re.compile(r'<span class="(vocabulary|definition|category)">(.*?)</span>', re.S)
TAG_RE = re.compile(r"<[^>]+>")
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
CHECKBOX_RE = re.compile(r"^-?\s*\[.\]\s*", re.M)
PHONETIC_RE = re.compile(r"^\s*\[([^\]]{1,60})\]")
POS_RE = re.compile(r"(?<![a-z])(?:v[ti]?|adj|adv|n|prep|pron|conj|interj|det|num|art|aux)\.", re.I)
INLINE_POS_RE = re.compile(r"^(?:\[[^\]\n]+\]\s*)?(?:\d+\s*)?" + POS_RE.pattern, re.I)
POS_LABELS = {"v": "动词", "n": "名词", "adj": "形容词", "adv": "副词",
              "prep": "介词", "pron": "代词", "conj": "连词", "interj": "感叹词",
              "det": "限定词", "num": "数词", "art": "冠词", "aux": "助动词"}


def clean(text: str) -> str:
    text = WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), text)
    text = TAG_RE.sub("", text)
    text = text.replace("**", "").replace("☀", " ")
    text = CHECKBOX_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def is_content_file(path: str, name: str) -> bool:
    folders = path.replace("\\", "/").split("/")[:-1]
    return not any(part.startswith("00总目录") for part in folders) and not (
        name.startswith("00") and "组目录" in name
    )


def definition_parts(definition: str):
    """Read a POS prefix, then leave the grammar and gloss intact."""
    prefix = re.match(r"^(?:\d+[.)、]?\s*)?((?:[a-z]+\.\s*[&＆/、,，]?\s*)+)", definition, re.I)
    if not prefix:
        return [], definition
    codes = [tag.lower().rstrip(".") for tag in POS_RE.findall(prefix.group(1))]
    return [POS_LABELS["v" if code in {"vt", "vi"} else code] for code in codes], definition


def embedding_text(word, definitions, parts_of_speech, theme):
    # Keep a compact gloss from each definition, retaining the original wording.
    glosses = []
    for definition in definitions:
        chinese = re.search(r"[\u3400-\u9fff]", definition)
        core = definition[chinese.start():] if chinese else definition
        glosses.append(re.split(r"[。；;：:]", core, maxsplit=1)[0].strip()[:70])
    pos = " ".join(parts_of_speech)
    gloss = "；".join(dict.fromkeys(glosses))
    return f"{word} {pos} {gloss[:210]} 主题：{theme}".strip()


def parse_file(path: str, rel: str):
    raw = Path(path).read_text(encoding="utf-8-sig")
    if 'class="vocabulary"' not in raw:
        return []
    rel = rel.replace("\\", "/")
    parts = rel.split("/")
    theme = re.sub(r"^\d+\s*", "", os.path.splitext(parts[-1])[0])
    group = re.sub(r"^\d+\s*", "", parts[-2]) if len(parts) >= 2 else ""
    top_category = re.sub(r"^\d+\s*", "", parts[0]) if parts else ""

    entries = []
    current_category = ""
    category_number = 0
    current = None
    pos = 0

    def flush():
        nonlocal current
        if current is not None:
            body = clean(" ".join(current.pop("segments")))
            definitions = current["definitions"]
            if not definitions and INLINE_POS_RE.match(body):
                phonetic = PHONETIC_RE.match(body)
                inline = body[phonetic.end():].strip() if phonetic else body
                definitions.append(re.split(r"[：:\n]", inline, maxsplit=1)[0])
            if current["word"] and definitions:
                current["body"] = body
                entries.append(current)
            current = None

    for m in SPAN_RE.finditer(raw):
        between = raw[pos:m.start()]
        kind, inner = m.group(1), m.group(2)
        if kind == "category":
            if current is not None:
                current["segments"].append(between)
            flush()
            current_category = clean(inner).rstrip("：:").strip()
            category_number += 1
        elif kind == "vocabulary":
            if current is not None:
                current["segments"].append(between)
            flush()
            current = {
                "word": clean(inner),
                "phonetic": "",
                "segments": [],
                "definitions": [],
                "sub_category": current_category,
                "section": category_number,
                "top_category": top_category,
                "group": group,
                "theme": theme,
                "source": rel,
            }
        elif kind == "definition":
            if current is not None:
                current["segments"].append(between)
                definition = clean(inner)
                if definition:
                    current["definitions"].append(definition)
                    current["segments"].append(definition)
        pos = m.end()
    if current is not None:
        current["segments"].append(raw[pos:])
    flush()

    for e in entries:
        m = PHONETIC_RE.match(e["body"])
        if m:
            e["phonetic"] = m.group(1).strip()
        e["parts_of_speech"] = sorted({tag for definition in e["definitions"]
                                       for tag in definition_parts(definition)[0]})
        e["embed_text"] = embedding_text(e["word"], e["definitions"], e["parts_of_speech"], e["theme"])
    return entries


def main():
    all_entries = []
    n_files = 0
    for dp, dns, fns in os.walk(VAULT):
        dns[:] = sorted(d for d in dns if d != ".obsidian")
        for fn in sorted(fns):
            if not fn.endswith(".md"):
                continue
            path = os.path.join(dp, fn)
            rel = os.path.relpath(path, VAULT)
            if not is_content_file(rel, fn):
                continue
            es = parse_file(path, rel)
            if es:
                n_files += 1
                all_entries.extend(es)
    with NamedTemporaryFile(mode="w", dir=Path(OUT).parent, encoding="utf-8",
                            newline="\n", delete=False) as f:
        tmp = Path(f.name)
        try:
            for i, e in enumerate(all_entries):
                e["id"] = i
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
            f.close()
            tmp.replace(OUT)
        finally:
            tmp.unlink(missing_ok=True)
    words = {e["word"].lower() for e in all_entries}
    print(f"content_files={n_files}")
    print(f"chunks={len(all_entries)} unique_words={len(words)}")
    print(f"out={OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
    if "--show" in sys.argv:
        for e in all_entries[:3]:
            print(json.dumps(e, ensure_ascii=False, indent=1)[:800], "\n---")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Hybrid lookup, reverse lookup and comparison for the vocabulary vault."""
import gzip
import json
import pickle
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

from index_store import DATA_DIR, DIMENSIONS, MODEL, read_snapshot, tokenize
from generation import answer, llm_config
from parse_vault import POS_LABELS

WORD_RE = re.compile(r"[a-z]+(?:[-'][a-z]+)*", re.I)
LOOKUP_RE = re.compile(
    r"^[\"'「『“]?(?P<term>[a-z][a-z0-9\-']*(?:\s+[a-z][a-z0-9\-']*)*)"
    r"[\"'」』”]?\s*(?:是?什么(?:意思|含义)|的(?:意思|含义|用法)|怎么(?:翻译|用|读)|如何使用)\s*[?？。！!]*$",
    re.I,
)
ENGLISH_LOOKUP_RE = re.compile(r"^(?:what does (.+?) mean|define (.+?))[?.!]*$", re.I)
COMPARE_RE = re.compile(r"区别|差别|差异|辨析|对比|相比|\b(?:vs\.?|versus|difference between|compare)\b", re.I)
EXPAND_RE = re.compile(r"同组|这一组|同类|还有哪些")
QUERY_POS_RE = re.compile("|".join(sorted(POS_LABELS.values(), key=len, reverse=True)))
STOP_WORDS = set("a an the to of in for and or as is are be does do how what which when between with mean define compare vs versus difference".split())


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).lower().replace("’", "'").split())


def entry_group(row):
    return row["source"], row.get("section", row["sub_category"])


class Retriever:
    def __init__(self, data_dir=DATA_DIR):
        data_dir = Path(data_dir)
        if (data_dir / "index.zip").is_file():
            self.rows, self.emb, corpus = read_snapshot(data_dir)
        else:
            # Read the original index layout for existing installations.
            with (data_dir / "chunks.jsonl").open(encoding="utf-8") as stream:
                self.rows = [json.loads(line) for line in stream]
            if any("definitions" in row for row in self.rows):
                raise ValueError("已生成新版词条，请运行 build_index.py 发布配套索引。")
            self.emb = np.load(data_dir / "embeddings.npy", allow_pickle=False)
            with gzip.open(data_dir / "bm25_corpus.pkl.gz", "rb") as stream:
                corpus = pickle.load(stream)
        count = len(self.rows)
        if not count or len(corpus) != count or self.emb.shape != (count, DIMENSIONS):
            raise ValueError("词条、向量和 BM25 索引数量不一致，请运行 build_index.py。")

        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi(corpus)
        self.by_word = defaultdict(list)
        self.by_group = defaultdict(list)
        self.by_pos = defaultdict(list)
        self.max_phrase = 1
        for i, row in enumerate(self.rows):
            word = normalize(row["word"])
            self.by_word[word].append(i)
            self.by_group[entry_group(row)].append(i)
            self.max_phrase = max(self.max_phrase, len(word.split()))
            for pos in row.get("parts_of_speech", []):
                self.by_pos[pos].append(i)
        self._model = None

    def _embed_query(self, query):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=MODEL, threads=4)
        vector = np.asarray(next(iter(self._model.embed([query]))), dtype=np.float32)
        return vector / max(np.linalg.norm(vector), 1e-9)

    def _headwords(self, query):
        bare = query.strip("\"'“”「」『』 ?？。!！")
        if bare in self.by_word:
            return [bare], True
        lookup = LOOKUP_RE.fullmatch(query) or ENGLISH_LOOKUP_RE.fullmatch(query)
        if lookup:
            return [next(term for term in lookup.groups() if term)], True
        words = []
        tokens = list(WORD_RE.finditer(query))
        i = 0
        while i < len(tokens):
            # Prefer a complete phrase before considering its component words.
            for end in range(min(len(tokens), i + self.max_phrase), i, -1):
                term = query[tokens[i].start():tokens[end - 1].end()]
                if term in self.by_word and term not in STOP_WORDS:
                    words.append(term)
                    i = end
                    break
            else:
                i += 1
        return list(dict.fromkeys(words)), False

    def _scores(self, query, mode):
        bm25 = self.bm25.get_scores(tokenize(query)) if mode != "vector" else None
        if bm25 is not None:
            bm25 = bm25 / max(float(bm25.max()), 1e-9)
        if mode == "bm25":
            return bm25
        similarities = self.emb @ self._embed_query(query)
        return similarities if mode == "vector" else 0.65 * similarities + 0.35 * bm25

    def _compare(self, ids_by_word, scores, expand):
        groups_by_word = []
        for ids in ids_by_word:
            groups = {}
            for i in ids:
                group = entry_group(self.rows[i])
                if group not in groups or scores[i] > scores[groups[group]]:
                    groups[group] = i
            groups_by_word.append(groups)
        shared = set.intersection(*(set(groups) for groups in groups_by_word))
        if shared:
            group = max(sorted(shared), key=lambda g: sum(scores[groups[g]] for groups in groups_by_word))
            selected = [groups[group] for groups in groups_by_word]
            if expand:
                others = sorted(self.by_group[group], key=lambda i: (-scores[i], i))
                selected.extend(i for i in others if i not in selected)
            return selected
        return [max(ids, key=lambda i: scores[i]) for ids in ids_by_word]

    def search(self, query, top_k=8, *, mode="hybrid"):
        query = normalize(query)
        if not query or top_k <= 0:
            return []
        words, strict = self._headwords(query)
        if strict and words[0] not in self.by_word:
            return []
        ids_by_word = [self.by_word[word] for word in words]
        requested_pos = QUERY_POS_RE.findall(query)
        pos_ids = {i for pos in requested_pos for i in self.by_pos.get(pos, [])} if requested_pos and self.by_pos else None
        if pos_ids is not None:
            ids_by_word = [[i for i in ids if i in pos_ids] for ids in ids_by_word]
        if words and not any(ids_by_word):
            return []

        if strict and len(ids_by_word[0]) == 1:
            return [dict(self.rows[ids_by_word[0][0]], score=1.0)]

        scores = self._scores(query, mode)
        expand = bool(ids_by_word and EXPAND_RE.search(query))
        if all(ids_by_word) and (expand or (len(ids_by_word) > 1 and COMPARE_RE.search(query))):
            ordered = self._compare(ids_by_word, scores, expand)
            if pos_ids is not None:
                ordered = [i for i in ordered if i in pos_ids]
        else:
            candidates = list(dict.fromkeys(i for ids in ids_by_word for i in ids))
            if not candidates:
                candidates = pos_ids if pos_ids is not None else range(len(self.rows))
            ordered = sorted(candidates, key=lambda i: (-scores[i], i))
            if not words:
                unique = {}
                for i in ordered:
                    unique.setdefault(normalize(self.rows[i]["word"]), i)
                    if len(unique) == top_k:
                        break
                ordered = list(unique.values())
        return [dict(self.rows[i], score=round(float(scores[i]), 4)) for i in ordered[:top_k]]


def format_sources(results):
    return [
        f"[{j}] {row['word']} — {row['source']} / {row['sub_category']}".rstrip(" /")
        for j, row in enumerate(results, 1)
    ]

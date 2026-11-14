"""Shared retrieval settings and the atomically published index snapshot."""
import hashlib
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIMENSIONS = 384
INDEX_VERSION = 1

# Older httpx releases reject brackets in NO_PROXY's IPv6 entries.
for key in ("NO_PROXY", "no_proxy"):
    if key in os.environ:
        os.environ[key] = ",".join(part.strip().strip("[]")
                                   for part in os.environ[key].split(",") if part.strip())


def tokenize(text):
    import jieba

    tokens = (token.strip() for token in jieba.cut(text.lower()))
    return [token for token in tokens if re.search(r"[^\s\W_]", token)]


def fingerprint(content):
    return hashlib.sha256(content).hexdigest()


@contextmanager
def atomic_output(target):
    target = Path(target)
    with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def read_snapshot(data_dir):
    with ZipFile(data_dir / "index.zip") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["version"] != INDEX_VERSION or manifest["model"] != MODEL:
            raise ValueError("索引版本或模型与当前程序不一致，请重建索引。")
        chunks = archive.read("chunks.jsonl")
        if fingerprint(chunks) != manifest["chunks_sha256"]:
            raise ValueError("索引内的词条指纹不匹配，请重建索引。")
        rows = [json.loads(line) for line in chunks.splitlines()]
        with archive.open("embeddings.npy") as stream:
            embeddings = np.load(stream, allow_pickle=False)
        corpus = json.loads(archive.read("bm25.json"))
        if len(rows) != manifest["count"] or embeddings.shape != (len(rows), DIMENSIONS):
            raise ValueError("索引内的词条和向量数量不一致，请重建索引。")
        return rows, embeddings, corpus

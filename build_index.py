#!/usr/bin/env python3
"""Build and atomically publish the MiniLM and BM25 indexes."""
import json
import os
import time
from importlib.metadata import version
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from index_store import DATA_DIR, DIMENSIONS, INDEX_VERSION, MODEL, atomic_output, fingerprint, tokenize

BATCH_SIZE = 64
CHECKPOINT_EVERY = 512


def build_vectors(rows, checkpoint, build_id):
    from fastembed import TextEmbedding

    threads = min(8, os.cpu_count() or 1)
    model = TextEmbedding(model_name=MODEL, threads=threads)
    texts = [row["embed_text"] for row in rows]
    # Group roughly similar lengths for ONNX batching. Store results in row order.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vectors = np.zeros((len(rows), DIMENSIONS), dtype=np.float32)
    done = 0
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as previous:
            if previous["build_id"].item() == build_id and previous["vectors"].shape == vectors.shape:
                done = int(previous["done"])
                vectors = previous["vectors"]
                print(f"续建 {done}/{len(rows)}", flush=True)

    resumed_at = done
    started = time.monotonic()
    while done < len(rows):
        end = min(done + CHECKPOINT_EVERY, len(rows))
        indices = order[done:end]
        batch = [texts[i] for i in indices]
        vectors[indices] = np.asarray(list(model.embed(batch, batch_size=BATCH_SIZE)), dtype=np.float32)
        done = end
        with atomic_output(checkpoint) as temporary, temporary.open("wb") as stream:
            np.savez(stream, build_id=build_id, done=done, vectors=vectors)
        rate = (done - resumed_at) / max(time.monotonic() - started, 0.01)
        print(f"向量 {done}/{len(rows)} | {rate:.0f} 条/秒 | 剩余约 {(len(rows) - done) / rate:.0f} 秒", flush=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("向量索引包含无效向量。")
    return vectors / norms


def main():
    chunks_path = DATA_DIR / "chunks.jsonl"
    chunks = chunks_path.read_bytes()
    rows = [json.loads(line) for line in chunks.splitlines()]
    if not rows or any("definitions" not in row for row in rows):
        raise ValueError("请先用当前版本的 parse_vault.py 生成词条。")
    manifest = {
        "version": INDEX_VERSION,
        "model": MODEL,
        "fastembed": version("fastembed"),
        "count": len(rows),
        "chunks_sha256": fingerprint(chunks),
    }
    build_id = fingerprint(json.dumps(manifest, sort_keys=True).encode())
    checkpoint = DATA_DIR / f"embeddings-{build_id[:16]}.npz"
    vectors = build_vectors(rows, checkpoint, build_id)

    corpus = []
    for row in rows:
        word = row["word"]
        text = f"{word} {word} {word} {row['theme']} {row['sub_category']} {row['body']}"
        corpus.append(tokenize(text))
    with atomic_output(DATA_DIR / "index.zip") as temporary:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("chunks.jsonl", chunks)
            archive.writestr("bm25.json", json.dumps(corpus, ensure_ascii=False))
            with archive.open("embeddings.npy", "w") as stream:
                np.save(stream, vectors, allow_pickle=False)
        if fingerprint(chunks_path.read_bytes()) != manifest["chunks_sha256"]:
            raise RuntimeError("建索引期间词条已更新，请重新运行 build_index.py。")
    checkpoint.unlink(missing_ok=True)
    print(f"已发布 data/index.zip，共 {len(rows)} 条。", flush=True)


if __name__ == "__main__":
    main()

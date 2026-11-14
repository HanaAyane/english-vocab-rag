#!/usr/bin/env python3
"""Quick CLI check of the retrieval layer. Usage: python search_cli.py "问题" [top_k]"""
import sys

from rag import Retriever, format_sources

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "表示突然转弯的动词有哪些"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rt = Retriever()
    results = rt.search(q, top_k=k)
    for line, r in zip(format_sources(results), results):
        print(f"{line}  (score={r['score']})")
        print(f"    {r['body'][:120]}...")

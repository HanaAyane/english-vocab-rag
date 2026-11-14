"""Manually labelled checks for headwords, parts of speech and source senses."""
import argparse
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from rag import Retriever

# Each target is a headword, a source theme substring and a POS expression.
# Alternatives in a group count as one target. Every group must match.
CASES = [
    ("查词", "swerve 是什么意思", 1, [[("swerve", "拐弯", "v."), ("swerve", "转弯", "v.")]]),
    ("查词", "black market 是什么意思", 1, [[("black market", "走私", "n.")]]),
    ("查词", "BALANCE SHEET 是什么意思", 1, [[("balance sheet", "报表", "n.")]]),
    ("查词", "black hole 在天文学中指什么", 1, [[("black hole", "星体", "n.")]]),
    ("查词", "跑步、经营公司和竞选里的 run 分别怎么用", 5, [
        [("run", "田径项目", "v."), ("run", "走跑跳踢", "v.")],
        [("run", "经营", "v.")], [("run", "竞选", "v.")],
    ]),
    ("查词", "revise 在英式英语中为考试复习是什么意思", 3, [[("revise", "复习 作业", "v.")]]),
    ("查词", "checklist 是什么意思", 1, [[("checklist", "清单 目录", "n.")]]),
    ("查词", "preface 指一本书的前言时怎么用", 3, [[("preface", "序言 目录", "n.")]]),
    ("反查", "表示突然转弯的动词有哪些", 3, [
        [("swerve", "拐弯", "v."), ("veer", "转弯", "v.")],
    ]),
    ("反查", "车为避免撞到人急转方向，用哪个动词", 5, [
        [("swerve", "转弯", "v."), ("swerve", "拐弯", "v."), ("veer", "转弯", "v.")],
    ]),
    ("反查", "表示恐惧、害怕的形容词", 4, [
        [("afraid", "恐惧", "adj.")], [("frightened", "恐惧", "adj."), ("terrified", "恐惧", "adj."), ("scared", "恐惧", "adj."), ("fearful", "恐惧", "adj.")],
    ]),
    ("反查", "表示方法、方式的名词有哪些", 5, [
        [("method", "方法", "n.")], [("approach", "方法", "n."), ("means", "方法", "n."), ("way", "方法", "n.")],
    ]),
    ("反查", "表示把诗背下来、记在脑子里的动词", 5, [[("memorize", "记忆", "v.")]]),
    ("反查", "朝代、王朝用英语怎么说", 3, [[("dynasty", "朝代", "n.")]]),
    ("辨析", "method 和 approach 有什么区别", 2, [
        [("method", "方法", "n.")], [("approach", "方法", "n.")],
    ]),
    ("辨析", "afraid 和 terrified 有什么区别", 2, [
        [("afraid", "恐惧", "adj.")], [("terrified", "恐惧", "adj.")],
    ]),
    ("辨析", "swerve 和 veer 用来形容车突然改变方向有什么区别", 2, [
        [("swerve", "转弯", "v."), ("swerve", "拐弯", "v.")], [("veer", "转弯", "v.")],
    ]),
    ("查无此词", "zqxvplm 是什么意思", 3, []),
]

POS = {"v.": r"v(?:t|i)?\.", "n.": r"n\.", "adj.": r"adj\."}
QUERY_POS = {"动词": "v.", "名词": "n.", "形容词": "adj."}


def has_pos(row, pos):
    body = " ".join(row.get("definitions") or [row["body"]])
    return bool(re.search(r"(?<![A-Za-z])" + POS[pos], body, re.IGNORECASE))


def rank(results, targets):
    for n, row in enumerate(results, 1):
        if any(row["word"].lower() == word and theme in row["theme"] and has_pos(row, pos)
               for word, theme, pos in targets):
            return n
    return 0


def run_suite(retriever, search=None):
    search = search or retriever.search
    counts = Counter()
    totals = Counter()
    durations = []
    first_ranks = []
    for category, query, k, targets in CASES:
        started = time.perf_counter()
        results = search(query, top_k=k)
        durations.append((time.perf_counter() - started) * 1000)
        ranks = [rank(results, group) for group in targets]
        requested = next((pos for label, pos in QUERY_POS.items() if label in query), None)
        pos_ok = not requested or all(has_pos(row, requested) for row in results)
        passed = (all(ranks) and pos_ok) if targets else not results
        if targets:
            first = min((r for r in ranks if r), default=0)
            first_ranks.append(1 / first if first else 0)
        totals[category] += 1
        counts[category] += int(passed)
        top = ", ".join(f"{r['word']}（{r['theme']}）" for r in results)
        print(f"[{'PASS' if passed else 'FAIL'}] {query} | ranks={ranks} POS={pos_ok} | {top}")
    print("通过", ", ".join(f"{k} {counts[k]}/{totals[k]}" for k in totals),
          f"| 合计 {sum(counts.values())}/{len(CASES)}")
    print(f"首个正确义项 MRR={statistics.mean(first_ranks):.3f} | 热查询中位数 {statistics.median(durations[1:]):.1f} ms")
    return sum(counts.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-all", action="store_true", help="Exit nonzero if any check fails")
    parser.add_argument("--mode", choices=["hybrid", "bm25", "vector"], default="hybrid")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    retriever = Retriever(args.data_dir) if args.data_dir else Retriever()
    passed = run_suite(retriever, lambda query, top_k: retriever.search(query, top_k, mode=args.mode))
    if args.require_all and passed != len(CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

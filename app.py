#!/usr/bin/env python3
"""Gradio demo for the vocabulary RAG prototype. Run: python app.py"""
import os

# Some environments set NO_PROXY with bracketed IPv6 entries ([::1], ...),
# which makes httpx (used by gradio at import time) raise InvalidURL.
# Sanitise before importing gradio; harmless elsewhere.
for _k in ("NO_PROXY", "no_proxy"):
    if _k in os.environ:
        os.environ[_k] = ",".join(
            p.strip("[]") for p in os.environ[_k].split(",") if p.strip()
        )

import gradio as gr

from rag import Retriever, answer, format_sources, llm_config

retriever = Retriever()
generation_ready = llm_config() is not None


def run(query, top_k, use_llm=True):
    if not query.strip():
        return "请输入问题。", ""
    text, results = answer(retriever, query, top_k=int(top_k), use_llm=use_llm)
    return text, "\n".join(format_sources(results))


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(label="问题", placeholder="例如：swerve 是什么意思 / 表示突然转弯的动词有哪些 / method 和 approach 有什么区别"),
        gr.Slider(3, 15, value=8, step=1, label="召回词条数 top-k"),
        gr.Checkbox(label="用大模型整理回答", value=False, interactive=generation_ready),
    ],
    outputs=[
        gr.Markdown(label="回答"),
        gr.Textbox(label="引用出处", lines=6),
    ],
    title="英语词义分类 RAG 知识库（原型）",
    description=f"基于《英语词义分类数据库（大学版）》的 {len(retriever.rows):,} 个词条。"
                "支持查词、中文反查和近义词辨析。"
                + ("大模型已配置，勾选后会将问题和召回的词条发给模型接口。"
                   if generation_ready else "当前使用本地词条检索。"),
    examples=[
        ["black market 是什么意思", 8],
        ["表示突然转弯的动词有哪些", 8],
        ["method 和 approach 有什么区别", 8],
        ["形容一个人意志坚定的词", 8],
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)

# 英语词义分类 RAG 知识库

一个在本机运行的英语学习工具。可以用中文找英文词，查单词和短语，也可以对照近义词在同一语境里的用法。每条结果附原始笔记路径。可选接入 DeepSeek，把找到的释义和例句整理成带引用的答案。

**词汇数据来源**　项目使用的《英语词义分类数据库（大学版）》由姜忠成（猛虎狂奔）编撰。[猛虎狂奔的哔哩哔哩主页](https://space.bilibili.com/1091814913)

## 试试这些问题

| 想做什么 | 可以这样问 | 会看到什么 |
| --- | --- | --- |
| 查短语 | `black market 是什么意思` | 完整短语、中文释义和出处 |
| 中文反查 | `表示突然转弯的动词有哪些` | `swerve`、`veer` 等候选，按动词筛选 |
| 近义词辨析 | `method 和 approach 有什么区别` | 两者在「方法、方式」语境中的释义和搭配 |
| 按组学习 | `method 和 approach 有什么区别，这一组还有哪些` | 相关义项和同组词 |

以上问题可以直接用随仓库提供的自写小词库试跑。

## 在 Windows 上运行

在仓库目录打开 PowerShell。以下步骤使用 Python 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path -LiteralPath data\vault)) {
    New-Item -ItemType Directory -Path data -Force | Out-Null
    Copy-Item -LiteralPath examples\vault -Destination data\vault -Recurse
}
.\.venv\Scripts\python.exe parse_vault.py
.\.venv\Scripts\python.exe build_index.py
.\.venv\Scripts\python.exe app.py
```

打开 [本地网页](http://127.0.0.1:7860/)，输入上面的任意问题。首次建索引时会下载多语言 MiniLM。检索在本机完成，直接显示释义、例句和出处。

**使用完整词库**时，请从作者渠道获取原始 Obsidian 笔记，使主题目录直接位于 `data/vault/` 下，然后重新执行解析和索引构建命令。已有示例词库时，先移走 `data/vault/` 中的 `01出行`、`02学习`、`03商业` 三个示例目录。仓库提供项目代码和自写演示笔记，原作者词库与生成的索引由使用者在本地准备。使用词库请遵循原版随附的版权声明。

### 可选接入 DeepSeek

在项目根目录新建 `.env`，填入自己的密钥，重启网页后勾选「用大模型整理回答」。

```dotenv
DEEPSEEK_API_KEY=你的密钥
LLM_MODEL=deepseek-flash
```

勾选后，问题和召回的词条会发送给模型接口。接口失败或引用编号无效时，网页会显示词典原文。也可以通过 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL` 配置其他 OpenAI 兼容接口。

## 检索怎么做

1. **按词条入库。** 解析 Obsidian 笔记中的 HTML 标记，保留词性、义项、主题、例句和原始路径。用词头、词性、简短释义和主题生成向量。
2. **按问题找义项。** 查词优先匹配完整词头或短语，中文反查融合 BM25 和 MiniLM 向量分数，并按明确指定的词性筛选。辨析时优先召回两个词同一主题分组下的义项。
3. **按需生成。** 将召回条目编号交给 DeepSeek，检查返回的引用编号能否对应原词条。索引在本地构建并整体更新，词条、向量和 BM25 数据一起保存。

## 评测

在完整词库上共索引 **29,470 条词条记录**，覆盖 **9,681 个不同词头**。同一套 18 条人工标注的回归题覆盖短语、词性、义项、辨析和查无此词。

| 检索方式 | 通过 |
| --- | ---: |
| 旧版混合检索 | 6/18 |
| 新版 BM25 + 词条规则 | 16/18 |
| 新版 MiniLM + 词条规则 | 14/18 |
| 新版混合检索 | 18/18 |

用完整词库建立索引后，可用 `python eval_retrieval.py --require-all` 复跑。这组结果用于回归检查，生成答案的引用内容按案例核对。

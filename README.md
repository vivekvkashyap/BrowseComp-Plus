# BrowseComp-Plus

| [Dataset](https://huggingface.co/datasets/Tevatron/browsecomp-plus) | [Leaderboard](https://huggingface.co/spaces/Tevatron/BrowseComp-Plus) | [Paper](https://arxiv.org/pdf/2508.06600) | [Project Page](https://texttron.github.io/BrowseComp-Plus/) |

BrowseComp-Plus evaluates Deep-Research agents against a fixed corpus of ~100K documents using 830 reasoning-intensive queries from OpenAI's BrowseComp. Instead of the live web, agents search a local index -- making evaluations **fair, reproducible, and transparent**.

---

## Repo Structure

```
BrowseComp-Plus/
├── search_agent/                  # LLM agent clients
│   ├── openai_client.py           # OpenAI (gpt-4.1, gpt-4o-mini, o3, ...)
│   ├── anthropic_client.py        # Anthropic (claude-sonnet-4, ...)
│   ├── bedrock_client.py          # AWS Bedrock (anthropic.claude-3-haiku-..., ...)
│   ├── glm_zai_client.py          # GLM / Z.AI (glm-4.6, ...)
│   ├── compact_utils.py           # Compact (summarization) tool logic
│   ├── prompts.py                 # Query templates
│   └── utils.py                   # Shared helpers
├── searcher/
│   └── searchers/
│       ├── bm25_searcher.py       # BM25 search via Pyserini
│       ├── faiss_searcher.py      # Dense retrieval via FAISS
│       └── custom_searcher.py     # Plug in your own retriever
├── topics-qrels/
│   ├── queries.tsv                # 830 queries (id \t question)
│   ├── qrel_evidence.txt          # Evidence relevance judgments
│   └── qrel_golds.txt             # Gold relevance judgments
├── scripts_build_index/
│   ├── decrypt_dataset.py         # Download & decrypt the dataset
│   └── download_indexes.sh        # Download pre-built BM25/FAISS indexes
├── scripts_evaluation/
│   ├── evaluate_run.py            # Evaluate agent outputs (LLM-as-judge)
│   └── evaluate_with_openai.py    # Evaluate using OpenAI models
├── runs/                          # Output directory (one JSON per query)
├── indexes/                       # Local search indexes (after download)
└── data/                          # Decrypted dataset files
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent Loop                           │
│                                                             │
│  User Query ──► LLM (gpt-4.1 / claude / glm)               │
│                   │                                         │
│                   ├──► search("query")                      │
│                   │       └── BM25 index returns k docs     │
│                   │           (each truncated to N tokens)  │
│                   │                                         │
│                   ├──► compact()        [optional]           │
│                   │       └── Summarizes conversation so    │
│                   │           far to free context space     │
│                   │                                         │
│                   ├──► search("refined query") ...          │
│                   │                                         │
│                   └──► Final Answer                         │
│                                                             │
│  Loop repeats until: answer given OR max_iterations hit     │
│                       OR max_tokens exhausted               │
└─────────────────────────────────────────────────────────────┘
```

Each iteration = one API call to the LLM. The LLM decides whether to search, compact, or answer.

---

## Setup

### 1. Install dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
uv sync
source .venv/bin/activate
```

Java 21 is also required (for Pyserini/BM25):

```bash
# via conda
conda install -c conda-forge openjdk=21

# or via apt
sudo apt update && sudo apt install -y openjdk-21-jdk
```

### 2. Download & decrypt the dataset

```bash
pip install datasets
huggingface-cli login  # if needed

python scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl \
  --generate-tsv topics-qrels/queries.tsv
```

This generates:
- `data/browsecomp_plus_decrypted.jsonl` -- full decrypted dataset
- `topics-qrels/queries.tsv` -- 830 queries (id + question), used as input

### 3. Download the BM25 index

```bash
huggingface-cli download Tevatron/browsecomp-plus-indexes \
  --repo-type=dataset \
  --include="bm25/*" \
  --local-dir ./indexes
```

After this, the index lives at `./indexes/bm25/`.

### 4. Set API keys

```bash
# Pick the one(s) you need:
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export ZAI_API_KEY="..."

# For Bedrock: configure AWS credentials
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"

# For W&B logging (optional):
# wandb login
```

---

## Running the Agent

All clients follow the same pattern. Pass `--query` with either:
- A **single question** in quotes (for testing)
- A **path to the TSV file** (for running the full 830-query dataset)

W&B logging is enabled by default. Add `--no-wandb` to disable, or `--wandb-entity your-team` for a shared team workspace. First-time W&B setup: `pip install wandb && wandb login`.

### Arguments Reference

| Argument | What it does | Default |
|---|---|---|
| `--query` | Single question (in quotes) or path to TSV file | `topics-qrels/queries.tsv` |
| `--model` | LLM model name | varies per client |
| `--max_tokens` | Total output token budget across all iterations per query | `10000` |
| `--max-iterations` | Max conversation rounds (search/compact/answer) per query | `100` |
| `--k` | Number of documents returned per search call | `5` |
| `--snippet-max-tokens` | Max tokens per document snippet (truncation limit) | `512` |
| `--searcher-type` | Retriever backend: `bm25`, `faiss`, `reasonir`, `custom` | (required) |
| `--index-path` | Path to the local search index | (required for bm25) |
| `--query-template` | Prompt template (see below) | `QUERY_TEMPLATE_NO_GET_DOCUMENT` |
| `--compact-model` | Model for the compact/summarization tool | same as `--model` |
| `--output-dir` | Where to save result JSONs (one per query) | varies per client |
| `--num-threads` | Parallel threads for processing multiple queries | `1` |
| `--get-document` | Flag: also register the `get_document` tool | disabled |
| `--system` | Optional system prompt override | none |
| `--temperature` | Sampling temperature | model default |
| `--top_p` | Top-p sampling | model default |
| `--wandb-project` | W&B project name | `browsecomp-evaluation` |
| `--wandb-entity` | W&B entity/team name | none |
| `--wandb-tags` | Space-separated tags for the W&B run | none |
| `--no-wandb` | Disable W&B logging | W&B enabled |

**Client-specific:**

| Argument | Client | What it does | Default |
|---|---|---|---|
| `--reasoning-effort` | OpenAI | `low`, `medium`, `high` (o-series) | none |
| `--no-thinking` | Anthropic | Disable extended thinking | thinking enabled |
| `--thinking-budget` | Anthropic | Token budget for extended thinking | `8192` |
| `--max-rate-limit-retries` | Anthropic | Retries on rate-limit errors | `5` |
| `--rate-limit-delay` | Anthropic | Delay (seconds) between retries | `60` |
| `--region` | Bedrock | AWS region for Bedrock | `us-east-1` |

### What the key arguments mean

- **`--max-iterations 100`** -- The agent can do up to 100 rounds of search/compact/think before being forced to stop. Most queries finish in 5-20 rounds.
- **`--k 5`** -- Each `search()` call returns the top 5 documents from BM25.
- **`--snippet-max-tokens 512`** -- Each document is truncated to 512 tokens. So one search returns ~2,500 tokens of content (5 docs x 512 tokens).
- **`--max_tokens 10000`** -- The total output token budget for the entire agent run on one query. This covers the model's reasoning, tool calls, and final answer across all iterations. If exhausted, the agent stops.
- **`--compact-model`** -- When the agent calls `compact()`, a separate API call summarizes the conversation history using this model. Using a cheaper model here (e.g., `gpt-4o-mini`) saves cost while the main agent uses a stronger model.

---

## Single Query Test

### OpenAI

```bash
python search_agent/openai_client.py \
  --query "What is the capital of France?" \
  --model gpt-4o-mini \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --compact-model gpt-4o-mini \
  --output-dir runs/test_openai/ \
  --max_tokens 10000 \
  --k 5 \
  --wandb-project browsecomp-evaluation \
  --wandb-tags test openai
```

### Anthropic

```bash
python search_agent/anthropic_client.py \
  --query "What is the capital of France?" \
  --model claude-sonnet-4-20250514 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --compact-model claude-sonnet-4-20250514 \
  --output-dir runs/test_anthropic/ \
  --max_tokens 10000 \
  --k 5 \
  --wandb-project browsecomp-evaluation \
  --wandb-tags test anthropic
```

### GLM (Z.AI)

```bash
python search_agent/glm_zai_client.py \
  --query "What is the capital of France?" \
  --model glm-4.6 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --compact-model glm-4.6 \
  --output-dir runs/test_glm/ \
  --max_tokens 20000 \
  --k 5 \
  --wandb-project browsecomp-evaluation \
  --wandb-tags test glm
```

### AWS Bedrock

```bash
python search_agent/bedrock_client.py \
  --query "What is the capital of France?" \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --region us-east-1 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --compact-model anthropic.claude-3-haiku-20240307-v1:0 \
  --output-dir runs/test_bedrock/ \
  --max_tokens 10000 \
  --k 5 \
  --wandb-project browsecomp-evaluation \
  --wandb-tags test bedrock
```

> Bedrock requires AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`). Other models: `anthropic.claude-3-sonnet-20240229-v1:0`, `anthropic.claude-3-5-sonnet-20240620-v1:0`.

---

## Full Dataset (830 Queries)

Point `--query` at the TSV file. The script saves one JSON per query and **automatically resumes** if interrupted.

### OpenAI

```bash
python search_agent/openai_client.py \
  --query topics-qrels/queries.tsv \
  --model gpt-4.1 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --compact-model gpt-4o-mini \
  --output-dir runs/bm25/gpt_4_1_compact/ \
  --max_tokens 10000 \
  --max-iterations 100 \
  --k 5 \
  --snippet-max-tokens 512 \
  --num-threads 4 \
  --wandb-project browsecomp-evaluation \
  --wandb-tags full-dataset bm25 gpt-4.1
```

### Anthropic

```bash
python search_agent/anthropic_client.py \
  --query topics-qrels/queries.tsv \
  --model claude-sonnet-4-20250514 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --compact-model claude-sonnet-4-20250514 \
  --output-dir runs/bm25/claude_compact/ \
  --max_tokens 10000 \
  --max-iterations 100 \
  --k 5 \
  --snippet-max-tokens 512 \
  --num-threads 4 \
  --wandb-project browsecomp-evaluation \
  --wandb-tags full-dataset bm25 claude-sonnet-4
```

### GLM (Z.AI)

```bash
python search_agent/glm_zai_client.py \
  --query topics-qrels/queries.tsv \
  --model glm-4.6 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --compact-model glm-4.6 \
  --output-dir runs/bm25/glm_compact/ \
  --max_tokens 20000 \
  --max-iterations 100 \
  --k 5 \
  --snippet-max-tokens 512 \
  --num-threads 4 \
  --wandb-project browsecomp-evaluation \
  --wandb-tags full-dataset bm25 glm-4.6
```

### AWS Bedrock

```bash
python search_agent/bedrock_client.py \
  --query topics-qrels/queries.tsv \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --region us-east-1 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --compact-model anthropic.claude-3-haiku-20240307-v1:0 \
  --output-dir runs/bm25/bedrock_compact/ \
  --max_tokens 10000 \
  --max-iterations 100 \
  --k 5 \
  --snippet-max-tokens 512 \
  --num-threads 4 \
  --wandb-project browsecomp-evaluation \
  --wandb-tags full-dataset bm25 bedrock-haiku
```

> Other Bedrock models: `anthropic.claude-3-sonnet-20240229-v1:0`, `anthropic.claude-3-5-sonnet-20240620-v1:0`

---

## Output Format

Each query produces a `run_<timestamp>.json` in the output directory:

```json
{
  "metadata": { "model": "gpt-4.1", "max_tokens": 10000, ... },
  "query_id": "769",
  "tool_call_counts": { "search": 6, "compact": 1 },
  "usage": {
    "input_tokens": 25000,
    "output_tokens": 3200,
    "total_tokens": 28200
  },
  "status": "completed",
  "retrieved_docids": ["12345", "67890", ...],
  "result": [
    { "type": "output_text", "output": "Explanation: ... Exact Answer: ... Confidence: ..." }
  ],
  "trajectory": [ ... ],
  "summarizer_usage": { "input_tokens": 5000, "output_tokens": 800, "num_calls": 1 }
}
```

---

## Evaluation

```bash
python scripts_evaluation/evaluate_run.py --input_dir runs/bm25/gpt_4_1_compact/
```

Uses Qwen3-32B as judge. Add `--tensor_parallel_size N` for multi-GPU.

---

## Query Templates

| Template | Tools Registered | Use Case |
|---|---|---|
| `QUERY_TEMPLATE_WITH_COMPACT` | search + compact | Recommended. Agent can summarize its history to free context. |
| `QUERY_TEMPLATE_WITH_COMPACT_AND_GET_DOCUMENT` | search + compact + get_document | Same + full document retrieval by docid. |
| `QUERY_TEMPLATE` | search + get_document | No compact. |
| `QUERY_TEMPLATE_NO_GET_DOCUMENT` | search only | Minimal. |
| `QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION` | search only | No citations in output. |

---

## Citation

```
@article{chen2025BrowseCompPlus,
      title={BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent}, 
      author={Chen, Zijian and Ma, Xueguang and Zhuang, Shengyao and Nie, Ping and Zou, Kai and Liu, Andrew and Green, Joshua and Patel, Kshama and Meng, Ruoxi and Su, Mingyi and Sharifymoghaddam, Sahel and Li, Yanxi and Hong, Haoran and Shi, Xinyu and Liu, Xuye and Thakur, Nandan and Zhang, Crystina and Gao, Luyu and Chen, Wenhu and Lin, Jimmy},
      year={2025},
      journal={arXiv preprint arXiv:2508.06600}
}
```

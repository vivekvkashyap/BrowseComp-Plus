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
```

---

## Running the Agent

All three clients follow the same pattern. Pass `--query` with either:
- A **single question** in quotes (for testing)
- A **path to the TSV file** (for running the full 830-query dataset)

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
| `--wandb-project` | W&B project name for logging | `browsecomp-evaluation` |
| `--wandb-entity` | W&B entity/team name (optional) | none |
| `--wandb-tags` | Space-separated tags for W&B run | none |
| `--no-wandb` | Disable W&B logging | W&B enabled |

**OpenAI-only:**

| Argument | What it does | Default |
|---|---|---|
| `--reasoning-effort` | `low`, `medium`, `high` (for o-series models) | none |

**Anthropic-only:**

| Argument | What it does | Default |
|---|---|---|
| `--no-thinking` | Disable extended thinking | thinking enabled |
| `--thinking-budget` | Token budget for extended thinking | `8192` |
| `--max-rate-limit-retries` | Retries on rate-limit errors | `5` |
| `--rate-limit-delay` | Delay (seconds) between retries | `60` |

**Bedrock-only:**

| Argument | What it does | Default |
|---|---|---|
| `--region` | AWS region for Bedrock | `us-east-1` |

**W&B Logging (all clients):**

| Argument | What it does | Default |
|---|---|---|
| `--wandb-project` | W&B project name | `browsecomp-evaluation` |
| `--wandb-entity` | W&B entity/team name | none (uses default) |
| `--wandb-tags` | Space-separated tags for the run | none |
| `--no-wandb` | Disable W&B logging | W&B enabled |

### What the key arguments mean

- **`--max-iterations 100`** -- The agent can do up to 100 rounds of search/compact/think before being forced to stop. Most queries finish in 5-20 rounds.
- **`--k 5`** -- Each `search()` call returns the top 5 documents from BM25.
- **`--snippet-max-tokens 512`** -- Each document is truncated to 512 tokens. So one search returns ~2,500 tokens of content (5 docs x 512 tokens).
- **`--max_tokens 10000`** -- The total output token budget for the entire agent run on one query. This covers the model's reasoning, tool calls, and final answer across all iterations. If exhausted, the agent stops.
- **`--compact-model`** -- When the agent calls `compact()`, a separate API call summarizes the conversation history using this model. Using a cheaper model here (e.g., `gpt-4o-mini`) saves cost while the main agent uses a stronger model.

---

## Weights & Biases (W&B) Logging

BrowseComp-Plus includes built-in support for logging evaluation runs to [Weights & Biases](https://wandb.ai/). This enables tracking metrics, visualizing results, and comparing different runs.

### Setup

1. **Install W&B** (already included in dependencies):
   ```bash
   pip install wandb
   ```

2. **Login to W&B** (first time only):
   ```bash
   wandb login
   ```
   Or set the `WANDB_API_KEY` environment variable.

### What Gets Logged

For each instance (query), W&B tracks:
- **Tool call counts**: Number of `search`, `get_document`, and `compact` calls
- **Token usage**: Input tokens, output tokens, cached tokens, reasoning tokens
- **Summarizer usage**: Separate tracking for compact tool calls
- **Status**: Completion status (completed, tool_use, etc.)
- **Retrieved documents**: Count of retrieved document IDs
- **Trajectory**: Full conversation history for each instance

All instance-level data is stored in a **W&B Table** that can be queried, filtered, and analyzed in the W&B UI.

### Usage Examples

**Basic usage** (uses default project `browsecomp-evaluation`):
```bash
python search_agent/anthropic_client.py \
  --query topics-qrels/queries.tsv \
  --model claude-sonnet-4-20250514 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --output-dir runs/bm25/claude/
```

**Custom project and tags**:
```bash
python search_agent/openai_client.py \
  --query topics-qrels/queries.tsv \
  --model gpt-4.1 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --output-dir runs/bm25/gpt_4_1/ \
  --wandb-project my-evaluation-project \
  --wandb-entity my-team \
  --wandb-tags production bm25 gpt-4.1
```

**Disable W&B logging**:
```bash
python search_agent/glm_zai_client.py \
  --query topics-qrels/queries.tsv \
  --model glm-4.6 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --output-dir runs/bm25/glm/ \
  --no-wandb
```

### Viewing Results in W&B

1. Go to [wandb.ai](https://wandb.ai) and navigate to your project
2. Each run shows:
   - **Metrics**: Per-instance tool calls, token usage, etc.
   - **Tables**: Queryable table with all instance-level data
   - **Trajectories**: Full conversation logs for each query
   - **Summary**: Aggregate statistics across all instances

3. Use W&B's filtering and grouping features to:
   - Compare different models or configurations
   - Analyze token usage patterns
   - Identify queries with high tool call counts
   - Track evaluation progress over time

### Notes

- W&B logging is **optional** - existing JSON file logging continues to work unchanged
- If W&B is unavailable or not logged in, the script continues without W&B (graceful degradation)
- All instance data is logged both to JSON files (for local analysis) and W&B (for visualization)

---

## Quick Test (Single Query)

### OpenAI

```bash
source .venv/bin/activate

python search_agent/openai_client.py \
  --query "What is the capital of France?" \
  --model gpt-4o-mini \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --compact-model gpt-4o-mini \
  --output-dir runs/test_openai/ \
  --max_tokens 10000 \
  --max-iterations 100 \
  --k 5 \
  --snippet-max-tokens 512
```

### Anthropic

```bash
source .venv/bin/activate

python search_agent/anthropic_client.py \
  --query "What is the capital of France?" \
  --model claude-sonnet-4-20250514 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --compact-model claude-sonnet-4-20250514 \
  --output-dir runs/test_anthropic/ \
  --max_tokens 10000 \
  --max-iterations 100 \
  --k 5 \
  --snippet-max-tokens 512
```

### GLM (Z.AI)

```bash
source .venv/bin/activate

python search_agent/glm_zai_client.py \
  --query "What is the capital of France?" \
  --model glm-4.6 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --compact-model glm-4.6 \
  --output-dir runs/test_glm/ \
  --max_tokens 20000 \
  --max-iterations 100 \
  --k 5 \
  --snippet-max-tokens 512
```

### AWS Bedrock

```bash
source .venv/bin/activate

# Ensure AWS credentials are configured (via environment variables or AWS CLI)
# export AWS_ACCESS_KEY_ID=your_access_key
# export AWS_SECRET_ACCESS_KEY=your_secret_key
# export AWS_SESSION_TOKEN=your_session_token  # (optional)

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
  --max-iterations 100 \
  --k 5 \
  --snippet-max-tokens 512
```

---

## Full Dataset (830 Queries)

Just point `--query` at the TSV file. The script processes all queries and saves one JSON per query in the output directory.

**Resume support:** If a run is interrupted, re-run the same command. It automatically skips already-completed queries by checking existing JSONs in the output directory.

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
  --num-threads 4
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
  --num-threads 4
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
  --num-threads 4
```

### AWS Bedrock

```bash
# Ensure AWS credentials are configured
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
  --num-threads 4
```

**Alternative models:**
- Sonnet: `anthropic.claude-3-sonnet-20240229-v1:0`
- Opus: `anthropic.claude-3-opus-20240229-v1:0`
- Latest Sonnet: `anthropic.claude-3-5-sonnet-20240620-v1:0`

**Note:** Ensure AWS credentials are set:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

---

## Weights & Biases (W&B) Logging

All three client scripts support optional logging to [Weights & Biases](https://wandb.ai/home) for tracking evaluation runs, metrics, and instance-level data.

### Setup

1. Install W&B (already included in dependencies):
   ```bash
   pip install wandb
   ```

2. Login to W&B (one-time setup):
   ```bash
   wandb login
   ```
   Or set the `WANDB_API_KEY` environment variable.

### Team Collaboration Setup

When working in a team, use the `--wandb-entity` flag to log results to a **shared team workspace**. This allows all team members to view and compare results in the same W&B project.

1. **Create or join a team** on [wandb.ai/home](https://wandb.ai/home):
   - Go to your W&B dashboard
   - Create a team/organization or get invited to an existing one
   - Note your team name (e.g., `my-research-team`)

2. **Use the team entity** when running experiments:
   ```bash
   --wandb-entity your-team-name
   ```

3. **All team members** should use the same entity name to log to the shared workspace.

**Example for team usage:**
```bash
python search_agent/openai_client.py \
  --query topics-qrels/queries.tsv \
  --model gpt-4.1 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --wandb-project browsecomp-evaluation \
  --wandb-entity my-research-team \
  --wandb-tags team-run experiment-1 \
  --output-dir runs/bm25/gpt_4_1_compact/
```

This logs all results to `my-research-team/browsecomp-evaluation`, making them visible to all team members with access.

### W&B Arguments

| Argument | What it does | Default |
|---|---|---|
| `--wandb-project` | W&B project name | `browsecomp-evaluation` |
| `--wandb-entity` | W&B entity/team name (optional) | none |
| `--wandb-tags` | Space-separated tags for the run | none |
| `--no-wandb` | Disable W&B logging | W&B enabled by default |

### What Gets Logged

**Per-instance metrics:**
- Tool call counts (search, get_document, compact)
- Token usage (input, output, cached, reasoning tokens)
- Summarizer usage (separate tracking for compact tool)
- Status (completed/failed/truncated)
- Retrieved document count
- Trajectory length (number of conversation turns)

**W&B Tables:**
- Instance-level data table with all metrics queryable in W&B UI
- Full conversation trajectories for each instance

**Run-level metrics:**
- Total instances processed
- Aggregate statistics

### Usage Examples

#### Single Query with W&B

**OpenAI:**
```bash
python search_agent/openai_client.py \
  --query "What is the capital of France?" \
  --model gpt-4o-mini \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --wandb-project browsecomp-evaluation \
  --wandb-tags test single-query \
  --output-dir runs/test_openai/
```

**Anthropic:**
```bash
python search_agent/anthropic_client.py \
  --query "What is the capital of France?" \
  --model claude-sonnet-4-20250514 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --wandb-project browsecomp-evaluation \
  --wandb-tags test single-query \
  --output-dir runs/test_anthropic/
```

**GLM (Z.AI):**
```bash
python search_agent/glm_zai_client.py \
  --query "What is the capital of France?" \
  --model glm-4.6 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --wandb-project browsecomp-evaluation \
  --wandb-tags test single-query \
  --output-dir runs/test_glm/
```

#### Full Dataset with W&B

**OpenAI:**
```bash
python search_agent/openai_client.py \
  --query topics-qrels/queries.tsv \
  --model gpt-4.1 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --wandb-project browsecomp-evaluation \
  --wandb-entity your-team-name \
  --wandb-tags full-dataset bm25 gpt-4.1 \
  --output-dir runs/bm25/gpt_4_1_compact/ \
  --num-threads 4
```

**Anthropic:**
```bash
python search_agent/anthropic_client.py \
  --query topics-qrels/queries.tsv \
  --model claude-sonnet-4-20250514 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --wandb-project browsecomp-evaluation \
  --wandb-entity your-team-name \
  --wandb-tags full-dataset bm25 claude-sonnet-4 \
  --output-dir runs/bm25/claude_compact/ \
  --num-threads 4
```

**GLM (Z.AI):**
```bash
python search_agent/glm_zai_client.py \
  --query topics-qrels/queries.tsv \
  --model glm-4.6 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --wandb-project browsecomp-evaluation \
  --wandb-entity your-team-name \
  --wandb-tags full-dataset bm25 glm-4.6 \
  --output-dir runs/bm25/glm_compact/ \
  --num-threads 4
```

**AWS Bedrock:**
```bash
# Ensure AWS credentials are configured first
python search_agent/bedrock_client.py \
  --query topics-qrels/queries.tsv \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --region us-east-1 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --wandb-project browsecomp-evaluation \
  --wandb-entity your-team-name \
  --wandb-tags full-dataset bm25 bedrock-claude-haiku \
  --output-dir runs/bm25/bedrock_compact/ \
  --num-threads 4
```

#### Disable W&B Logging

To disable W&B logging (useful for local testing or when W&B is unavailable):

```bash
python search_agent/openai_client.py \
  --query topics-qrels/queries.tsv \
  --model gpt-4.1 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --no-wandb \
  --output-dir runs/bm25/gpt_4_1_compact/
```

### Viewing Results in W&B

1. Go to [wandb.ai/home](https://wandb.ai/home) and navigate to your project
   - **Personal workspace**: `https://wandb.ai/your-username/browsecomp-evaluation`
   - **Team workspace**: `https://wandb.ai/team-name/browsecomp-evaluation`

2. Each run shows:
   - **Metrics**: Per-instance tool calls, token usage, success rates
   - **Tables**: Queryable instance-level data with all metrics
   - **Trajectories**: Full conversation history for each instance
   - **Config**: Experiment configuration (model, searcher type, etc.)

3. **Team collaboration features**:
   - Compare runs across different team members
   - Filter by tags to see specific experiments
   - Export tables for analysis
   - Share run links with team members

The W&B Tables allow you to filter, sort, and analyze instances by any metric (e.g., "show all instances with >10 search calls" or "instances that failed").

**Pro tip for teams:** Use consistent tagging conventions (e.g., `--wandb-tags member-name model-version date`) to make it easy to find and compare team members' experiments.

---

## Model Migration Guide

### Migrating from GLM to AWS Bedrock

If you were previously using GLM and want to switch to AWS Bedrock (Claude models), here's a direct comparison:

**Previous GLM command:**
```bash
python search_agent/glm_zai_client.py \
  --query topics-qrels/queries.tsv \
  --model glm-4.6 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --output-dir runs/bm25/glm_compact_3/ \
  --max_tokens 10000 \
  --max-iterations 100 \
  --k 5 \
  --snippet-max-tokens 512 \
  --num-threads 4
```

**Equivalent Bedrock command:**
```bash
# 1. Set AWS credentials first
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# 2. Run with Bedrock
python search_agent/bedrock_client.py \
  --query topics-qrels/queries.tsv \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --region us-east-1 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --compact-model anthropic.claude-3-haiku-20240307-v1:0 \
  --output-dir runs/bm25/bedrock_compact_3/ \
  --max_tokens 10000 \
  --max-iterations 100 \
  --k 5 \
  --snippet-max-tokens 512 \
  --num-threads 4
```

**Key differences:**
1. **Script name:** `glm_zai_client.py` → `bedrock_client.py`
2. **Model format:** `glm-4.6` → `anthropic.claude-3-haiku-20240307-v1:0` (Bedrock model ID format)
3. **Added argument:** `--region us-east-1` (specify AWS region)
4. **Added argument:** `--compact-model` (required for compact tool)
5. **Credentials:** GLM API key → AWS credentials (environment variables or AWS CLI config)

**Available Bedrock models:**
- **Haiku** (fast, economical): `anthropic.claude-3-haiku-20240307-v1:0`
- **Sonnet** (balanced): `anthropic.claude-3-sonnet-20240229-v1:0`
- **Sonnet 3.5** (latest): `anthropic.claude-3-5-sonnet-20240620-v1:0`
- **Opus** (most capable): `anthropic.claude-3-opus-20240229-v1:0`

**Output format:** Same JSON format, compatible with existing evaluation scripts.

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

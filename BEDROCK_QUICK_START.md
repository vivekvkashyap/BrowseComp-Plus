# Quick Start: Migrating from GLM to AWS Bedrock

## Your Previous GLM Command

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

## Your New Bedrock Command

### Step 1: Set AWS Credentials

```bash
# Set these environment variables with your AWS credentials
export AWS_ACCESS_KEY_ID=your_access_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_key_here
export AWS_DEFAULT_REGION=us-east-1
```

Alternatively, configure via AWS CLI:
```bash
aws configure
```

### Step 2: Run with Bedrock

```bash
cd /home/ece/Pavan/vivek_3/min_verify/BrowseComp-Plus
source .venv/bin/activate

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

## What Changed?

| Aspect | GLM | Bedrock |
|--------|-----|---------|
| **Script** | `glm_zai_client.py` | `bedrock_client.py` |
| **Model ID** | `glm-4.6` | `anthropic.claude-3-haiku-20240307-v1:0` |
| **Region** | Not needed | `--region us-east-1` (required) |
| **Compact Model** | Optional | `--compact-model anthropic.claude-3-haiku-20240307-v1:0` (required if using compact) |
| **Credentials** | GLM API key (env var) | AWS credentials (env vars or AWS CLI) |
| **Output Format** | JSON | JSON (same format) |

## Available Bedrock Models

Choose the model that fits your needs:

### Claude 3 Haiku (Recommended for testing)
```bash
--model anthropic.claude-3-haiku-20240307-v1:0
```
- **Best for:** Cost-effective testing and production
- **Speed:** Fastest
- **Cost:** Lowest (~$0.25/M input tokens)

### Claude 3 Sonnet
```bash
--model anthropic.claude-3-sonnet-20240229-v1:0
```
- **Best for:** Balanced performance
- **Speed:** Medium
- **Cost:** Medium

### Claude 3.5 Sonnet (Latest)
```bash
--model anthropic.claude-3-5-sonnet-20240620-v1:0
```
- **Best for:** Best overall performance
- **Speed:** Fast
- **Cost:** Medium-high

### Claude 3 Opus
```bash
--model anthropic.claude-3-opus-20240229-v1:0
```
- **Best for:** Most complex tasks
- **Speed:** Slower
- **Cost:** Highest (check region availability)

## Logging Output

The Bedrock client generates the same JSON output format as GLM:

**Output location:**
```
runs/bm25/bedrock_compact_3/run_<timestamp>.json
```

**Log structure includes:**
- ✅ Tool call counts (search, get_document, compact)
- ✅ Token usage (input, output, cached, total)
- ✅ Summarizer usage (separate tracking for compact calls)
- ✅ Retrieved document IDs
- ✅ Full trajectory (conversation history)
- ✅ Status (completed, tool_use, etc.)

**Example log:**
```json
{
  "metadata": {
    "model": "anthropic.claude-3-haiku-20240307-v1:0",
    "output_dir": "runs/bm25/bedrock_compact_3/",
    "max_tokens": 10000
  },
  "query_id": "123",
  "tool_call_counts": {
    "search": 5,
    "compact": 1
  },
  "usage": {
    "input_tokens": 12345,
    "input_tokens_cached": 8000,
    "output_tokens": 2345,
    "included_reasoning_tokens": null,
    "total_tokens": 14690
  },
  "status": "completed",
  "retrieved_docids": ["doc1", "doc2", ...],
  "result": [...],
  "trajectory": [...],
  "summarizer_usage": {
    "input_tokens": 500,
    "output_tokens": 200,
    "total_tokens": 700,
    "num_calls": 1
  }
}
```

## Prerequisites Checklist

Before running, ensure:

- [ ] AWS account with Bedrock access
- [ ] Claude models enabled in AWS Bedrock console (us-east-1 region)
- [ ] AWS credentials configured (environment variables or AWS CLI)
- [ ] IAM permissions for `bedrock:InvokeModel`
- [ ] Virtual environment activated (`source .venv/bin/activate`)
- [ ] boto3 installed (already in pyproject.toml)

## Testing

### Quick Single Query Test

```bash
cd /home/ece/Pavan/vivek_3/min_verify/BrowseComp-Plus
source .venv/bin/activate

python search_agent/bedrock_client.py \
  --query "What is the capital of France?" \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --region us-east-1 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --output-dir runs/test_bedrock/ \
  --max_tokens 10000 \
  --k 5
```

If this works, you're ready to run the full dataset!

## Troubleshooting

### Error: "Access Denied"
**Solution:** Enable Claude models in AWS Bedrock console
1. Go to AWS Console → Bedrock → Model access
2. Click "Manage model access"
3. Enable Anthropic Claude models
4. Wait for approval (usually instant)

### Error: "Model Not Found"
**Solution:** Check region availability
- Try `--region us-east-1` or `--region us-west-2`
- Not all models are available in all regions

### Error: "Module 'boto3' not found"
**Solution:** Install boto3 (should already be in dependencies)
```bash
source .venv/bin/activate
pip install boto3>=1.35.0
```

### Error: "Credentials not found"
**Solution:** Set AWS credentials
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

## Resume Capability

Just like GLM, if the run is interrupted, simply re-run the same command. The script automatically:
- ✅ Skips already-completed queries
- ✅ Continues from where it left off
- ✅ Checks existing JSON files in output directory

## Comparing Results

The output format is identical to GLM, so you can:
- Use the same evaluation scripts
- Compare results side-by-side
- Analyze with the same tools

## Need Help?

- See `docs/BEDROCK_SETUP.md` for detailed setup guide
- See `docs/BEDROCK_INTEGRATION.md` for technical details
- See main `README.md` for general BrowseComp-Plus documentation

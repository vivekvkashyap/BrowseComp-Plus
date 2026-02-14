# Bedrock Integration Summary

This document summarizes the AWS Bedrock integration added to BrowseComp-Plus.

## Files Created

### 1. `search_agent/bedrock_client.py`
- **Purpose**: Main client script for AWS Bedrock (using boto3)
- **Key Features**:
  - Uses boto3 to call AWS Bedrock Runtime API
  - Supports Claude models via Bedrock (e.g., `anthropic.claude-3-haiku-20240307-v1:0`)
  - Implements native tool calling with search, get_document, and compact tools
  - Full W&B logging support
  - Multi-threaded processing for TSV datasets
  - Resume capability (skips already-completed queries)
  - Configurable region (default: `us-east-1`)
  - Same output format as Anthropic and OpenAI clients

### 2. `docs/BEDROCK_SETUP.md`
- **Purpose**: Comprehensive setup and usage guide
- **Contents**:
  - AWS account and Bedrock prerequisites
  - AWS credentials configuration (3 methods)
  - IAM permissions required
  - Available Claude models on Bedrock
  - Usage examples (single query, full dataset, custom parameters)
  - Region availability information
  - Cost considerations
  - Troubleshooting guide
  - Comparison with direct Anthropic API

## Files Modified

### 1. `search_agent/compact_utils.py`
- **Added**: `call_compact_bedrock()` function
- **Purpose**: Handles conversation history summarization using Bedrock
- **Parameters**:
  - `client`: boto3 bedrock-runtime client
  - `model_id`: Bedrock model ID (e.g., `anthropic.claude-3-haiku-20240307-v1:0`)
  - `history_text`: Conversation history to summarize
  - `compact_prompt`: Optional custom prompt
- **Returns**: `(summary_text, usage_dict)` with token usage tracking

### 2. `pyproject.toml`
- **Added**: `boto3>=1.35.0` dependency
- **Purpose**: Required for AWS Bedrock API calls

### 3. `README.md`
- **Updated Sections**:
  - Repo Structure: Added `bedrock_client.py` entry
  - Command-Line Arguments: Added "Bedrock-only" section with `--region` argument
  - Quick Test examples: Added AWS Bedrock example with AWS credentials setup
  - Full Dataset examples: Added Bedrock full dataset processing example

## Usage Examples

### Quick Test (Single Query)
```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Run single query
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

### Full Dataset Processing
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
  --num-threads 4
```

## Key Features

### 1. AWS Credentials Support
- Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- AWS CLI configuration (~/.aws/credentials)
- IAM roles (for EC2/Lambda)

### 2. Tool Calling
- **search**: Query the BM25/FAISS index
- **get_document**: Retrieve full document by ID
- **compact**: Summarize conversation history to free context space

### 3. Logging
- JSON output files (one per query)
- W&B integration for experiment tracking
- Full trajectory logging
- Token usage tracking (input, output, cached, total)

### 4. Multi-threaded Processing
- Process multiple queries in parallel
- Configurable with `--num-threads`
- Automatic resume on interruption

### 5. Configuration Options
- Region selection (--region)
- Model selection (any Claude model available in Bedrock)
- Temperature and top_p control
- Query templates (with/without compact, get_document)
- Custom system prompts

## Bedrock-Specific Considerations

### Model IDs
Bedrock uses different model ID format than direct Anthropic API:
- Anthropic: `claude-3-haiku-20240307`
- Bedrock: `anthropic.claude-3-haiku-20240307-v1:0`

### Regions
Not all models are available in all regions. Common regions:
- us-east-1 (N. Virginia) - Most complete model selection
- us-west-2 (Oregon)
- eu-central-1 (Frankfurt)

### Pricing
- Charged per token (input + output)
- Pricing may differ from direct Anthropic API
- Track usage through AWS Cost Explorer or W&B

### IAM Permissions Required
```json
{
    "Effect": "Allow",
    "Action": ["bedrock:InvokeModel"],
    "Resource": ["arn:aws:bedrock:*::foundation-model/anthropic.claude-*"]
}
```

## Output Format

The Bedrock client produces identical output format to other clients:

```json
{
  "metadata": {
    "model": "anthropic.claude-3-haiku-20240307-v1:0",
    "output_dir": "runs/bm25/bedrock/",
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

## Testing Checklist

Before running on full dataset:

- [ ] AWS credentials configured
- [ ] Bedrock model access enabled in AWS Console
- [ ] Required IAM permissions attached
- [ ] Model available in selected region
- [ ] Dependencies installed (`boto3>=1.35.0`)
- [ ] Single query test successful
- [ ] Output JSON format correct
- [ ] W&B logging working (if enabled)

## Next Steps

1. Configure AWS credentials
2. Enable Bedrock model access in AWS Console
3. Run single-query test to verify setup
4. Process full dataset with desired configuration
5. Evaluate results using existing evaluation scripts

For detailed setup instructions, see `docs/BEDROCK_SETUP.md`.

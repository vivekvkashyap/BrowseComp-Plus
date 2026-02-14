# AWS Bedrock Client Setup Guide

This guide explains how to use the AWS Bedrock client for BrowseComp-Plus evaluation.

## Prerequisites

### 1. AWS Account and Bedrock Access

- You need an AWS account with access to AWS Bedrock
- Ensure you have enabled the Claude models in your AWS Bedrock console
- Navigate to: AWS Console → Bedrock → Model access → Request model access
- Enable the Anthropic Claude models you want to use

### 2. AWS Credentials

Configure your AWS credentials using one of these methods:

#### Option 1: Environment Variables
```bash
export AWS_ACCESS_KEY_ID=your_access_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_access_key
export AWS_SESSION_TOKEN=your_session_token  # Optional, for temporary credentials
export AWS_DEFAULT_REGION=us-east-1
```

#### Option 2: AWS CLI Configuration
```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
```

#### Option 3: IAM Role (for EC2/Lambda)
If running on AWS infrastructure, attach an IAM role with Bedrock permissions.

### 3. Required IAM Permissions

Your AWS credentials need the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
            ]
        }
    ]
}
```

## Available Models

AWS Bedrock supports various Anthropic Claude models. Common model IDs:

- `anthropic.claude-3-haiku-20240307-v1:0` - Fast and cost-effective
- `anthropic.claude-3-sonnet-20240229-v1:0` - Balanced performance
- `anthropic.claude-3-opus-20240229-v1:0` - Most capable (if available in your region)
- `anthropic.claude-3-5-sonnet-20240620-v1:0` - Latest Sonnet (check region availability)

Check the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) for the most up-to-date list.

## Installation

The boto3 dependency is already included in `pyproject.toml`:

```bash
# Install dependencies
uv sync
source .venv/bin/activate

# Or with pip
pip install boto3>=1.35.0
```

## Usage Examples

### Single Query Test

```bash
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

### Full Dataset Evaluation

```bash
python search_agent/bedrock_client.py \
  --query topics-qrels/queries.tsv \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --region us-east-1 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --compact-model anthropic.claude-3-haiku-20240307-v1:0 \
  --output-dir runs/bm25/bedrock/ \
  --max_tokens 10000 \
  --max-iterations 100 \
  --k 5 \
  --snippet-max-tokens 512 \
  --num-threads 4
```

### With Custom Temperature and Top-P

```bash
python search_agent/bedrock_client.py \
  --query topics-qrels/queries.tsv \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --region us-east-1 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --temperature 0.7 \
  --top_p 0.95 \
  --output-dir runs/bm25/bedrock_custom/ \
  --max_tokens 10000 \
  --k 5
```

### With W&B Logging

```bash
python search_agent/bedrock_client.py \
  --query topics-qrels/queries.tsv \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --region us-east-1 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --output-dir runs/bm25/bedrock_wandb/ \
  --wandb-project my-bedrock-evaluation \
  --wandb-entity my-team \
  --wandb-tags bedrock claude-haiku bm25 \
  --max_tokens 10000 \
  --k 5
```

## Command-Line Arguments

### Bedrock-Specific Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Bedrock model ID (e.g., `anthropic.claude-3-haiku-20240307-v1:0`) | `anthropic.claude-3-haiku-20240307-v1:0` |
| `--region` | AWS region for Bedrock service | `us-east-1` |

### Common Arguments

All standard BrowseComp-Plus arguments are supported:

- `--query`: Query text or path to TSV file
- `--searcher-type`: Type of searcher (`bm25`, `faiss`, `custom`)
- `--index-path`: Path to search index
- `--max_tokens`: Maximum output tokens
- `--k`: Number of search results per query
- `--snippet-max-tokens`: Tokens per document snippet
- `--query-template`: Query template to use
- `--compact-model`: Model for compact/summarization tool
- `--num-threads`: Parallel processing threads
- `--temperature`: Sampling temperature
- `--top_p`: Top-p sampling parameter

See the main README for complete argument documentation.

## Region Availability

Claude models are available in select AWS regions. Common regions:

- `us-east-1` (N. Virginia)
- `us-west-2` (Oregon)
- `eu-central-1` (Frankfurt)
- `ap-northeast-1` (Tokyo)

Check the [AWS Bedrock regions page](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html) for current availability.

## Cost Considerations

AWS Bedrock charges per token (input + output). To minimize costs:

1. **Use Haiku for testing**: Claude 3 Haiku is the most cost-effective option
2. **Set appropriate max_tokens**: Limit total output budget
3. **Use compact tool wisely**: The `--compact-model` can use a cheaper model for summarization
4. **Monitor usage**: Use AWS CloudWatch or W&B to track token consumption

Example pricing (as of documentation time, verify current rates):
- Claude 3 Haiku: ~$0.25 per million input tokens, ~$1.25 per million output tokens
- Claude 3 Sonnet: Higher cost, better performance

## Troubleshooting

### "Access Denied" Error

**Problem**: `ClientError: An error occurred (AccessDeniedException)`

**Solution**:
1. Verify your AWS credentials are configured correctly
2. Ensure you have the required IAM permissions (see above)
3. Check that you've requested model access in the Bedrock console

### "Model Not Found" Error

**Problem**: `ClientError: An error occurred (ResourceNotFoundException)`

**Solution**:
1. Verify the model ID is correct and matches your region
2. Ensure the model is available in your selected region
3. Check that you've enabled access to the model in Bedrock console

### Rate Limiting

Unlike the Anthropic API client, the Bedrock client does not have built-in rate limit retry logic. If you encounter throttling:

1. Reduce `--num-threads` to decrease parallel requests
2. Add manual delays between requests
3. Request a quota increase through AWS Support

### Region Errors

**Problem**: Model not available in region

**Solution**:
- Try a different region using `--region us-west-2` or `--region us-east-1`
- Check model availability for your region in AWS documentation

## Differences from Direct Anthropic API

The Bedrock client uses the same Claude models but accessed through AWS infrastructure:

**Advantages**:
- AWS billing and cost management
- Integration with AWS services
- Enterprise compliance features
- Potential lower latency if running on AWS infrastructure

**Differences**:
- Model IDs use AWS format (e.g., `anthropic.claude-3-haiku-20240307-v1:0`)
- AWS credentials instead of Anthropic API key
- Regional availability constraints
- No automatic rate limit retry (yet)

## Output Format

The Bedrock client produces the same output format as other clients:

- JSON files in `--output-dir` (one per query)
- Same structure as Anthropic/OpenAI clients
- Compatible with existing evaluation scripts
- Full W&B logging support

## Next Steps

1. Verify your AWS credentials and Bedrock access
2. Run a single-query test to confirm everything works
3. Process the full dataset with your chosen model
4. Evaluate results using the standard evaluation scripts

For evaluation, see the main README's "Evaluation" section.

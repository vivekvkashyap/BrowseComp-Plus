# Evaluating Claude Models via AWS Bedrock

First, please make sure that you have followed the main README.md to download the decrypted dataset, setup your environment, and downloaded the indexes you need.

## Prerequisites

1. **AWS Account** with Bedrock access
2. **Enable Claude models** in AWS Bedrock console (us-east-1 recommended)
3. **AWS Credentials** configured via:
   ```bash
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export AWS_DEFAULT_REGION=us-east-1
   ```
   Or use `aws configure` to set up credentials.

4. **IAM Permissions**: Ensure `bedrock:InvokeModel` permission for Claude models

## BM25

```bash
python search_agent/bedrock_client.py \
  --query topics-qrels/queries.tsv \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --region us-east-1 \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --compact-model anthropic.claude-3-haiku-20240307-v1:0 \
  --output-dir runs/bm25/bedrock-haiku/ \
  --max_tokens 10000 \
  --max-iterations 100 \
  --k 5 \
  --snippet-max-tokens 512 \
  --num-threads 4
```

**Available models:**
- Haiku: `anthropic.claude-3-haiku-20240307-v1:0` (fast, cost-effective)
- Sonnet: `anthropic.claude-3-sonnet-20240229-v1:0` (balanced)
- Sonnet 3.5: `anthropic.claude-3-5-sonnet-20240620-v1:0` (latest)

To evaluate the results:
```bash
python scripts_evaluation/evaluate_run.py --input_dir runs/bm25/bedrock-haiku --tensor_parallel_size 1
```

## FAISS (Dense Retrieval)

```bash
python search_agent/bedrock_client.py \
  --query topics-qrels/queries.tsv \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --region us-east-1 \
  --searcher-type faiss \
  --index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
  --model-name "Qwen/Qwen3-Embedding-8B" \
  --normalize \
  --query-template QUERY_TEMPLATE_WITH_COMPACT \
  --compact-model anthropic.claude-3-haiku-20240307-v1:0 \
  --output-dir runs/faiss/bedrock-haiku/ \
  --max_tokens 10000 \
  --k 5 \
  --num-threads 4
```

To evaluate:
```bash
python scripts_evaluation/evaluate_run.py --input_dir runs/faiss/bedrock-haiku --tensor_parallel_size 1
```

## Notes

- **Region availability:** Not all models are available in all regions. Check [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html)
- **Costs:** Bedrock charges per token. Use Haiku for cost-effective testing
- **Resume:** If interrupted, re-run the same command - it automatically skips completed queries
- **Output format:** Same JSON format as other clients, fully compatible with evaluation scripts

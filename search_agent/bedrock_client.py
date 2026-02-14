import argparse
import csv
import datetime
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from rich import print as rprint
from tqdm import tqdm

load_dotenv(override=False)

script_env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=script_env_path, override=False)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from compact_utils import call_compact_bedrock, format_history_for_compact
from prompts import format_query
from transformers import AutoTokenizer
from utils import extract_retrieved_docids_from_result

from searcher.searchers import SearcherType
from wandb_logger import WandbLogger


# ---------------------------------------------------------------------------
# SearchToolHandler -- local tool execution (Anthropic tool schema format)
# ---------------------------------------------------------------------------


class SearchToolHandler:
    """Manages tool definitions (Anthropic format for Bedrock) and local execution of
    search, get_document, and compact tools."""

    def __init__(
        self,
        searcher,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True,
        include_compact: bool = False,
    ):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document
        self.include_compact = include_compact

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return tool definitions in Anthropic's native format
        (using ``input_schema``, not the OpenAI ``parameters`` wrapper)."""
        tools: list[dict[str, Any]] = [
            {
                "name": "search",
                "description": self.searcher.search_description(self.k),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string",
                        }
                    },
                    "required": ["query"],
                },
            }
        ]

        if self.include_get_document:
            tools.append(
                {
                    "name": "get_document",
                    "description": self.searcher.get_document_description(),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "docid": {
                                "type": "string",
                                "description": "Document ID to retrieve",
                            }
                        },
                        "required": ["docid"],
                    },
                }
            )

        if self.include_compact:
            tools.append(
                {
                    "name": "compact",
                    "description": (
                        "Summarize and compact your conversation history to free up context space. "
                        "Call this proactively when you have accumulated many search results and your "
                        "context is getting long. This does NOT end your research — continue searching "
                        "after compacting. Takes no arguments."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                }
            )

        return tools

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "search":
            return self._search(arguments["query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _search(self, query: str) -> str:
        candidates = self.searcher.search(query, self.k)

        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    truncated_tokens = tokens[: self.snippet_max_tokens]
                    cand["snippet"] = self.tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({"docid": cand["docid"], "snippet": cand["snippet"]})
            else:
                results.append(
                    {
                        "docid": cand["docid"],
                        "score": cand["score"],
                        "snippet": cand["snippet"],
                    }
                )

        return json.dumps(results, indent=2)

    def _get_document(self, docid: str) -> str:
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Bedrock client helpers
# ---------------------------------------------------------------------------


def _make_bedrock_request(
    client,
    model_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int,
    system_prompt: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
):
    """Call Bedrock Runtime's invoke_model with Anthropic Claude format."""
    
    # Format the request payload using Claude's messages API
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": messages,
    }

    if tools:
        native_request["tools"] = tools

    if system_prompt:
        native_request["system"] = system_prompt

    if temperature is not None:
        native_request["temperature"] = temperature

    if top_p is not None:
        native_request["top_p"] = top_p

    # Convert the native request to JSON
    request_body = json.dumps(native_request)

    try:
        # Invoke the model
        response = client.invoke_model(
            modelId=model_id,
            body=request_body,
            contentType="application/json",
            accept="application/json"
        )

        # Decode the response body
        model_response = json.loads(response["body"].read())
        
        return model_response

    except ClientError as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        raise e
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        raise e


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_content_block(block) -> dict:
    """Serialize a content block to a JSON-safe dict."""
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="python")
    if hasattr(block, "to_dict"):
        return block.to_dict()
    return {"type": "text", "text": str(block)}


# ---------------------------------------------------------------------------
# Core conversation loop with Bedrock tool calling
# ---------------------------------------------------------------------------


def run_conversation_with_tools(
    client,
    *,
    query: str,
    model_id: str,
    max_tokens: int,
    tool_handler: SearchToolHandler,
    system_prompt: str | None = None,
    query_template: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_iterations: int = 100,
    compact_model: str | None = None,
):
    """Run a multi-turn conversation with Bedrock tool calling.

    Returns (normalized_results, cumulative_usage, stop_reason, trajectory).
    """
    tools = tool_handler.get_tool_definitions()

    # Resolve compact model: default to the agent model
    _compact_model = compact_model or model_id

    formatted_query = format_query(query, query_template)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": formatted_query},
    ]

    cumulative_usage = {
        "input_tokens": 0,
        "input_tokens_cached": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    summarizer_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "num_calls": 0,
    }

    normalized_results: list[dict[str, Any]] = []
    stop_reason: Optional[str] = None

    # --- Trajectory: flat conversation log ---
    
    trajectory: list[dict[str, Any]] = []
    if system_prompt:
        trajectory.append({"role": "system", "content": system_prompt})
    trajectory.append({"role": "user", "content": formatted_query})

    # Treat max_tokens as a global output budget across the entire conversation.
    global_max_tokens = max_tokens

    for iteration in range(max_iterations):
        remaining_tokens = global_max_tokens - cumulative_usage["output_tokens"]
        if remaining_tokens <= 0:
            print(f"Warning: Reached global max_tokens output budget ({global_max_tokens})")
            break

        response = _make_bedrock_request(
            client,
            model_id=model_id,
            messages=messages,
            tools=tools,
            max_tokens=min(remaining_tokens, global_max_tokens),
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
        )

        stop_reason = response.get("stop_reason")

        # Accumulate usage
        if "usage" in response:
            usage = response["usage"]
            cumulative_usage["input_tokens"] += usage.get("input_tokens", 0)
            cumulative_usage["output_tokens"] += usage.get("output_tokens", 0)
            cumulative_usage["total_tokens"] += (
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            )
            # Bedrock doesn't provide cache_creation_input_tokens separately
            # but we can track cache_read_input_tokens if available
            cumulative_usage["input_tokens_cached"] += usage.get("cache_read_input_tokens", 0)

        # Process content blocks
        content_blocks = response.get("content", [])
        trajectory.append({"role": "assistant", "content": content_blocks})

        # Process content blocks into normalized results
        for block in content_blocks:
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "")
                if text.strip():
                    normalized_results.append({
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": text,
                    })
            elif btype == "tool_use":
                normalized_results.append({
                    "type": "tool_call",
                    "tool_name": block.get("name"),
                    "arguments": json.dumps(block.get("input", {})),
                    "output": None,
                })

        # If no tool calls, we are done
        if stop_reason != "tool_use":
            break

        # --- Append assistant response to messages (contains tool_use blocks) ---
        messages.append({"role": "assistant", "content": content_blocks})

        # --- Execute tool calls and collect results ---
        tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]

        tool_results: list[dict[str, Any]] = []
        did_compact = False

        for block in tool_use_blocks:
            tool_name = block.get("name")
            tool_input = block.get("input", {})
            tool_use_id = block.get("id")

            try:
                # ---- Handle compact tool ----
                if tool_name == "compact":
                    print(f"[Compact] Summarizing conversation history (iter {iteration + 1})…")

                    history_text = format_history_for_compact(messages)
                    summary, compact_usage = call_compact_bedrock(client, _compact_model, history_text)
                    print(f"[Compact] Summary produced ({len(summary)} chars). Replacing history.")

                    # Accumulate summarizer token usage
                    if compact_usage:
                        summarizer_usage["input_tokens"] += compact_usage.get("input_tokens", 0)
                        summarizer_usage["output_tokens"] += compact_usage.get("output_tokens", 0)
                        summarizer_usage["total_tokens"] += compact_usage.get("total_tokens", 0)
                        summarizer_usage["num_calls"] += 1

                    result_text = "Context compacted successfully. Your conversation history has been summarized. Continue your research."

                    # Patch the last tool_call in normalized_results
                    for nr in reversed(normalized_results):
                        if nr.get("type") == "tool_call" and nr.get("tool_name") == "compact":
                            nr["output"] = result_text
                            break

                    # Record compact in trajectory
                    trajectory.append({
                        "role": "compact_summary",
                        "input": history_text,
                        "content": summary,
                    })
                    trajectory.append({
                        "role": "tool",
                        "tool_name": "compact",
                        "call_id": tool_use_id,
                        "content": result_text,
                    })

                    # Replace messages: original query + summary + continue prompt
                    messages = [
                        {"role": "user", "content": formatted_query},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Here is a summary of my prior research:\n\n" + summary
                                    ),
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": "Continue your research based on the summary above.",
                        },
                    ]

                    did_compact = True
                    # Stop processing other tool calls from this response
                    # (messages were reset, so any other pending calls are orphaned)
                    break

                # ---- Handle search / get_document tools ----
                output = tool_handler.execute_tool(tool_name, tool_input)

                # Patch the matching tool_call in normalized_results
                for nr in reversed(normalized_results):
                    if (
                        nr.get("type") == "tool_call"
                        and nr.get("tool_name") == tool_name
                        and nr.get("output") is None
                    ):
                        nr["output"] = output
                        break

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": output,
                })

                # Record tool result in trajectory
                trajectory.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "call_id": tool_use_id,
                    "content": output,
                })

            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                print(error_msg)

                # Patch error into normalized_results
                for nr in reversed(normalized_results):
                    if (
                        nr.get("type") == "tool_call"
                        and nr.get("tool_name") == tool_name
                        and nr.get("output") is None
                    ):
                        nr["output"] = error_msg
                        break

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": error_msg,
                    "is_error": True,
                })

                # Record tool error in trajectory
                trajectory.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "call_id": tool_use_id,
                    "content": error_msg,
                    "error": True,
                })

        # Append tool results to messages (unless compact reset messages)
        if not did_compact and tool_results:
            messages.append({"role": "user", "content": tool_results})

    if stop_reason is None:
        print(f"Warning: Conversation hit max iterations ({max_iterations}) without final response")

    return normalized_results, cumulative_usage, stop_reason, trajectory, summarizer_usage


# ---------------------------------------------------------------------------
# Persist response to JSON
# ---------------------------------------------------------------------------


def _persist_response(
    out_dir: str,
    *,
    model: str,
    query_id: str | None,
    system_prompt: str | None,
    max_tokens: int,
    normalized_results: list[dict[str, Any]],
    cumulative_usage: dict,
    stop_reason: Optional[str],
    trajectory: list[dict[str, Any]] | None = None,
    summarizer_usage: dict | None = None,
    wandb_logger: Optional[WandbLogger] = None,
    searcher_type: Optional[str] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    tool_call_counts: dict[str, int] = {}
    for item in normalized_results:
        if item.get("type") == "tool_call" and item.get("tool_name"):
            name = item["tool_name"]
            tool_call_counts[name] = tool_call_counts.get(name, 0) + 1

    normalized_usage = {
        "input_tokens": cumulative_usage.get("input_tokens", 0),
        "input_tokens_cached": cumulative_usage.get("input_tokens_cached", 0),
        "output_tokens": cumulative_usage.get("output_tokens", 0),
        "included_reasoning_tokens": None,
        "total_tokens": cumulative_usage.get("total_tokens", 0),
    }

    status = "completed" if stop_reason == "end_turn" else stop_reason

    ts = dt.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = os.path.join(out_dir, f"run_{ts}.json")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "model": model,
                    "output_dir": str(out_dir),
                    "max_tokens": max_tokens,
                },
                "query_id": query_id,
                "tool_call_counts": tool_call_counts,
                "usage": normalized_usage,
                "status": status,
                "retrieved_docids": extract_retrieved_docids_from_result(normalized_results),
                "result": normalized_results,
                "trajectory": trajectory or [],
                "summarizer_usage": summarizer_usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "num_calls": 0},
            },
            f,
            indent=2,
            default=str,
        )

    print("Saved response to", filename, "| tool call counts:", tool_call_counts)

    # Log to W&B if enabled
    if wandb_logger:
        retrieved_docids = extract_retrieved_docids_from_result(normalized_results)
        
        # Prepare full instance data for artifact storage
        full_instance_data = {
            "metadata": {
                "model": model,
                "output_dir": str(out_dir),
                "max_tokens": max_tokens,
            },
            "query_id": query_id,
            "tool_call_counts": tool_call_counts,
            "usage": normalized_usage,
            "status": status,
            "retrieved_docids": retrieved_docids,
            "result": normalized_results,
            "trajectory": trajectory or [],
            "summarizer_usage": summarizer_usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "num_calls": 0},
        }
        
        wandb_logger.log_instance(
            query_id=query_id,
            model=model,
            searcher_type=searcher_type or "unknown",
            tool_call_counts=tool_call_counts,
            usage=normalized_usage,
            summarizer_usage=summarizer_usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "num_calls": 0},
            status=status,
            retrieved_docids=retrieved_docids,
            trajectory=trajectory,
            json_file=filename,
            full_instance_data=full_instance_data,
        )


# ---------------------------------------------------------------------------
# TSV dataset processing
# ---------------------------------------------------------------------------


def _process_tsv_dataset(
    tsv_path: str, client, args, tool_handler: SearchToolHandler, wandb_logger: Optional[WandbLogger] = None
):
    dataset_path = Path(tsv_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    out_dir = Path(args.output_dir).expanduser().resolve()

    queries: list[tuple[str, str]] = []
    with dataset_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))

    processed_ids: set[str] = set()
    if out_dir.exists():
        for json_path in out_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid_saved = meta.get("query_id")
                    if qid_saved:
                        processed_ids.add(str(qid_saved))
            except Exception:
                continue

    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]

    print(
        f"Processing {len(remaining)} remaining queries (skipping {len(processed_ids)}) from {dataset_path} …"
    )

    completed_lock = threading.Lock()
    completed_count = [0]

    def _handle_single_query(qid: str, qtext: str, pbar=None):
        try:
            normalized_results, cumulative_usage, finish_reason, traj, sum_usage = run_conversation_with_tools(
                client,
                query=qtext,
                model_id=args.model,
                max_tokens=args.max_tokens,
                tool_handler=tool_handler,
                system_prompt=args.system,
                query_template=args.query_template,
                temperature=args.temperature,
                top_p=args.top_p,
                max_iterations=args.max_iterations,
                compact_model=getattr(args, "compact_model", None),
            )

            with completed_lock:
                completed_count[0] += 1
                if pbar:
                    pbar.set_postfix(completed=completed_count[0])

            _persist_response(
                str(out_dir),
                model=args.model,
                query_id=qid,
                system_prompt=args.system,
                max_tokens=args.max_tokens,
                normalized_results=normalized_results,
                cumulative_usage=cumulative_usage,
                stop_reason=finish_reason,
                trajectory=traj,
                summarizer_usage=sum_usage,
                wandb_logger=wandb_logger,
                searcher_type=args.searcher_type,
            )

        except Exception as exc:
            print(f"[Error] Query id={qid} failed: {exc}")
            sys.exit(1)

    if args.num_threads <= 1:
        with tqdm(remaining, desc="Queries", unit="query") as pbar:
            for qid, qtext in pbar:
                _handle_single_query(qid, qtext, pbar)
    else:
        with (
            ThreadPoolExecutor(max_workers=args.num_threads) as executor,
            tqdm(total=len(remaining), desc="Queries", unit="query") as pbar,
        ):
            futures = [
                executor.submit(_handle_single_query, qid, qtext, pbar)
                for qid, qtext in remaining
            ]

            for _ in as_completed(futures):
                pbar.update(1)

    # Finalize W&B logging
    if wandb_logger:
        wandb_logger.finish()


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Call AWS Bedrock Runtime API with Claude via boto3 for native tool calling and local search."
    )
    parser.add_argument(
        "--query",
        default="topics-qrels/queries.tsv",
        help="User query text or path to TSV. Wrap in quotes if contains spaces.",
    )
    parser.add_argument(
        "--model",
        default="anthropic.claude-3-haiku-20240307-v1:0",
        help="Model ID for Bedrock (default: %(default)s)",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region for Bedrock (default: %(default)s)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10000,
        help="Maximum tokens to generate (default: %(default)s)",
    )
    parser.add_argument("--system", default=None, help="Optional system prompt")

    parser.add_argument(
        "--output-dir",
        default="runs/bm25/bedrock",
        help="Directory to save request/response logs (default: %(default)s)",
    )
    parser.add_argument(
        "--query-template",
        choices=[
            "QUERY_TEMPLATE",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION",
            "QUERY_TEMPLATE_WITH_COMPACT",
            "QUERY_TEMPLATE_WITH_COMPACT_AND_GET_DOCUMENT",
        ],
        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
        help="Specify the query template to use (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for the model (default: use model defaults)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top P for the model (default: use model defaults)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of parallel threads for dataset processing (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of conversation rounds with function calls (default: %(default)s)",
    )

    # Searcher configuration
    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )
    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Number of tokens to include for each document snippet in search results using Qwen/Qwen3-0.6B tokenizer (default: 512).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Fixed number of search results to return for all queries in this session (default: 5).",
    )
    parser.add_argument(
        "--get-document",
        action="store_true",
        help="If set, register both the search tool and the get_document tool.",
    )
    parser.add_argument(
        "--compact-model",
        type=str,
        default=None,
        help="Model to use for the compact (summarization) tool. Defaults to the same as --model.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token for accessing private datasets/models.",
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        help="Hugging Face home directory for caching models and datasets.",
    )

    # W&B logging configuration
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="browsecomp-evaluation",
        help="W&B project name (default: %(default)s)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity/team name (optional)",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="+",
        default=None,
        help="W&B tags for the run (space-separated)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )

    # Dynamic searcher args
    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()

    if args.hf_token:
        print(f"[DEBUG] Setting HF token from CLI argument: {args.hf_token[:10]}...")
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token

    if args.hf_home:
        print(f"[DEBUG] Setting HF home from CLI argument: {args.hf_home}")
        os.environ["HF_HOME"] = args.hf_home

    # AWS credentials should be configured via environment variables or AWS CLI
    # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN (optional)
    
    # Create a Bedrock Runtime client
    client = boto3.client("bedrock-runtime", region_name=args.region)

    # Initialize W&B logger
    wandb_logger = None
    if not args.no_wandb:
        tags = args.wandb_tags or []
        # Create logs directory for this run
        log_dir = os.path.join(args.output_dir, "wandb_logs")
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            tags=tags,
            enabled=True,
            log_dir=log_dir,
        )
        if wandb_logger.enabled:
            # Initialize run with experiment config
            config = {
                "model": args.model,
                "region": args.region,
                "searcher_type": args.searcher_type,
                "max_tokens": args.max_tokens,
                "k": args.k,
                "snippet_max_tokens": args.snippet_max_tokens,
                "query_template": args.query_template,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_iterations": args.max_iterations,
                "get_document": args.get_document,
                "compact_model": args.compact_model,
            }
            wandb_logger.init_run(config)
            wandb_logger.log_message(f"Starting evaluation run with {args.model} on {args.searcher_type}")

    # Initialize searcher
    searcher = searcher_class(args)

    # Determine if compact tool should be enabled based on query template
    enable_compact = args.query_template in (
        "QUERY_TEMPLATE_WITH_COMPACT",
        "QUERY_TEMPLATE_WITH_COMPACT_AND_GET_DOCUMENT",
    ) or args.compact_model is not None

    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document,
        include_compact=enable_compact,
    )

    tools_registered = ["search"]
    if args.get_document:
        tools_registered.append("get_document")
    if enable_compact:
        tools_registered.append("compact")
    tools_str = ", ".join(tools_registered)

    print(
        f"Search agent started with {searcher.search_type} search (snippet_max_tokens={args.snippet_max_tokens}, k={args.k})"
    )
    print(f"Registered tools: {tools_str}")

    # If --query looks like a TSV path, process dataset
    if isinstance(args.query, str):
        qstr = args.query.strip()
        if qstr.lower().endswith(".tsv"):
            potential_path = Path(qstr)
            try:
                if potential_path.is_file():
                    _process_tsv_dataset(str(potential_path), client, args, tool_handler, wandb_logger)
                    return
            except OSError:
                pass

    print("Sending request to AWS Bedrock with native tool calling...")
    normalized_results, cumulative_usage, finish_reason, trajectory, sum_usage = run_conversation_with_tools(
        client,
        query=args.query,
        model_id=args.model,
        max_tokens=args.max_tokens,
        tool_handler=tool_handler,
        system_prompt=args.system,
        query_template=args.query_template,
        temperature=args.temperature,
        top_p=args.top_p,
        max_iterations=args.max_iterations,
        compact_model=args.compact_model,
    )

    _persist_response(
        args.output_dir,
        model=args.model,
        query_id=None,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        normalized_results=normalized_results,
        cumulative_usage=cumulative_usage,
        stop_reason=finish_reason,
        trajectory=trajectory,
        summarizer_usage=sum_usage,
        wandb_logger=wandb_logger,
        searcher_type=args.searcher_type,
    )

    # Finalize W&B logging
    if wandb_logger:
        wandb_logger.finish()

    # Print final output text if present
    final_texts = [item["output"] for item in normalized_results if item.get("type") == "output_text"]
    if final_texts:
        rprint(final_texts[-1])


if __name__ == "__main__":
    main()

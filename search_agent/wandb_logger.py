"""W&B logging module for BrowseComp-Plus evaluation runs."""

import os
import json
import threading
from typing import Any, Dict, List, Optional
import wandb
from datetime import datetime


class WandbLogger:
    """Logger for tracking evaluation runs in Weights & Biases."""

    def __init__(
        self,
        project: str = "browsecomp-evaluation",
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        enabled: bool = True,
        log_dir: Optional[str] = None,
    ):
        self.enabled = enabled
        self.run = None
        self.table = None
        self.table_rows: List[Dict[str, Any]] = []
        self.instance_count = 0
        self.log_dir = log_dir
        self.log_buffer: List[str] = []
        self.log_file_path = None

        # Aggregate totals
        self.total_search_calls = 0
        self.total_compact_calls = 0

        # Single lock for all shared state
        self._lock = threading.Lock()

        if not self.enabled:
            return

        try:
            self.run = wandb.init(
                project=project,
                entity=entity,
                tags=tags or [],
                reinit=True,
            )

            self.table = wandb.Table(
                columns=[
                    "query_id",
                    "model",
                    "status",
                    "search_calls",
                    "compact_calls",
                    "base_input_tokens",
                    "base_output_tokens",
                    "base_cached_tokens",
                    "base_reasoning_tokens",
                    "base_total_tokens",
                    "summarizer_input_tokens",
                    "summarizer_output_tokens",
                    "summarizer_total_tokens",
                    "summarizer_num_calls",
                    "json_file",
                ]
            )

            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
                self.log_file_path = os.path.join(
                    self.log_dir,
                    f"run_log_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.txt",
                )

        except Exception as e:
            print(f"[Warning] Failed to initialize W&B: {e}. Continuing without W&B logging.")
            self.enabled = False
            self.run = None
            self.table = None

    def init_run(self, config: Dict[str, Any]) -> None:
        """Initialize W&B run with experiment configuration."""
        if not self.enabled or not self.run:
            return
        try:
            wandb.config.update(config)
        except Exception as e:
            print(f"[Warning] Failed to update W&B config: {e}")

    def log_message(self, message: str) -> None:
        """Log a message to the run log buffer and log file."""
        if not self.enabled:
            return
        self.log_buffer.append(f"[{datetime.utcnow().isoformat()}] {message}")
        if self.log_file_path:
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(f"{message}\n")
            except Exception:
                pass

    def log_instance(
        self,
        query_id: Optional[str],
        model: str,
        searcher_type: str,
        tool_call_counts: Dict[str, int],
        usage: Dict[str, Any],
        summarizer_usage: Dict[str, Any],
        status: str,
        retrieved_docids: List[str],
        trajectory: Optional[List[Dict[str, Any]]] = None,
        json_file: Optional[str] = None,
        full_instance_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single instance to W&B."""
        if not self.enabled or not self.run:
            return

        try:
            # --- Tool calls ---
            search_calls = tool_call_counts.get("search", 0)
            compact_calls = tool_call_counts.get("compact", 0)

            # --- Base model token usage ---
            base_input = usage.get("input_tokens", 0)
            base_output = usage.get("output_tokens", 0)
            base_cached = usage.get("input_tokens_cached", 0)
            base_reasoning = usage.get("included_reasoning_tokens", 0) or 0
            base_total = usage.get("total_tokens", 0)

            # --- Summarizer token usage ---
            sum_input = summarizer_usage.get("input_tokens", 0)
            sum_output = summarizer_usage.get("output_tokens", 0)
            sum_total = summarizer_usage.get("total_tokens", 0)
            sum_calls = summarizer_usage.get("num_calls", 0)

            # Thread-safe update of shared state
            with self._lock:
                self.instance_count += 1
                self.total_search_calls += search_calls
                self.total_compact_calls += compact_calls
                cur_total_search = self.total_search_calls
                cur_total_compact = self.total_compact_calls

                self.table_rows.append({
                    "query_id": str(query_id) if query_id else f"instance_{self.instance_count}",
                    "model": model,
                    "status": status,
                    "search_calls": search_calls,
                    "compact_calls": compact_calls,
                    "base_input_tokens": base_input,
                    "base_output_tokens": base_output,
                    "base_cached_tokens": base_cached,
                    "base_reasoning_tokens": base_reasoning,
                    "base_total_tokens": base_total,
                    "summarizer_input_tokens": sum_input,
                    "summarizer_output_tokens": sum_output,
                    "summarizer_total_tokens": sum_total,
                    "summarizer_num_calls": sum_calls,
                    "json_file": json_file or "",
                })

            # ---- Single wandb.log() call -- exactly the charts you want ----
            wandb.log({
                # Per-instance tool calls (bar/line chart per instance)
                "search_calls": search_calls,
                "compact_calls": compact_calls,

                # Running totals (cumulative line chart)
                "total_search_calls": cur_total_search,
                "total_compact_calls": cur_total_compact,

                # Base model token usage per instance
                "base_model/input_tokens": base_input,
                "base_model/output_tokens": base_output,
                "base_model/input_tokens_cached": base_cached,
                "base_model/included_reasoning_tokens": base_reasoning,
                "base_model/total_tokens": base_total,

                # Summarizer usage per instance
                "summarizer/input_tokens": sum_input,
                "summarizer/output_tokens": sum_output,
                "summarizer/total_tokens": sum_total,
                "summarizer/num_calls": sum_calls,
            })

            # Store instance data as artifact
            if full_instance_data:
                try:
                    artifact_name = f"instance_{query_id or self.instance_count}"
                    artifact = wandb.Artifact(
                        name=artifact_name,
                        type="instance_data",
                        description=f"Full data for query {query_id}",
                    )
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                        json.dump(full_instance_data, tmp, indent=2, default=str)
                        tmp_path = tmp.name
                    artifact.add_file(tmp_path, name=f"{artifact_name}.json")
                    wandb.log_artifact(artifact)
                    os.unlink(tmp_path)
                except Exception as e:
                    print(f"[Warning] Failed to create artifact for {query_id}: {e}")

            self.log_message(
                f"Completed instance {query_id}: {search_calls} searches, {compact_calls} compacts, status={status}"
            )

        except Exception as e:
            print(f"[Warning] Failed to log instance {query_id} to W&B: {e}")

    def add_to_table(self, row_data: Dict[str, Any]) -> None:
        """Add a row to the W&B Table."""
        if not self.enabled or not self.table:
            return
        with self._lock:
            self.table_rows.append(row_data)

    def log_summary(self, aggregate_metrics: Dict[str, Any]) -> None:
        """Log aggregate/summary metrics for the entire run."""
        if not self.enabled or not self.run:
            return
        try:
            for key, value in aggregate_metrics.items():
                wandb.run.summary[key] = value
        except Exception as e:
            print(f"[Warning] Failed to log summary metrics: {e}")

    def finish(self) -> None:
        """Finalize W&B run and upload all data."""
        if not self.enabled or not self.run:
            return

        try:
            # Build and log the table
            if self.table and self.table_rows:
                for row in self.table_rows:
                    self.table.add_data(
                        row.get("query_id", ""),
                        row.get("model", ""),
                        row.get("status", ""),
                        row.get("search_calls", 0),
                        row.get("compact_calls", 0),
                        row.get("base_input_tokens", 0),
                        row.get("base_output_tokens", 0),
                        row.get("base_cached_tokens", 0),
                        row.get("base_reasoning_tokens", 0),
                        row.get("base_total_tokens", 0),
                        row.get("summarizer_input_tokens", 0),
                        row.get("summarizer_output_tokens", 0),
                        row.get("summarizer_total_tokens", 0),
                        row.get("summarizer_num_calls", 0),
                        row.get("json_file", ""),
                    )
                wandb.log({"evaluation_table": self.table})

            # Final summary in Overview tab
            if self.instance_count > 0:
                wandb.run.summary["total_instances"] = self.instance_count
                wandb.run.summary["total_search_calls"] = self.total_search_calls
                wandb.run.summary["total_compact_calls"] = self.total_compact_calls

            # Upload run log as artifact
            if self.log_file_path and os.path.exists(self.log_file_path):
                try:
                    log_artifact = wandb.Artifact(
                        name="run_logs",
                        type="logs",
                        description="Complete run logs for this evaluation",
                    )
                    log_artifact.add_file(self.log_file_path)
                    wandb.log_artifact(log_artifact)
                    print(f"[W&B] Uploaded run logs to artifacts")
                except Exception as e:
                    print(f"[Warning] Failed to upload log artifact: {e}")

            wandb.finish()
            print(f"[W&B] Logged {self.instance_count} instances to W&B")
            print(f"[W&B] Total search calls: {self.total_search_calls}, Total compact calls: {self.total_compact_calls}")
        except Exception as e:
            print(f"[Warning] Failed to finish W&B run: {e}")
            try:
                wandb.finish()
            except Exception:
                pass

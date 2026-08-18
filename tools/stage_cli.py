"""Shared framework-only stage command."""

from __future__ import annotations

import argparse
import json
import os

from _bootstrap import PROJECT_ROOT  # noqa: F401

from dfc_sam.engine.run_plan import inspect_stage
from dfc_sam.engine.stage_runner import run_formal_stage
from dfc_sam.engine.stages import Stage


def run_stage(stage: Stage) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--all-samples")
    parser.add_argument("--split-manifest")
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metrics-repo")
    parser.add_argument("--resume")
    parser.add_argument("--resume-source-git-commit", action="append", default=[])
    parser.add_argument("--allow-world-size-transition", action="store_true")
    parser.add_argument("--reset-early-stopping-on-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.execute:
        raise SystemExit("Choose exactly one of --dry-run or --execute")
    plan = inspect_stage(args.config, stage)
    is_primary = int(os.environ.get("RANK", "0")) == 0
    if is_primary:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    required = {
        "--all-samples": args.all_samples,
        "--split-manifest": args.split_manifest,
        "--output-root": args.output_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"Formal execution is missing arguments: {missing}")
    result = run_formal_stage(
        config_path=args.config,
        stage=stage,
        all_samples=args.all_samples,
        split_manifest=args.split_manifest,
        output_root=args.output_root,
        device_name=args.device,
        official_metrics_repository=args.metrics_repo,
        resume=args.resume,
        resume_source_git_commits=tuple(args.resume_source_git_commit),
        allow_world_size_transition=args.allow_world_size_transition,
        reset_early_stopping_on_resume=args.reset_early_stopping_on_resume,
    )
    if is_primary:
        print(json.dumps(result, ensure_ascii=False, indent=2))

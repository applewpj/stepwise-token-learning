from __future__ import annotations

import argparse
from pathlib import Path

from tokenlearning.hub import DEFAULT_REPO_ID, MANIFEST_RELPATH

EPILOG = """examples:
  tokenlearning reproduce synthetic           # appendix figure, about a minute on CPU
  tokenlearning reproduce fig3 --quick        # Figure 3 from a 50k-token subset
  tokenlearning reproduce all                 # Figures 3-5 at the paper setting (GPU recommended)
  tokenlearning download --list               # released loss matrices and their sizes
  tokenlearning run --config configs/step_axis_1m.yaml --device cpu --max-tokens 100000
"""


def _add_override_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("overrides")
    group.add_argument("--device", default=None, help="auto, cpu, or cuda. The default (auto) uses CUDA when available.")
    group.add_argument("--max-tokens", type=int, default=None, help="Fit a random subset of this many tokens.")
    group.add_argument("--epochs", type=int, default=None, help="Optimizer steps for the per-token sigmoid fit.")
    group.add_argument("--seed", type=int, default=None, help="Seed for the random token subset (default 0).")
    group.add_argument("--output-dir", default=None, help="Where to write metrics, caches, and figures.")


def _overrides(args: argparse.Namespace) -> dict:
    return {
        "device": args.device,
        "epochs": args.epochs,
        "max_tokens": args.max_tokens,
        "subset_seed": args.seed,
        "output_dir": args.output_dir,
    }


def _run(args: argparse.Namespace) -> None:
    from tokenlearning.config import apply_overrides, load_config
    from tokenlearning.pipeline import run_analysis

    config = load_config(args.config)
    config["_config_path"] = str(Path(args.config))
    metrics = run_analysis(apply_overrides(config, **_overrides(args)))
    print(f"wrote results to {metrics['output_dir']}")


def _download(args: argparse.Namespace) -> None:
    from tokenlearning.hub import ensure_example, find_repo_root, load_manifest, manifest_entries
    from tokenlearning.reproduce import ALIASES, TARGETS

    root = find_repo_root()
    manifest_path = Path(args.manifest) if args.manifest else root / MANIFEST_RELPATH
    entries = manifest_entries(load_manifest(manifest_path))
    figure_of = {target.asset: target.title.split(",")[0] for target in TARGETS.values()}

    if args.list:
        print(f"{'id':<40} {'size':>9}  {'tokens x points':<16} {'used for':<10} status")
        for entry_id, entry in entries.items():
            rows, points = entry["loss_matrix_shape"]
            present = (root / str(entry["local_path"])).exists()
            print(
                f"{entry_id:<40} {float(entry['size_bytes']) / 1e6:>6,.1f} MB  {f'{rows:,} x {points}':<16} "
                f"{figure_of.get(entry_id, ''):<10} {'downloaded' if present else '-'}"
            )
        return

    names = list(entries) if args.all else list(args.names) + list(args.name or [])
    if not names:
        raise SystemExit("Name at least one example (see `tokenlearning download --list`) or pass --all.")
    for name in names:
        target = ALIASES.get(name.lower(), name.lower())
        entry_id = TARGETS[target].asset if target in TARGETS else name
        if entry_id not in entries:
            raise SystemExit(f"Unknown example {name!r}. Available: {', '.join(entries)}")
        ensure_example(entries[entry_id], root, repo_id=args.repo_id, revision=args.revision, force=args.force)


def _reproduce(args: argparse.Namespace) -> None:
    from tokenlearning.reproduce import describe_targets, reproduce

    if args.list or not args.targets:
        print(describe_targets(), flush=True)
        if not args.list:
            raise SystemExit("\nName a target, e.g. `tokenlearning reproduce synthetic` or `tokenlearning reproduce fig3 --quick`.")
        return
    results = reproduce(
        args.targets,
        quick=args.quick,
        repo_id=args.repo_id,
        revision=args.revision,
        dry_run=args.dry_run,
        **_overrides(args),
    )
    if any(not check["ok"] for item in results for check in item["checks"]):
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenlearning",
        description="Stepwise Token Learning: reproduce the paper figures or analyze your own token-level loss matrix.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="{reproduce,download,run}")

    repro = subparsers.add_parser(
        "reproduce",
        help="Download the released data and regenerate a paper figure.",
        description="Download the released loss matrix (checksum verified), fit it, and write the figures.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    repro.add_argument("targets", nargs="*", metavar="TARGET", help="fig3, fig4, fig5, synthetic, or all.")
    repro.add_argument("--quick", action="store_true", help="Fit a random 50,000-token subset for 2,000 epochs instead of the paper setting.")
    repro.add_argument("--list", action="store_true", help="List targets and exit.")
    repro.add_argument("--dry-run", action="store_true", help="Print what would run without downloading or fitting.")
    repro.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"HuggingFace dataset repo (default {DEFAULT_REPO_ID}).")
    repro.add_argument("--revision", default=None, help="Dataset revision (branch, tag, or commit).")
    _add_override_args(repro)

    download = subparsers.add_parser("download", help="Download released loss matrices from HuggingFace.")
    download.add_argument("names", nargs="*", metavar="NAME", help="Example id or figure target, e.g. step_axis_1m or fig3.")
    download.add_argument("--name", action="append", default=None, help=argparse.SUPPRESS)
    download.add_argument("--all", action="store_true", help="Download every released matrix (about 600 MB).")
    download.add_argument("--list", action="store_true", help="List released matrices and exit.")
    download.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"HuggingFace dataset repo (default {DEFAULT_REPO_ID}).")
    download.add_argument("--manifest", default=None, help="Manifest file (default examples/manifest.yaml).")
    download.add_argument("--revision", default=None, help="Dataset revision (branch, tag, or commit).")
    download.add_argument("--force", action="store_true", help="Download again even if a verified copy exists.")

    run = subparsers.add_parser("run", help="Run the analysis for a YAML config.")
    run.add_argument("--config", required=True, help="YAML analysis config.")
    _add_override_args(run)
    return parser


def run_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tokenlearning-run", description="Run token learning-time spectrum analysis.")
    parser.add_argument("--config", required=True, help="YAML analysis config.")
    _add_override_args(parser)
    _run(parser.parse_args(argv))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        _run(args)
    elif args.command == "download":
        _download(args)
    elif args.command == "reproduce":
        _reproduce(args)


if __name__ == "__main__":
    main()

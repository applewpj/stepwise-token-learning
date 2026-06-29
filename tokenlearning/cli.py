from __future__ import annotations

import argparse

from tokenlearning.pipeline import run_analysis


def run_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run token learning-time spectrum analysis.")
    parser.add_argument("--config", required=True, help="YAML analysis config.")
    args = parser.parse_args(argv)
    metrics = run_analysis(args.config)
    print(f"wrote results to {metrics['output_dir']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tokenlearning")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run analysis from YAML config.")
    run_parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        run_main(["--config", args.config])


if __name__ == "__main__":
    main()

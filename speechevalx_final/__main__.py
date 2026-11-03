"""Command-line interface for dataset conversion, validation and evaluation."""
import argparse
import json
from pathlib import Path
from speechevalx.runner import run


def main():
    parser = argparse.ArgumentParser(description="Paired ASR-to-action evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    converter = commands.add_parser("convert", help="Convert existing local benchmark JSON exports")
    converter.add_argument("--format", choices=["voiceagentbench", "bfcl_audio"], required=True)
    converter.add_argument("--input", type=Path, required=True)
    converter.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "convert":
            from speechevalx.datasets.convert import convert_dataset
            result = convert_dataset(args.input, args.output, args.format)
        else:
            result = run(args.config, validate_only=args.command == "validate")
    except Exception as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    if args.command != "run":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("Results: " + result["output"])
    print("Instrumentation failures: " + str(result["instrumentation_failures"]))
    if result.get("diagnostics"):
        print("Diagnostics: " + result["diagnostics"]["status"])
    failed = result["instrumentation_failures"] > 0 or (result.get("diagnostics") or {}).get("status") == "failed"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI entry point for subby: python -m subby"""

import argparse
import json
import sys

from .io import load_json, run, format_output


def main():
    parser = argparse.ArgumentParser(
        prog="subby",
        description="Compute phylogenetic sufficient statistics from a JSON specification.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input JSON file (reads from stdin if omitted)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file (writes to stdout if omitted)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON output indentation (default: 2)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate input JSON against schema without running",
    )

    args = parser.parse_args()

    # Load input
    if args.input:
        data = load_json(args.input)
    else:
        data = json.load(sys.stdin)

    # Validate-only mode
    if args.validate:
        try:
            from .io import load_input
            load_input(data)
            print("Valid.", file=sys.stderr)
            sys.exit(0)
        except Exception as e:
            print(f"Invalid: {e}", file=sys.stderr)
            sys.exit(1)

    # Run
    result = run(data)

    # Output
    output_str = format_output(result, indent=args.indent)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
            f.write("\n")
    else:
        print(output_str)


if __name__ == "__main__":
    main()

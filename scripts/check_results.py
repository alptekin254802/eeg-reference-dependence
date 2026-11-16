"""Check consistency of the released aggregate statistics and figure inputs."""
import argparse
import json
from pathlib import Path

def resolve(document, pointer):
    value = document
    for token in pointer.lstrip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    aggregate = json.loads((args.results / "aggregate.json").read_text(encoding="utf-8"))
    figures = json.loads((args.results / "figure_values.json").read_text(encoding="utf-8"))
    for row in figures["values"]:
        if resolve(aggregate, row["pointer"]) != row["value"]:
            raise ValueError("Figure value mismatch: " + row["pointer"])
    print(f"{len(figures['values'])} figure values match aggregate statistics.")

if __name__ == "__main__":
    main()

"""Export recorded aggregate statistics and supplementary tables to CSV and Markdown."""
import argparse
import csv
import json
from pathlib import Path

def flatten(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, path + "/" + key.replace("~", "~0").replace("/", "~1"))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from flatten(child, path + "/" + str(i))
    else:
        yield path, value

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--output", type=Path, default=Path("outputs/tables"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    aggregate = json.loads((a.results / "aggregate.json").read_text(encoding="utf-8"))
    with (a.output / "aggregate_statistics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["result_pointer", "value"])
        w.writerows(flatten(aggregate))
    tables = json.loads((a.results / "tables.json").read_text(encoding="utf-8"))["tables"]
    sections = []
    for table in tables:
        columns, rows = table["columns"], table["rows"]
        if any(len(row) != len(columns) for row in rows):
            raise ValueError("Inconsistent column count: " + table["id"])
        with (a.output / (table["id"] + ".csv")).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(columns); w.writerows(rows)
        sections.append("## " + table["caption"] + "\n\n")
        for row in [columns, ["---"] * len(columns)] + rows:
            sections.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |\n")
        sections.append("\n")
    (a.output / "tables.md").write_text("".join(sections), encoding="utf-8")
    print(f"Exported {len(tables)} tables and aggregate_statistics.csv to {a.output}")

if __name__ == "__main__":
    main()

"""Read the dataset, count categories, and display one labelled example.

Run: python3 explore_dataset.py
Try: python3 explore_dataset.py --category SyntaxError
This script only reads the CSV. It does not clean, split, or train on it.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path


DATASET = Path(__file__).resolve().parent.parent / "data" / "raw" / "sui_move_dataset_v1.csv"
CATEGORIES = ("Perfect", "SecurityError", "SemanticError", "StyleError", "SyntaxError")


def main_category(label):
    """Count subcategories together; preserve the original label in the CSV."""
    category = label.split("/", 1)[0].strip()
    return "SyntaxError" if category == "Syntax Error" else category


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=CATEGORIES, default="SemanticError")
    args = parser.parse_args()

    # DictReader turns each CSV record into a dictionary keyed by column name.
    # It correctly handles code containing commas and multiple lines.
    with DATASET.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)

    print(f"Dataset: {DATASET.name}")
    print(f"Records: {len(rows):,}")
    print(f"Columns: {', '.join(columns)}")

    counts = Counter(main_category(row["Label"]) for row in rows)
    print("\nMain categories (subcategories counted together):")
    for category, count in sorted(counts.items()):
        print(f"  {category:15} {count:5,}")

    print("\nEmpty or whitespace-only fields:")
    for column in columns:
        missing = sum(not row[column].strip() for row in rows)
        print(f"  {column}: {missing}")

    unique_code = len({row["Code"] for row in rows})
    print(f"\nDistinct code strings, exact comparison: {unique_code:,}")
    print(f"Repeated code occurrences beyond the first: {len(rows) - unique_code}")

    # A rough text diagnostic, not a cleaning rule. It also changes whitespace
    # inside comments and string literals, so it cannot establish equivalent code.
    collapsed_code = {" ".join(row["Code"].split()) for row in rows}
    print(f"Distinct strings after collapsing whitespace: {len(collapsed_code):,}")
    print("Whitespace count is diagnostic only; formatting may be an issue label.")

    # Show the first matching example, not a representative or random sample.
    record_number, example = next(
        (number, row)
        for number, row in enumerate(rows, start=1)
        if main_category(row["Label"]) == args.category
    )
    print(f"\nFirst {args.category} example, record {record_number}:")
    print(f"Original label: {example['Label']}")
    print(f"\nCode:\n{example['Code']}")
    print(f"\nOutput/Fix:\n{example['Output/Fix']}")


if __name__ == "__main__":
    main()

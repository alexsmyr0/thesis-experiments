"""Combine the existing v2 module and newline rules without removing comments.

Start from only_modules.csv to retain its exact randomized names and wrappers.
Use the existing newline transform and exemptions, with no new row filtering.
Run after build_neutral_datasets.py if regenerating the original v2 variants.
"""

import hashlib
import json

import pandas as pd

from build_neutral_datasets import KEEP_WHITESPACE, OUT, flatten


def main():
    control_path = OUT / "control_unmodified.csv"
    modules_path = OUT / "only_modules.csv"
    control = pd.read_csv(control_path, encoding="utf-8-sig", keep_default_na=False)
    modules = pd.read_csv(modules_path, encoding="utf-8-sig", keep_default_na=False)
    pd.testing.assert_frame_equal(control.drop(columns="Code"), modules.drop(columns="Code"))

    sublabels = control["Label"].str.split("/", n=1).str[1].fillna("").str.strip()
    exempt = sublabels.isin(KEEP_WHITESPACE)
    combined = modules.copy()
    combined.loc[~exempt, "Code"] = modules.loc[~exempt, "Code"].map(flatten)

    # Every non-whitespace character survives, including comments and module names.
    compact = lambda code: "".join(code.split())
    assert combined["Code"].map(compact).equals(modules["Code"].map(compact))
    assert combined.loc[exempt, "Code"].equals(modules.loc[exempt, "Code"])
    assert combined["Code"].str.strip().ne("").all()
    # prepare_data.py uses this key for deduplication: no further row loss is allowed.
    fingerprints = combined["Code"].map(lambda code: " ".join(code.split()))
    assert not fingerprints.duplicated().any()

    destination = OUT / "modules_newlines.csv"
    combined.to_csv(destination, index=False, encoding="utf-8-sig")
    report = {
        "rows": len(combined),
        "rows_changed_from_only_modules": int(combined["Code"].ne(modules["Code"]).sum()),
        "whitespace_exempt_rows": int(exempt.sum()),
        "rows_with_newlines": int(combined["Code"].str.contains("\n").sum()),
        "comments_removed": False,
        "module_names_reused_from": modules_path.name,
        "row_order_labels_outputs_match_control": True,
        "non_whitespace_characters_match_only_modules": True,
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (control_path, modules_path, destination)},
    }
    (OUT / "modules_newlines_build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

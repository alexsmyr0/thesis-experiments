"""Run paired module/newline ablations, with separate output for every seed."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATASETS = ("control_unmodified", "only_modules", "only_newlines", "modules_newlines")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="validate splits and run SVC only")
    args = parser.parse_args()
    assert len(set(args.seeds)) == len(args.seeds)
    assert all(0 <= seed < 2**32 for seed in args.seeds)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Refuse accidental reuse of an earlier run's destination.
    assert not any((out / f"seed_{seed}").exists() for seed in args.seeds)
    frames = {name: pd.read_csv(REPO / "data/v2" / f"{name}.csv", keep_default_na=False)
              for name in DATASETS}
    control = frames[DATASETS[0]]
    for frame in frames.values():
        pd.testing.assert_frame_equal(control.drop(columns="Code"), frame.drop(columns="Code"))
    dataset_hashes = {name: hashlib.sha256((REPO / "data/v2" / f"{name}.csv").read_bytes()).hexdigest()
                      for name in DATASETS}
    (out / "sweep_plan.json").write_text(json.dumps({
        "seeds": args.seeds, "datasets": DATASETS, "dataset_sha256": dataset_hashes,
        "split_seed_equals_training_seed": True, "quick": args.quick,
        "fp16": False, "models": ["tfidf_svm", "tfidf_svm_balanced"] +
        ([] if args.quick else ["codebert_lora"]),
    }, indent=2) + "\n")

    for seed in args.seeds:
        reference = None
        environments = {}
        # Verify actual row identities across all variants BEFORE any model training.
        for name in DATASETS:
            destination = out / f"seed_{seed}" / name
            env = {**os.environ, "THESIS_SEED": str(seed), "THESIS_OUT": str(destination),
                   "THESIS_CSV": str(REPO / "data/v2" / f"{name}.csv"),
                   "PY": sys.executable, "CODEBERT_ARGS": ""}
            environments[name] = env
            subprocess.run([sys.executable, str(REPO / "src/prepare_data.py")], env=env, check=True)
            splits = json.loads((destination / "split_manifest.json").read_text())["splits"]
            all_rows = [row for rows in splits.values() for row in rows]
            assert sorted(all_rows) == list(range(len(control))), "Missing or repeated source rows"
            if reference is None:
                reference = splits
            assert splits == reference, f"Unpaired split: seed={seed}, dataset={name}"
        print(f"Paired split verified: seed={seed}, rows={len(control)}", flush=True)
        for name in DATASETS:
            print(f"\n######## seed={seed} dataset={name} ########", flush=True)
            cmd = ["sh", str(REPO / "scripts/run_all.sh"), "--codebert=lora", "--tfidf=svm"]
            if args.quick:
                cmd.append("--quick")
            subprocess.run(cmd, env=environments[name], check=True)
            shutil.rmtree(out / f"seed_{seed}" / name / "checkpoints", ignore_errors=True)


if __name__ == "__main__":
    main()

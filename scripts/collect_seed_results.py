"""Import Kaggle archives without overwriting results, then summarize paired seeds.

Use --legacy-seed 42 for the two original archives whose metrics predate explicit seeds.
Metrics/logs remain byte-for-byte copies. Provenance records the archive and source run.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import statistics
import zipfile

REPO = Path(__file__).resolve().parent.parent
DATASETS = ("control_unmodified", "only_modules", "only_newlines", "modules_newlines")
MODELS = ("codebert_lora", "tfidf_svm", "tfidf_svm_balanced")


def write_once(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite different results: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def import_archive(archive, destination, legacy_seed):
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as z:
        roots = {PurePosixPath(n).parts[0] for n in z.namelist()}
        assert len(roots) == 1, "Expected one timestamped run folder"
        root = roots.pop()
        metadata = json.loads(z.read(f"{root}/run_metadata.json"))
        leaves = {str(PurePosixPath(n).parent.parent) for n in z.namelist()
                  if n.endswith("_metrics.json")}
        for leaf in sorted(leaves):
            parts = PurePosixPath(leaf).parts
            dataset = parts[-1]
            assert dataset in DATASETS, f"Unexpected dataset: {dataset}"
            seed = int(parts[-2][5:]) if parts[-2].startswith("seed_") else legacy_seed
            assert seed is not None, "Old archives require an explicit --legacy-seed"
            for model in MODELS:
                metric = json.loads(z.read(f"{leaf}/results/{model}_metrics.json"))
                assert metric.get("split_seed", seed) == seed
                assert metric.get("training_seed", seed) == seed
            target = destination / f"seed_{seed}" / dataset
            # Preflight paths and conflicts before writing this leaf.
            files = {}
            for name in z.namelist():
                if not name.startswith(leaf + "/") or name.endswith("/"):
                    continue
                relative = PurePosixPath(name[len(leaf) + 1:])
                assert not relative.is_absolute() and ".." not in relative.parts
                assert relative.parts[0] in {"data", "logs", "results", "split_manifest.json"}
                files[Path(*relative.parts)] = z.read(name)
            source = REPO / "data/v2" / f"{dataset}.csv"
            if Path("split_manifest.json") not in files:
                # Recover exact row identities from archived splits; do not rerun training.
                records = list(csv.DictReader(io.StringIO(source.read_bytes().decode("utf-8-sig"))))
                row_ids = {r["Code"].strip(): i for i, r in enumerate(records)}
                assert len(row_ids) == len(records)
                splits = {}
                for split in ("train", "val", "test"):
                    rows = csv.DictReader(io.StringIO(files[Path(f"data/{split}.csv")].decode()))
                    splits[split] = [row_ids[r["code"]] for r in rows]
                files[Path("split_manifest.json")] = (json.dumps({
                    "split_seed": seed, "source_csv": source.name,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "row_numbering": "zero-based data row in source CSV (header excluded)",
                    "recovered_from_archived_splits": True, "splits": splits,
                }, indent=2) + "\n").encode()
            manifest = json.loads(files[Path("split_manifest.json")])
            assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
            provenance = {"archive_sha256": archive_hash, "archive_run": root,
                          "dataset": dataset, "split_seed": seed, "training_seed": seed,
                          "seed_inferred_from_original_run_configuration": "seeds" not in metadata,
                          "run_metadata": metadata}
            files[Path("provenance.json")] = (json.dumps(provenance, indent=2) + "\n").encode()
            files[Path("pip_freeze.txt")] = z.read(f"{root}/pip_freeze.txt")
            for relative, content in files.items():
                if (target / relative).exists() and (target / relative).read_bytes() != content:
                    raise ValueError(f"Conflicting result: {target / relative}")
            for relative, content in files.items():
                write_once(target / relative, content)
            print(f"Imported {target}")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(destination):
    rows, manifests = [], {}
    for folder in sorted(destination.glob("seed_*")):
        seed = int(folder.name[5:])
        for dataset in DATASETS:
            leaf = folder / dataset
            if not leaf.exists():
                continue
            manifest = json.loads((leaf / "split_manifest.json").read_text())
            splits = manifest["splits"]
            assert sorted(r for split in splits.values() for r in split) == list(range(3133))
            if seed in manifests:
                assert splits == manifests[seed], f"Unpaired datasets for seed {seed}"
            manifests[seed] = splits
            for model in MODELS:
                m = json.loads((leaf / "results" / f"{model}_metrics.json").read_text())
                rows.append({"seed": seed, "dataset": dataset, "model": model,
                             "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    if not rows:
        return
    write_csv(destination / "metrics_by_seed.csv", rows)
    lookup = {(r["seed"], r["dataset"], r["model"]): r for r in rows}
    deltas = []
    for row in rows:
        if row["dataset"] == "control_unmodified":
            continue
        control = lookup.get((row["seed"], "control_unmodified", row["model"]))
        if control:
            deltas.append({"seed": row["seed"], "dataset": row["dataset"], "model": row["model"],
                           "accuracy_delta_pp": 100 * (row["accuracy"] - control["accuracy"]),
                           "macro_f1_delta_pp": 100 * (row["macro_f1"] - control["macro_f1"])})
    if deltas:
        write_csv(destination / "paired_deltas.csv", deltas)
    aggregate = []
    for dataset in DATASETS:
        for model in MODELS:
            matches = [r for r in rows if r["dataset"] == dataset and r["model"] == model]
            if not matches:
                continue
            item = {"dataset": dataset, "model": model, "n_seeds": len(matches)}
            for metric in ("accuracy", "macro_f1"):
                values = [r[metric] for r in matches]
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else ""
            aggregate.append(item)
    write_csv(destination / "aggregate.csv", aggregate)
    status = {"seeds_present": sorted(manifests), "expected_seeds": [7, 13, 42],
              "complete": all((s, d, m) in lookup for s in (7, 13, 42) for d in DATASETS for m in MODELS),
              "paired_split_checks": "passed", "metric_rows": len(rows),
              "interpretation": "Exploratory three-seed replication; SD is not a confidence interval."}
    (destination / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", type=Path, nargs="*")
    parser.add_argument("--legacy-seed", type=int)
    parser.add_argument("--out", type=Path, default=REPO / "experiments/module_newline_seeds")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for archive in args.archives:
        import_archive(archive, args.out, args.legacy_seed)
    summarize(args.out)


if __name__ == "__main__":
    main()

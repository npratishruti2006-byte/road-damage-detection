# Copies a training run's required files into a backup folder and verifies checksums.
import hashlib, json, shutil
from datetime import datetime
from pathlib import Path

REQUIRED = ["best.pt", "last.pt", "results.csv", "args.yaml"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def backup_experiment(run_dir, exp_name, config, out_root="experiments"):
    run_dir = Path(run_dir)
    out = Path(out_root) / exp_name
    out.mkdir(parents=True, exist_ok=True)

    missing = [f for f in REQUIRED if not (run_dir / f).exists() or (run_dir / f).stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Experiment '{exp_name}' is missing or empty: {missing}")

    for f in REQUIRED:
        shutil.copy(run_dir / f, out / f)
    for p in run_dir.glob("*.png"):
        shutil.copy(p, out / p.name)
    (out / "config.json").write_text(json.dumps(config, indent=2))

    manifest = {
        "experiment": exp_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "files": {f: {"bytes": (out / f).stat().st_size, "sha256": sha256(out / f)} for f in REQUIRED},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Backed up '{exp_name}' to {out}")
    return out


def verify_backup(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    ok = True
    for f, meta in manifest["files"].items():
        p = out / f
        good = p.exists() and p.stat().st_size == meta["bytes"] and sha256(p) == meta["sha256"]
        print(f"{'OK  ' if good else 'FAIL'} {f}")
        ok = ok and good
    return ok

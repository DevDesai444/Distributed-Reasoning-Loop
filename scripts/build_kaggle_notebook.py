#!/usr/bin/env python3
"""
Render `notebooks/kaggle_receipts.ipynb` from this script.

Rationale: the canonical artifact is committed JSON, but hand-editing
.ipynb JSON in a PR is unreadable. So we keep the source of truth here,
in Python, and regenerate the notebook with `nbformat.write`. Diffs in
review become readable patches against this file instead of opaque
notebook JSON.

Run with:

    python scripts/build_kaggle_notebook.py

It overwrites `notebooks/kaggle_receipts.ipynb`. Commit both files
together.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import nbformat


REPO_URL = "https://github.com/DevDesai444/Distributed-Reasoning-Loop.git"
NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "kaggle_receipts.ipynb"


# ---- cells ----------------------------------------------------------------

CELL_0_CLONE = """\
# Cell 0: clone the repo into Kaggle's writable working directory.
# Kaggle notebooks start with no source checkout; everything happens
# inside /kaggle/working which is the only writable location.
!cd /kaggle/working && \\
    (test -d Distributed-Reasoning-Loop || git clone {repo}) && \\
    cd Distributed-Reasoning-Loop && \\
    git fetch --quiet && \\
    git checkout main && \\
    git pull --quiet
%cd /kaggle/working/Distributed-Reasoning-Loop
""".format(repo=REPO_URL)


CELL_1_SETUP = """\
# Cell 1: install the package + dependencies, confirm GPU is visible,
# pin the HuggingFace cache to /kaggle/working so model weights persist
# across cells but not across notebook reruns.
!pip install -q -e .
import os, torch
os.environ["HF_HOME"] = "/kaggle/working/.hf_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1))
else:
    print("WARNING: no GPU detected. Full-mode runs require a T4.")
"""


CELL_2_SYNTHETIC = """\
# Cell 2: synthetic data generation.
# Target 30k unique DPO pairs, 12 sampled paths per problem, capped at 8
# pairs per problem so a few easy ones don't dominate the dataset.
# Expect ~2-3 hours on Kaggle T4.
!python scripts/collect_receipts.py --mode full --stage synthetic \\
    --output-dir /kaggle/working/receipts_run \\
    --generator-model Qwen/Qwen2.5-1.5B-Instruct \\
    --target-pairs 30000 \\
    --num-paths 12 \\
    --max-pairs-per-problem 8
"""


CELL_3_AGREEMENT = """\
# Cell 3: three-way agreement eval.
# 200 GSM8K test problems x 4 samples per problem. The judge is the
# 7B Qwen Instruct model; the policy is the 1.5B model. Expect ~1.5
# hours: judge inference dominates.
!python scripts/collect_receipts.py --mode full --stage agreement \\
    --output-dir /kaggle/working/receipts_run \\
    --generator-model Qwen/Qwen2.5-1.5B-Instruct \\
    --judge-model Qwen/Qwen2.5-7B-Instruct \\
    --n-problems 200 \\
    --n-samples 4
"""


CELL_4_ROBUSTNESS = """\
# Cell 4: robustness eval against six perturbations.
# Same problem slice as agreement, fixed seed=0 so reruns are
# reproducible. Expect ~1 hour: six perturbations x 200 problems x 4
# samples each, but most run sequentially through the same model.
!python scripts/collect_receipts.py --mode full --stage robustness \\
    --output-dir /kaggle/working/receipts_run \\
    --generator-model Qwen/Qwen2.5-1.5B-Instruct \\
    --n-problems 200 \\
    --n-samples 4 \\
    --seed 0
"""


CELL_5_VALIDATE = """\
# Cell 5: re-read receipts already on disk and assert each one passes
# schema validate(). No new runs. This is the gate that catches a
# malformed receipt before download.
!python scripts/collect_receipts.py --stage validate \\
    --output-dir /kaggle/working/receipts_run
"""


CELL_6_SUMMARY = """\
# Cell 6: human-readable summary so we can eyeball the numbers before
# downloading.
import json, pathlib
receipts_dir = pathlib.Path("/kaggle/working/receipts_run/receipts")
for p in sorted(receipts_dir.glob("*.json")):
    data = json.loads(p.read_text())
    kind = data.get("kind")
    print(f"--- {p.name} ({kind}) ---")
    if kind == "synthetic":
        s = data["stats"]
        print(f"  unique pairs: {s['unique_pairs_after_dedup']}")
        print(f"  verifier accept rate: {s['verifier_accept_rate']:.3f}")
        print(f"  dedup ratio: {s['dedup_ratio']:.3f}")
    elif kind == "agreement":
        for pair in data["pairs"]:
            kappa = pair.get("cohens_kappa")
            print(f"  {pair['pair_name']}: kappa={kappa}")
        sb = data["shortcut_bucket"]
        print(f"  shortcuts: {sb['judge_flagged']}/{sb['total_verifier_accepted']} "
              f"(rate={sb['shortcut_rate']:.3f})")
    elif kind == "robustness":
        print(f"  overall robustness: {data['overall_robustness']}")
        for row in data["per_perturbation"]:
            print(f"    {row['name']:24s} retention={row['retention']}")
    print(f"  wall_clock_seconds: {data['wall_clock_seconds']:.0f}")
print()
print("Download the three JSONs in /kaggle/working/receipts_run/receipts/")
print("and attach them to the M2-R3 PR.")
"""


# Markdown intro so anyone opening the notebook on Kaggle sees what it does
# before running it.
MARKDOWN_INTRO = """\
# Receipt Collection Notebook

This notebook drives three pipelines on a Kaggle T4 GPU and produces
three versioned JSON receipts:

- `synthetic_v1.json` — synthetic-data scale-up summary
- `agreement_v1.json` — three-way (verifier / RM / judge) agreement
- `robustness_v1.json` — perturbation robustness retention

Expected wall-clock: roughly 4-6 hours on a single T4. Each stage
writes its receipt as soon as it finishes, so if Kaggle kills the
session mid-run, the already-completed receipts are still under
`/kaggle/working/receipts_run/receipts/`.

The notebook does not edit the repo or commit anything. After cell 6
prints the summary, download the three JSON files from the file
browser and attach them to the project PR.
"""


def _code(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source=src.rstrip() + "\n")


def _markdown(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source=src.rstrip() + "\n")


def build_notebook() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    }
    cells: List[nbformat.NotebookNode] = [
        _markdown(MARKDOWN_INTRO),
        _code(CELL_0_CLONE),
        _code(CELL_1_SETUP),
        _code(CELL_2_SYNTHETIC),
        _code(CELL_3_AGREEMENT),
        _code(CELL_4_ROBUSTNESS),
        _code(CELL_5_VALIDATE),
        _code(CELL_6_SUMMARY),
    ]
    nb["cells"] = cells
    return nb


def main() -> int:
    nb = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

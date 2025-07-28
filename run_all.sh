#!/usr/bin/env bash 
# run_all.sh  – Convert selected .ipynb files to .py with Jupytext and run them.

set -euo pipefail #e exit on error and u explicits throws an error for unset variables 
# and pipfail makes sure that any part of the pipe that fails the program exits. 
# ---------------- USER-EDITABLE SECTION ----------------
# Directory that holds your notebooks
NOTEBOOK_DIR="notebooks"


NOTEBOOKS=(
    "gen_report_2stage.ipynb"
    "kv_extraction_baseline.ipynb"
    "unit_time_evaluation.ipynb"
    "llm_judge_evaluation.ipynb"
)

# -------------------------------------------------------
JUPYTEXT_DIR="${NOTEBOOK_DIR}/jupytext"
echo "▶︎ Creating/clearing $JUPYTEXT_DIR"
mkdir -p "$JUPYTEXT_DIR"
rm -f "$JUPYTEXT_DIR"/*.py            # Optional: start fresh each time

echo "Converting notebooks to Python with jupytext: " 
for nb in "${NOTEBOOKS[@]}"; do 
    base="$(basename "$nb" .ipynb)"
    src="${NOTEBOOK_DIR}/${nb}"
    dest="${JUPYTEXT_DIR}/${base}.py"

    if [[ ! -f "$src" ]]; then 
        echo "ERROR: Notebook '$src' not found." >&2
        exit 1 
    fi  

    jupytext --to py:percent "$src" -o "$dest"
    echo "  ✅ $src -> $dest ✅"
done 

echo ">> Running scripts sequentially..."
for nb in "${NOTEBOOKS[@]}"; do
    base="$(basename "$nb" .ipynb)"
    script="${JUPYTEXT_DIR}/${base}.py"
    echo "____ Running $script ____"
    # We assume the env is already setup on terminal to run. 
    python "$script"
done 

echo "✅ All notebooks executed successfully."

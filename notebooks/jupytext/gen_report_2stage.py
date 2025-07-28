# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: tada_bench
#     language: python
#     name: tada_bench
# ---

# %% [markdown]
# # Structured to Unstructured generation 
#

# %%
# Standard libraries
import os
import io
import sys
import filelock
from datetime import datetime
from typing import List, Optional, Union, Dict, Any, Tuple
from pathlib import Path

# Data processing and display
import pandas as pd
from tqdm import tqdm

# Environment and configuration
from dotenv import load_dotenv

# DSPy and concurrency
import dspy
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Dataset loading
from datasets import load_dataset


# %%
project_root = Path.cwd().parent
sys.path.append(str(project_root))
print(project_root)
from src.report_generation import *


# %% [markdown]
# # Initial setup and configs

# %% tags=["parameters"]
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
load_dotenv()

# folder_name = "SEC_WikiDB"
folder_name = "SEC_WikiDB_subset"

data_type = "unfiltered"
subset = "all"

force_reprocess = False

LITELLM_MODEL = os.environ["LITELLM_MODEL"]
LLM_API_BASE = os.environ["BASE_URL"]
model_name = os.environ["RG_MODEL_NAME"]
API_KEY = os.environ["API_KEY"]

out_dir = os.environ["OUTPUT_DIR"]
max_workers = int(os.environ["MAX_WORKERS"])  # max_workers = 50
# Row sampling configuration
ROW_SAMPLE_SIZE = os.environ["ROW_SAMPLE_SIZE"]
ROW_RANDOM_SEED = int(os.environ["ROW_RANDOM_SEED"])
randomize = False
liteLLM_retries = int(os.environ["LITELLM_RETRIES"])  # 15

# Convert ROW_SAMPLE_SIZE to None if it's 'None'
if ROW_SAMPLE_SIZE == "None":
    ROW_SAMPLE_SIZE = None
else:
    ROW_SAMPLE_SIZE = int(ROW_SAMPLE_SIZE)


# %%
out_name = model_name
out_name = out_name.replace(".", "_").lower()
out_name = Path(out_name).stem
print(
    model_name,
    out_name,
)

output_path = f"{out_dir}/{folder_name}_{data_type}_{subset}"

output_lock_dir = f"{out_dir}/locks"  # Directory for file locks
os.makedirs(output_lock_dir, exist_ok=True)


# %%
# Load subset for faster experimentation. "SEC_WikiDB subset unfiltered - all file types" - The smaller 49 csv files for quick prototyping.

dataset = load_dataset(
    "ibm-research/struct-text",
    f"{folder_name}_{data_type}_{subset}",
    streaming=False,
    cache_dir=output_path,
)


# %% [markdown]
# # Model Configs

# %%
completion_args = {
    "model": f"{LITELLM_MODEL}/{model_name}",
    "api_base": f"{LLM_API_BASE}",
    "temperature": 0,
    "api_key": API_KEY,
    "cache": 0,
    "num_retries": liteLLM_retries,
}


# %%
# Configure DSPy with the language model
lm = dspy.LM(**completion_args)
dspy.configure(lm=lm)
print(f"Model: {model_name}, Output name: {out_name}")


# %%
resp = lm("Ping ")
print(resp)


# %% [markdown]
# # Execute processing

# %%
splits = {
    "train": dataset["train"],
    "val": dataset["validation"],
    "test": dataset["test"],
}


# %%
for split_type, split_data in splits.items():
    print(split_type, split_data)
    print("--" * 10)


# %%
report_planner = dspy.Predict(ReportPlannerWithSamples)
report_text_generator = dspy.Predict(ReportTextGenerator)


# %%
for split_type, split_data in splits.items():
    print(split_type)
    label_names = split_data.features["report_type"].names
    print(label_names)
    for split_row in split_data:
        row_label = label_names[split_row["report_type"]]
        file_name = split_row["file_name"]
        if row_label == "ground_truth":
            # print(row_label)
            print(file_name)
            process_report_generation(
                data_row=split_row,
                split_type=split_type,
                output_path=output_path,
                out_name=out_name,
                folder_name=folder_name,
                output_lock_dir=output_lock_dir,
                force_reprocess=force_reprocess,
                report_planner=report_planner,
                report_text_generator=report_text_generator,
                ROW_RANDOM_SEED=ROW_RANDOM_SEED,
                ROW_SAMPLE_SIZE=ROW_SAMPLE_SIZE,
                randomize=randomize,
                max_workers=max_workers,
            )
    print("==" * 10)


# %%

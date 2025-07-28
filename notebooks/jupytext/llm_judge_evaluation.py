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

# %%
import os
import re
import json
import sys
import time
import random
import filelock
import io
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from tqdm.notebook import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# LangChain and DSPy
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import dspy

# Environment setup
from dotenv import load_dotenv


# %%
project_root = Path.cwd().parent
sys.path.append(str(project_root))

from src.evaluation_utils import TextQualityEvaluator, getenv_bool


# %% [markdown]
# # Initial Setup and configs

# %% tags=["parameters"]
load_dotenv()
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


force_reprocess = True


folder_name = "SEC_WikiDB_subset"
data_type = "unfiltered"
subset = "all"
run_from_localdir = getenv_bool("RUN_LOCAL", default=False)


LITELLM_MODEL = os.environ["LITELLM_MODEL"]
LLM_API_BASE = os.environ["BASE_URL"]
model_name = os.environ["MODEL_NAME"]
rg_model_name = os.environ["RG_MODEL_NAME"]
API_KEY = os.environ["API_KEY"]

out_dir = os.environ["OUTPUT_DIR"]
max_workers = int(os.environ["MAX_WORKERS"])
liteLLM_retries = int(os.environ["LITELLM_RETRIES"])  # 15

# Row sampling configuration
# Number of rows to sample per dataset (None for all rows)
ROW_SAMPLE_SIZE = os.environ["ROW_SAMPLE_SIZE"]
ROW_RANDOM_SEED = int(os.environ["ROW_RANDOM_SEED"])

# Convert ROW_SAMPLE_SIZE to None if it's 'None'
if ROW_SAMPLE_SIZE == "None":
    ROW_SAMPLE_SIZE = None
else:
    ROW_SAMPLE_SIZE = int(ROW_SAMPLE_SIZE)


# %%
out_name = model_name
out_name = out_name.replace(".", "_")
out_name = Path(out_name).stem


rg_model_name = rg_model_name.replace(".", "_")
rg_model_name = Path(rg_model_name).stem

# Path setup
output_path = f"{out_dir}/{folder_name}_{data_type}_{subset}"
eval_reports_path = Path(f"{out_dir}/eval_reports_{folder_name}")
eval_reports_path.mkdir(parents=True, exist_ok=True)
error_log = f"{eval_reports_path}/llm_judge_errors_{out_name}.txt"


output_lock_dir = f"{out_dir}/locks"  # Directory for file locks
os.makedirs(output_lock_dir, exist_ok=True)


# %%
from datasets import load_dataset

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
print(f"Model: {model_name}, Output name: {out_name}, RG model: {rg_model_name}")


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
lm = dspy.LM(**completion_args)
dspy.configure(lm=lm)


# %%
resp = lm("Ping")
print(resp)


# %%
text_quality_evaluator = TextQualityEvaluator()


# %% [markdown]
# # Parallel Processing 

# %%
def process_dataset(
    dataset_name,
    ground_truth_data,
    report_types_data,
    generated_reports_data,
    split_type,
    rg_model_name,
):
    """Process a single dataset for LLM judge evaluation."""
    # Read metadata (with lock to avoid race conditions)
    meta_csv_path = (
        Path(output_path).parent / f"meta_data_{rg_model_name}_{folder_name}.csv"
    )
    with filelock.FileLock(f"{output_lock_dir}/metadata.lock"):
        if run_from_localdir:
            # Read metadata
            if not meta_csv_path.exists():
                raise FileNotFoundError(f"Metadata file not found at {meta_csv_path}")
            meta_df = pd.read_csv(meta_csv_path)
        else:
            meta_ds = load_dataset(
                "ibm-research/struct-text",
                data_files=f"meta_data_{rg_model_name}_{folder_name}.csv",
            )
            meta_df = meta_ds["train"].to_pandas()
            # save first:
            meta_df.to_csv(meta_csv_path, index=False)

    # Get the row for this dataset
    dataset_row = meta_df[meta_df["dataset_name"] == dataset_name]
    if len(dataset_row) == 0:
        raise ValueError(f"Dataset '{dataset_name}' not found in metadata.")
    if len(dataset_row) > 1:
        raise ValueError(f"Multiple entries for dataset '{dataset_name}' in metadata.")

    # Get the single row
    row = dataset_row.iloc[0]

    # Define output path and column
    column_name = f"text_quality_csv_path_{out_name}"
    output_csv = Path(eval_reports_path) / split_type
    Path(output_csv).mkdir(parents=True, exist_ok=True)

    output_csv = (
        output_csv / f"{dataset_name}_{rg_model_name}_text_quality_{out_name}.csv"
    )
    # Skip if already processed
    existing_path = row.get(column_name, pd.NA)
    if (
        not force_reprocess
        and not pd.isna(existing_path)
        and os.path.exists(existing_path)
        and os.path.getsize(existing_path) > 0
    ):
        print(f"Skipping {dataset_name} - already processed")
        return

    # Read data

    if run_from_localdir:
        original_csv = row["ground_truth_csv_path"]
        planned_csv = row["report_types_csv_path"]
        generated_csv = row["generated_reports_csv_path"]

        print(f"Processing {dataset_name}")
        print(f"Loading data from: {original_csv}, {planned_csv}, {generated_csv}")

        orig_df = pd.read_csv(original_csv)
        planned_df = pd.read_csv(planned_csv)
        generated_df = pd.read_csv(generated_csv)
        row_plan = planned_df.iloc[0]
    else:
        # process from the hf datasets. The dataset_row should have that info:
        orig_df = pd.read_csv(io.StringIO(ground_truth_data["csv_text"]))
        planned_df = pd.read_csv(io.StringIO(report_types_data["csv_text"]))
        generated_df = pd.read_csv(io.StringIO(generated_reports_data["csv_text"]))
        row_plan = planned_df.iloc[0]

    # Sample rows if requested:
    total_rows = len(generated_df)
    if ROW_SAMPLE_SIZE is not None and total_rows > ROW_SAMPLE_SIZE:
        # use fixed seed for reproducible sampling unique to the dataset as well.
        # apparently random uses a hashmap and can take in unique strings as well
        random.seed(f"{ROW_RANDOM_SEED}_{dataset_name}")
        sampled_indices = random.sample(range(total_rows), ROW_SAMPLE_SIZE)
        sampled_indices.sort()

        orig_df_sampled = orig_df.iloc[sampled_indices].reset_index(drop=True)
        generated_df_sampled = generated_df.iloc[sampled_indices].reset_index(drop=True)
        print(f"Sampled {len(sampled_indices)} rows from {total_rows} total rows")
    else:
        # Use all the rows
        orig_df_sampled = orig_df
        generated_df_sampled = generated_df
        sampled_indices = list(range(total_rows))
        print(f"Processing all {total_rows} rows (no sampling)")

    # Initialize results list
    text_quality_data = [None] * len(sampled_indices)

    #############################################################
    def process_llm_as_judge(sample_idx, original_idx):
        try:
            row_orig = orig_df_sampled.iloc[sample_idx]
            row_gen = generated_df_sampled.iloc[sample_idx]
            result = text_quality_evaluator(
                source_data=row_orig.to_dict(), generated_report=row_gen.to_dict()
            )
            data_out = {
                "original_index": original_idx,  # Keep track of original index
                "src_data": result.source_data,
                "gen_text": result.generated_report,
                "factual_claims": result.claim_analysis,
                "factuality_score": result.factuality_score,
                "factual_reasoning": result.factuality_reasoning,
                "hallucination_claims": result.statement_analysis,
                "hallucination_score": result.hallucination_score,
                "hallucination_reasoning": result.hallucination_reasoning,
                "coherence_claims": result.coherence_issues,
                "coherence_score": result.coherence_score,
                "coherence_reasoning": result.coherence_reasoning,
                "overall_quality_score": result.overall_quality_score,
            }
            return sample_idx, data_out, None
        except Exception as e:
            error_msg = f"Error processing {dataset_name}, row {original_idx}: {str(e)}"
            return (
                sample_idx,
                {},
                error_msg,
            )

    # #############################################################
    # Process rows in parallel
    result_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, original_idx in enumerate(sampled_indices):
            future = executor.submit(process_llm_as_judge, i, original_idx)
            future_to_idx[future] = i

        with tqdm(
            total=len(sampled_indices), desc=f"LLM judge eval for {dataset_name}"
        ) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    sample_idx, data_out, error = future.result()
                    if data_out is not None:
                        with result_lock:
                            text_quality_data[sample_idx] = data_out
                    if error is not None:
                        with open(error_log, "a") as f:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            f.write(
                                f"{timestamp} - {dataset_name} - {idx} (original: {data_out['original_index']}) - {error}\n"
                            )
                except Exception as e:
                    print(f"Error handling result for {idx}: {str(e)}")
                pbar.update(1)
    ############################################
    # Clean None values and create DataFrame
    text_quality_data = [
        (
            item
            if isinstance(item, dict)
            else {"error": "Missing result", "original_index": sampled_indices[i]}
        )
        for i, item in enumerate(text_quality_data)
    ]
    text_quality_df = pd.DataFrame(text_quality_data)

    # Add sampling metadata
    sampling_info = {
        "dataset_name": dataset_name,
        "total_rows": total_rows,
        "sampled_rows": len(sampled_indices),
        "sampling_method": (
            "random" if ROW_SAMPLE_SIZE and total_rows > ROW_SAMPLE_SIZE else "full"
        ),
        "sample_seed": (
            f"{ROW_RANDOM_SEED}_{dataset_name}"
            if ROW_SAMPLE_SIZE and total_rows > ROW_SAMPLE_SIZE
            else None
        ),
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save sampling info and results
    with open(output_csv.with_suffix(".json"), "w") as f:
        json.dump(sampling_info, f, indent=2)

    # Write results
    text_quality_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    # Update metadata with a file lock to prevent race conditions
    with filelock.FileLock(f"{output_lock_dir}/metadata.lock"):
        # Re-read metadata to get the latest version
        meta_df = pd.read_csv(meta_csv_path)
        # Ensure the dataset still exists
        if dataset_name not in meta_df["dataset_name"].values:
            print(
                f"Warning: Dataset {dataset_name} no longer in metadata. Skipping update."
            )
            return

        # Find the row again (index might have changed)
        row_idx = meta_df[meta_df["dataset_name"] == dataset_name].index[0]
        # Update the metadata
        meta_df.at[row_idx, column_name] = str(output_csv)

        # Save the metadata
        meta_df.to_csv(meta_csv_path, index=False)

    print(f"Metadata updated for {dataset_name}")



# %% [markdown]
# # Run

# %%
label_names = [
    "ground_truth",
    "report_types",
    "generated_reports",
]  # Hardcoded: See HF ;


# %%
if not run_from_localdir:
    splits = {
        "train": dataset["train"],
        "val": dataset["validation"],
        "test": dataset["test"],
    }
    # this is the only run available on hf.
    rg_model_name = "Qwen2_5-72B-Instruct"

    for split_type, split_data in splits.items():
        # create a dictionary to group files by dataset_name:
        dataset_groups = defaultdict(
            lambda: {
                "ground_truth": None,
                "report_types": None,
                "generated_reports": None,
            }
        )

        # First split by all files by datast name:
        for idx, split_row in enumerate(split_data):
            row_label = label_names[split_row["report_type"]]
            file_name = split_row["file_name"]

            # extract the dataset name based on the file type:
            if "_ground_truth.csv" in file_name:
                dataset_name = file_name.replace("_ground_truth.csv", "")
                dataset_groups[dataset_name]["ground_truth"] = (idx, split_row)

            if "_report_types_" in file_name:
                dataset_name = file_name.replace(
                    f"_report_types_{rg_model_name}.csv", ""
                )
                dataset_groups[dataset_name]["report_types"] = (idx, split_row)

            if "_generated_reports_" in file_name:
                dataset_name = file_name.replace(
                    f"_generated_reports_{rg_model_name}.csv", ""
                )
                dataset_groups[dataset_name]["generated_reports"] = (idx, split_row)

        ######################################################################################
        # now process with each dataset group with all three files:
        for dataset_name, files in dataset_groups.items():
            if all(files.values()):
                print(f"\nProcessing dataset: {dataset_name} in {split_type}")
                # Extract the data for all three files
                ground_truth_data = files["ground_truth"][1]
                report_types_data = files["report_types"][1]
                generated_reports_data = files["generated_reports"][1]

                print(ground_truth_data["file_name"], ground_truth_data["report_type"])
                print(report_types_data["file_name"], report_types_data["report_type"])
                print(
                    generated_reports_data["file_name"],
                    generated_reports_data["report_type"],
                )

                process_dataset(
                    dataset_name,
                    ground_truth_data,
                    report_types_data,
                    generated_reports_data,
                    split_type,
                    rg_model_name,
                )


else:
    for split_type in ["train", "val", "test"]:
        # for split_type in ["val", "test"]:
        split_dir = os.path.join(output_path, split_type)
        if not os.path.exists(split_dir):
            print(f"Split directory {split_dir} does not exist. Skipping {split_type}.")
            continue

        # create a dictionary to group files by dataset_name:
        dataset_groups = defaultdict(
            lambda: {
                "ground_truth": None,
                "report_types": None,
                "generated_reports": None,
            }
        )
        # List all CSV files in the split directory
        for file_name in os.listdir(split_dir):
            if not file_name.endswith(".csv"):
                continue

            full_path = os.path.join(split_dir, file_name)
            with open(full_path, "r", encoding="utf-8") as f:
                csv_text = f.read()

            # extract the dataset name based on the file type:
            if "_ground_truth.csv" in file_name:
                dataset_name = file_name.replace("_ground_truth.csv", "")
                dataset_groups[dataset_name]["ground_truth"] = {
                    "file_name": file_name,
                    "report_type": 0,  # Dummy value
                }

            if "_report_types_" in file_name:
                dataset_name = file_name.replace(
                    f"_report_types_{rg_model_name}.csv", ""
                )
                dataset_groups[dataset_name]["report_types"] = {
                    "file_name": file_name,
                    "report_type": 1,  # Dummy value
                }

            if "_generated_reports_" in file_name:
                dataset_name = file_name.replace(
                    f"_generated_reports_{rg_model_name}.csv", ""
                )
                dataset_groups[dataset_name]["generated_reports"] = {
                    "file_name": file_name,
                    "report_type": 2,  # Dummy value
                }
        ######################################################################################
        # now process each dataset group with all three files:
        for dataset_name, files in dataset_groups.items():
            # local mode: Assume data is structured in output_path/split_type/ with CSV files
            print(
                f"RUNNING LOCALLY GENERATED REPORTS: LABEL: run_from_localdir {run_from_localdir}"
            )
            if all(files.values()):
                print(f"\nProcessing dataset: {dataset_name} in {split_type}")
                # Extract the data for all three files
                ground_truth_data = files["ground_truth"]
                report_types_data = files["report_types"]
                generated_reports_data = files["generated_reports"]

                print(ground_truth_data["file_name"], ground_truth_data["report_type"])
                print(report_types_data["file_name"], report_types_data["report_type"])
                print(
                    generated_reports_data["file_name"],
                    generated_reports_data["report_type"],
                )

                process_dataset(
                    dataset_name,
                    ground_truth_data,
                    report_types_data,
                    generated_reports_data,
                    split_type,
                    rg_model_name,
                )


# %%

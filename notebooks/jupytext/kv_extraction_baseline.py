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
# # Imports + config

# %%
from datasets import load_dataset
from dotenv import load_dotenv
from collections import defaultdict
import sys, pathlib
from pathlib import Path


# %%
project_root = Path.cwd()
print(f"KV Extraction baseline project root: {project_root}")
sys.path.append(str(project_root))
from src.kv_extraction import *
from src.evaluation_utils import getenv_bool


# %% [markdown]
# # Initial Setup and Configurations

# %% tags=["parameters"]
# Environment setup
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


"""
The following should be set in your .env file
BASE_URL - the URL of the API you are using to call the LLM
MODEL_NAME - the name of the LLM model you are using
API_KEY - your api key for LLM access
"""
load_dotenv()


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
max_workers = int(os.environ["MAX_WORKERS_KV"])
liteLLM_retries = int(os.environ["LITELLM_RETRIES"])  # 15

# Row sampling configuration
# Number of rows to sample per dataset (None for all rows)
ROW_SAMPLE_SIZE = os.environ["ROW_SAMPLE_SIZE"]
ROW_RANDOM_SEED = int(os.environ["ROW_RANDOM_SEED"])
randomize = False


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

max_threads = max_workers
error_log = f"{out_dir}/eval_reports_{folder_name}/kv_extraction_errors_{out_name}.txt"
Path(error_log).parent.mkdir(exist_ok=True, parents=True)

wikidb_path = f"{out_dir}/{folder_name}"
output_path = f"{out_dir}/{folder_name}_{data_type}_{subset}"

# Path setup
eval_reports_path = Path(f"{out_dir}/eval_reports_{folder_name}")
eval_reports_path.mkdir(parents=True, exist_ok=True)

# Ensure lock directory exists
output_lock_dir = f"{out_dir}/locks"
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
    "num_retries": liteLLM_retries,
}


# %%
get_llm_response("Ping", API_KEY, completion_args)


# %% [markdown]
# # Example run to process a target dataset

# %%

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
                    dataset_name=dataset_name,
                    split_type=split_type,
                    ground_truth_data=ground_truth_data,
                    report_types_data=report_types_data,
                    generated_reports_data=generated_reports_data,
                    run_from_localdir=run_from_localdir,
                    output_path=output_path,
                    folder_name=folder_name,
                    out_name=out_name,
                    eval_reports_path=eval_reports_path,
                    rg_model_name=rg_model_name,
                    error_log=error_log,
                    API_KEY=API_KEY,
                    completion_args=completion_args,
                    ROW_SAMPLE_SIZE=ROW_SAMPLE_SIZE,
                    ROW_RANDOM_SEED=ROW_RANDOM_SEED,
                    max_workers=max_workers,
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
                    dataset_name=dataset_name,
                    split_type=split_type,
                    ground_truth_data=ground_truth_data,
                    report_types_data=report_types_data,
                    generated_reports_data=generated_reports_data,
                    run_from_localdir=run_from_localdir,
                    output_path=output_path,
                    folder_name=folder_name,
                    out_name=out_name,
                    eval_reports_path=eval_reports_path,
                    rg_model_name=rg_model_name,
                    error_log=error_log,
                    API_KEY=API_KEY,
                    completion_args=completion_args,
                    ROW_SAMPLE_SIZE=ROW_SAMPLE_SIZE,
                    ROW_RANDOM_SEED=ROW_RANDOM_SEED,
                    max_workers=max_workers,
                )


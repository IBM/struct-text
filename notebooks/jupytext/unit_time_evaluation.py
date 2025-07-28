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
# # Initial setup 

# %%
# Initial setup
from __future__ import annotations

import json
import re
import os
import io
import sys
import random
import math

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, PydanticUndefinedAnnotation
from collections import defaultdict

import pandas as pd
import numpy as np
import dateutil
import datetime as dt


import dspy

# Execution and Multithreading
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging, logging.handlers, queue, threading
from tqdm import tqdm
import filelock
import ast

# Environment setup
from dotenv import load_dotenv
from datasets import load_dataset
from contextlib import contextmanager, redirect_stdout



# %%
project_root = Path.cwd().parent
print(f"Unit time accuracy project root: {project_root}")
sys.path.append(str(project_root))

from src.evaluation_utils import *



# %%
ROW_LOG = logging.getLogger("row")


# hotfix this back to the
@contextmanager
def capture_row():
    """
    Temporariliy diver all records from `logger_name` into an in-memory StringIO Buffer.
    yeilds the buffer and resores the logger afterwards
    """
    log_buf = io.StringIO()
    out_buf = io.StringIO()

    # logging.getLogger(logger_name)
    hdlr = logging.StreamHandler(log_buf)
    hdlr.setFormatter(logging.Formatter("%(message)s"))
    ROW_LOG.addHandler(hdlr)
    ROW_LOG.propagate = False

    with redirect_stdout(out_buf):
        try:
            yield lambda: log_buf.getvalue() + out_buf.getvalue()
        finally:
            ROW_LOG.removeHandler(hdlr)



# %% [markdown]
# # Initial setup and configs
#

# %% tags=["parameters"]
load_dotenv()
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# Row sampling configuration
output_stream_to_console = True
is_verbose = False
force_reprocess = True
exec_parallel_run = True  # True
# if you want the logging stream to also output onto the console:


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


# Number of rows to sample per dataset (None for all rows)
ROW_SAMPLE_SIZE = os.environ["ROW_SAMPLE_SIZE"]
ROW_RANDOM_SEED = int(os.environ["ROW_RANDOM_SEED"])
randomize = False
liteLLM_retries = int(os.environ["LITELLM_RETRIES"])  # 15

# Convert ROW_SAMPLE_SIZE to None if it's 'None'
if ROW_SAMPLE_SIZE == "None":
    ROW_SAMPLE_SIZE = None
else:
    ROW_SAMPLE_SIZE = int(ROW_SAMPLE_SIZE)

corenlp_libs = os.environ["CORE_NLP_LIBS"]



# %%
out_name = model_name
out_name = out_name.replace(".", "_")
out_name = Path(out_name).stem

rg_model_name = rg_model_name.replace(".", "_")
rg_model_name = Path(rg_model_name).stem

# Error log file
error_log = f"{out_dir}/eval_reports_{folder_name}/unit_time_errors_{out_name}.txt"
wikidb_path = f"{out_dir}/{folder_name}"
output_path = f"{out_dir}/{folder_name}_{data_type}_{subset}"

# Path setup
eval_reports_path = Path(f"{out_dir}/eval_reports_{folder_name}")
eval_reports_path.mkdir(parents=True, exist_ok=True)

# Ensure lock directory exists
output_lock_dir = f"{out_dir}/locks"  # Directory for file locks
os.makedirs(output_lock_dir, exist_ok=True)



# %%
from datasets import load_dataset

# Load subset for faster experimentation. "SEC_WikiDB subset unfiltered - all file types" - The smaller 49 csv files for quick prototyping.
if not run_from_localdir:
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



# %% [markdown]
# # Unit Time accuracy builder

# %%
from src.evaluation_utils import ensure_sutime_client, concretise_timex, ureg

# spin up (only the first call actually starts JVM)
sutime_client = ensure_sutime_client(corenlp_libs)

doc = sutime_client.annotate("Delivery expected next Tuesday.")
for ent in doc.mentions:
    print(ent)



# %% [markdown]
# ## Unit Time Accuracy - Parsers and helpler funcs

# %%
Numeric = Tuple[float, Optional[str]]  # (value, int)


def _canonicalise(value: float, unit: Optional[str]) -> float:
    """coanonicalization with the pint package"""
    if ureg and unit:
        try:
            q = value * ureg(unit)
            return q.to_base_units().magnitude
        except Exception:
            pass
    return value


def _extract_numerics_with_ner(text: str) -> List[Numeric]:
    """Use the coreNLP to parse the named entity recognition types pertaining to the numeric"""
    nums = []
    doc_json = sutime_client.annotate(text, output_format="json")

    # Extract the Named entity recongition focussing on {"NUMBER", "MONEY", "ORDINAL", "PERCENT"}:
    # REF: https://stanfordnlp.github.io/CoreNLP/ner.html#description
    for s in doc_json["sentences"]:
        for em in s.get("entitymentions", []):
            if em["ner"] in {"NUMBER", "MONEY", "ORDINAL", "PERCENT"}:
                # regex my way out of this mess
                text_clean = em["text"].replace(",", "")
                # Extract only the numeric part:
                numeric_match = re.search(r"[\d.]+", text_clean)
                if numeric_match:
                    value = float(numeric_match.group())
                    nums.append((value, em["ner"]))
    return nums


# ---------------------------------------------------------------------------


def _extract_dates_with_sutime(text):
    # Extract dates from text using SUTime via Stanford's CoreNLP.
    # List to store extracted datetime objects
    date_objects = []
    try:
        # Annotate the text
        doc_json = sutime_client.annotate(text, output_format="json")

        # Extract TIMEX annotations, focusing on standard date formats
        for s in doc_json["sentences"]:
            for em in s.get("entitymentions", []):
                if "timex" in em:
                    t = em["timex"]
                    date_str = t["value"]
                    date_objects.append(
                        {"text": em["text"], "value": t["value"], "type": t["type"]}
                    )

    except Exception as e:
        print(f"Error in SUTime date extraction: {e}")

    return date_objects


def _convert_to_datetime(date_str: str) -> Optional[dt.datetime]:
    try:
        # Handle full ISO format dates
        return dt.datetime.fromisoformat(date_str)
    except ValueError:
        if re.match(r"^\d{4}$", date_str):
            return dt.datetime(int(date_str), 1, 1)
        # Handle year-month formats
        elif re.match(r"^\d{4}-\d{2}$", date_str):
            year, month = date_str.split("-")
            return dt.datetime(int(year), int(month), 1)

        return None


# ---------------------------------------------------------------------------
def _calculate_summary_statistics(
    numeric_metrics: List[Dict], temporal_metrics: List[Dict]
) -> Dict[str, float]:
    summary = {}
    # Numeric summary statistics
    if numeric_metrics:
        num_precisions = [
            m["precision"] for m in numeric_metrics if m.get("precision") is not None
        ]
        num_recalls = [
            m["recall"] for m in numeric_metrics if m.get("recall") is not None
        ]
        num_f1s = [m["f1"] for m in numeric_metrics if m.get("f1") is not None]

        summary.update(
            {
                "numeric_mean_precision": (
                    float(pd.Series(num_precisions).mean()) if num_precisions else None
                ),
                "numeric_mean_recall": (
                    float(pd.Series(num_recalls).mean()) if num_recalls else None
                ),
                "numeric_mean_f1": (
                    float(pd.Series(num_f1s).mean()) if num_f1s else None
                ),
                "numeric_evaluations_count": len(numeric_metrics),
            }
        )
    # Temporal summary statistics
    if temporal_metrics:
        tmp_precisions = [
            m["precision"] for m in temporal_metrics if m.get("precision") is not None
        ]
        tmp_recalls = [
            m["recall"] for m in temporal_metrics if m.get("recall") is not None
        ]
        tmp_f1s = [m["f1"] for m in temporal_metrics if m.get("f1") is not None]

        summary.update(
            {
                "temporal_mean_precision": (
                    float(pd.Series(tmp_precisions).mean()) if tmp_precisions else None
                ),
                "temporal_mean_recall": (
                    float(pd.Series(tmp_recalls).mean()) if tmp_recalls else None
                ),
                "temporal_mean_f1": (
                    float(pd.Series(tmp_f1s).mean()) if tmp_f1s else None
                ),
                "temporal_evaluations_count": len(temporal_metrics),
            }
        )
    # Overall statistics
    all_f1s = []
    if numeric_metrics:
        all_f1s.extend([m["f1"] for m in numeric_metrics if m.get("f1") is not None])
    if temporal_metrics:
        all_f1s.extend([m["f1"] for m in temporal_metrics if m.get("f1") is not None])

    summary["overall_mean_f1"] = float(pd.Series(all_f1s).mean()) if all_f1s else None
    summary["total_evaluations"] = len(numeric_metrics) + len(temporal_metrics)

    return summary



# %% [markdown]
# ## Main Class 

# %%
class UnitTimeAccuracyEvaluator:
    def __init__(
        self,
        *,  # “Everything that comes after this must be passed by keyword, not by position.”
        original_csv: pd.DataFrame,
        plan_csv: pd.DataFrame,
        generated_csv: pd.DataFrame,
        numeric_tol: float = 1e-4,
        temporal_tol_days: int = 0,
        verbose: bool = True,
        temporal_extractor: dspy.Signature,
    ) -> None:
        self.df_orig = original_csv
        self.df_plan = plan_csv
        self.df_gen = generated_csv

        self.numeric_tol = numeric_tol
        self.temporal_tol_days = temporal_tol_days

        self.verbose = verbose

        # Build the dspy:
        self.temporal_extractor = dspy.Predict(temporal_extractor)

        # build plan mapping:
        self.plan_map: Dict[str, List[str]] = {
            col: ast.literal_eval(self.df_plan.iloc[0, idx])
            for idx, col in enumerate(self.df_plan.columns)
        }

    # ------------------------------------------------------------------
    def _compare_numeric(self, pred: float, truth: float) -> bool:
        if truth == 0:
            return abs(pred - truth) < self.numeric_tol
        return abs(pred - truth) / abs(truth) <= self.numeric_tol

    def _compare_temporal(self, pred_str: str, truth_str: str) -> bool:
        # Direct string match for quarters
        if "-Q" in pred_str and "-Q" in truth_str:
            return pred_str == truth_str

        pred_dt = _convert_to_datetime(pred_str)
        truth_dt = _convert_to_datetime(truth_str)
        # if self.verbose:
        #     print(f"Date time post parsing: Pred: {pred_dt} --> Truth: {truth_dt}")
        if not pred_dt or not truth_dt:
            return False

        delta = abs(pred_dt.date() - truth_dt.date())
        # if self.verbose:
        #     print(f"delta: {delta}, pred: {pred_dt.date()}, truth: {truth_dt.date()}")
        #     print("--" * 10)
        return delta.days <= self.temporal_tol_days

    # ------------------------------------------------------------------

    def _check_numeric_improved(
        self, rep_type: str, text: str, row_gen: pd.Series
    ) -> Tuple[Dict[str, float], List[str]]:
        fails: List[str] = []
        cols = self.plan_map.get(rep_type, [])
        # extracted = _extract_numerics(text)
        extracted = _extract_numerics_with_ner(text)
        if self.verbose:
            print(f"Processing truth row: \n{row_gen.get(cols)}")
            print(f"Gen Txt: {text}")
            print(f"Report type: {rep_type}")
            print(f"Extracted nums: {extracted}")
            print(f"Num of Extracted nums: {len(extracted)}")
            print("===" * 3)

        # Count valid ground truth numeric values
        valid_ground_truth = []
        for col in cols:
            truth_raw = row_gen.get(col)
            if pd.isna(truth_raw):
                continue
            try:
                truth_val = float(truth_raw)
                valid_ground_truth.append((col, truth_val))
            except Exception:
                continue

        # Case 1: No ground truth numerics exist
        if not valid_ground_truth:
            if not extracted:
                # correctly identified no numeric information
                return {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "interpretation": "correct_no_numerics",
                }, []
            else:
                # Error: Extractd numerics when none should exist.
                # This is layering both the error from the pandas and maybe the way the number is represented.
                # I recognize this but I think that's ok. Most of the spot checking did'nt trigger a false part here.
                return {
                    "precision": 0.0,  # 0 correct out of len(extracted) extractions
                    "recall": None,  # undefined: can't miss what dosen't exist (TP/(TP+FN) = 0/(0+0) = 0/0)
                    "f1": None,  # Cant calculate F1 when recall is undefined
                    "interpretation": "false_extractions_no_ground_truth",
                }, [f"extracted {len(extracted)} numbers when none expected"]
        # Case 2: Ground truth numerics exist:
        if not extracted:
            # Error: Should have extracted but didn't
            return {
                "precision": None,  # Undefined: because no extractions to evaluate: (TP/(TP+FP)=0/(0+0) = undefined)
                "recall": 0.0,  # Missed eveyrthing that existed
                "f1": None,  # can't calulate f1 when precision is undefined.
                "true_positives": 0,
                "extracted_count": 0,
                "ground_truth_count": len(valid_ground_truth),
                "interpretation": "no_extractions_made",
            }, [f"made no extractions when {len(valid_ground_truth)} dates expected"]

        # case 3: both ground truth and extractions exist - standard eval:
        used_extractions = set()
        matched_truths = 0

        for col, truth_val in valid_ground_truth:
            truth_cannon = _canonicalise(truth_val, None)

            if self.verbose:
                print(f"Truth raw: {truth_val}, col: {col}")
                print(f"Truth canonical: {truth_cannon}")

            found_match = False
            for i, (pred_val, pred_unit) in enumerate(extracted):
                if i in used_extractions:
                    continue

                pred_cannon = _canonicalise(pred_val, pred_unit)
                if self._compare_numeric(pred_cannon, truth_cannon):
                    used_extractions.add(i)
                    matched_truths += 1
                    found_match = True
                    if self.verbose:
                        print(
                            f"Matched Pred: {pred_cannon}, matched_truths_count: {matched_truths}"
                        )
                        print("--" * 15)

                    break

            if not found_match:
                fails.append(f"numeric mismatch: {col}")

        # Standard precisio/recall:
        tp = matched_truths
        fp = len(extracted) - len(used_extractions)
        fn = len(valid_ground_truth) - matched_truths

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "interpretation": "standard_evaluation",
        }, fails

    # ------------------------------------------------------------------
    def _check_temporal_improved_with_llm(
        self,
        rep_type: str,
        text: str,
        row_gen: pd.Series,
    ) -> Tuple[Dict[str, float], List[str], str]:
        """Modified with LLM pre-parsing to handle the date variablilty."""
        fails: List[str] = []
        cols = self.plan_map.get(rep_type, [])

        temporal_extractor = self.temporal_extractor
        # Step 1: Use LLM to extract temporal information
        # instaed of using SUTime directly we use the LLM first:

        if self.verbose:
            print(f"Using LLM to extract temporal info from text...")
            print(f"Processing truth row: \n{row_gen.get(cols)}")
            print(f"Gen Txt: {text}")

        try:
            result = temporal_extractor(text=text)
            # Parse the json output from DSPy with pydantic verification.
            extracted_dspy = result.temporal_json
            ################ Extract normalized text ###########
            extracted_dates = []
            for ed in extracted_dspy:
                ed = (
                    ed.model_dump()
                )  # since this is a pydantic output to use it as a dict you need this line.
                if "normalized_value" in ed:
                    extracted_dates.append(
                        {
                            "value": ed["normalized_value"],
                            "type": ed.get("type", "unknown"),
                            "text": ed.get("original_text", ""),
                        }
                    )
            if not extracted_dates:  # empty list
                if self.verbose:
                    print(f"DSPy found no dates, falling back to SUTime")
                extracted_dates = _extract_dates_with_sutime(text)

        except json.JSONDecodeError as e:
            print(f"Failed to parse DSPy JSON output: {e}")
            print(f"Raw output: {result.temporal_json}")
            print("ErErErEr" * 20)
            # Show the exact representation with all hidden characters
            print(f"Raw output repr: {repr(result.temporal_json)}")

            extracted_dates = []
        except Exception as e:
            print(f"DSPy temporal extraction failed: {e}")
            # Fallback to SUTime
            # extracted_dates = []
            extracted_dates = _extract_dates_with_sutime(text)

        ##########################################################
        masked_out = text

        if self.verbose:
            print(f"Number of extracted dates: {len(extracted_dates)}")
            print(f"Extracted Dates: {extracted_dates}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # STEP 3: Process ground truth
        valid_ground_truth = []
        for col in cols:
            truth_raw = row_gen.get(col)
            if pd.isna(truth_raw):
                continue

            parsed_result = None
            # First use SUTime for truth parsing. Speeds up processing.
            try:
                truth_date = _extract_dates_with_sutime(str(truth_raw))
                if self.verbose:
                    print(f"Truth value for {col}: {truth_raw} -> {truth_date}")

                # Check if SUTime found anything useful
                if truth_date and truth_date[0]["type"] in ("DATE", "TIME"):
                    parsed_result = (col, truth_date[0]["value"], truth_raw)
            except Exception as e:
                # SUTime failed or returned empty - check if we should try DSPy
                print(f"SUTime failed or returned empty for: {e}")

                try:
                    result = temporal_extractor(text=f"Extract date from: {truth_raw}")
                    dspy_truth = json.loads(result.temporal_json)
                    if dspy_truth and dspy_truth[0].get("normalized_value"):
                        parsed_result = (
                            col,
                            dspy_truth[0]["normalized_value"],
                            truth_raw,
                        )
                        if self.verbose:
                            print(
                                f"DSPy extracted from truth: {dspy_truth[0]['normalized_value']}"
                            )
                except Exception as e:
                    if self.verbose:
                        print(f"DSPy didn't find temporal in '{truth_raw}': {e}")

            if parsed_result:
                valid_ground_truth.append(parsed_result)

        # Evaluation cases:
        if not valid_ground_truth:
            if not extracted_dates:
                return (
                    {
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "interpretation": "correct_no_temporal",
                    },
                    [],
                    masked_out,
                )
            else:
                return (
                    {
                        "precision": 0.0,
                        "recall": None,
                        "f1": None,
                        "interpretation": "false_temporal_extractions",
                    },
                    [
                        f"extracted {len(extracted_dates)} temporal elements when none expected"
                    ],
                    masked_out,
                )

        if not extracted_dates:
            return (
                {
                    "precision": None,
                    "recall": 0.0,
                    "f1": None,
                    "interpretation": "missed_all_temporal",
                },
                [f"missed {len(valid_ground_truth)} temporal elements"],
                masked_out,
            )

        # Match extracted vs truth
        matches_per_extraction = {}
        matched_truths = 0
        # print("--" * 10)
        for col, truth_date_obj, truth_raw in valid_ground_truth:
            if self.verbose:
                print(f"Comparing truth: {col}: {truth_raw} -> {truth_date_obj}")

            found_match = False
            for i, extracted_date in enumerate(extracted_dates):
                # if i in used_extractions:
                #     continue
                # NO SKIP for extractions!!!!

                extracted_date_obj = extracted_date["value"]
                extracted_date_obj = concretise_timex(extracted_date_obj)

                if self._compare_temporal(extracted_date_obj, truth_date_obj):
                    # used_extractions.add(i)
                    matched_truths += 1
                    found_match = True

                    # Track usage for stats:
                    matches_per_extraction[i] = matches_per_extraction.get(i, 0) + 1

                    if self.verbose:
                        print(f"Matched: {extracted_date_obj} -> {truth_date_obj}")
                    break

            if not found_match:
                fails.append(f"date mismatch: {col}={truth_raw}")

        # Calculate metrics
        tp = matched_truths
        # fp = len(extracted_dates) - len(used_extractions)
        fp = len(
            [i for i in range(len(extracted_dates)) if i not in matches_per_extraction]
        )
        fn = len(valid_ground_truth) - matched_truths

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return (
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "extracted_count": len(extracted_dates),
                "ground_truth_count": len(valid_ground_truth),
                "interpretation": "asymmetric_parsing",
            },
            fails,
            masked_out,
        )

    # ------------------------------------------------------------------

    def _structure_fails(
        self, num_fails: List[str], tmp_fails: List[str], rep_type: str
    ) -> Dict[str, object]:
        total_fails = len(num_fails) + len(tmp_fails)

        fail_summary = []
        if num_fails:
            fail_summary.append(f"numeric({len(num_fails)})")
        if tmp_fails:
            fail_summary.append(f"temporal({len(tmp_fails)})")

        return {
            "total_count": total_fails,
            "numeric": "; ".join(num_fails) if num_fails else None,
            "temporal": "; ".join(tmp_fails) if tmp_fails else None,
            "summary": "; ".join(fail_summary) if fail_summary else "no_fails",
        }

    # ------------------------------------------------------------------

    def run(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        results: List[Dict[str, object]] = []

        all_numeric_metrics = []
        all_temporal_metrics = []
        for idx, row in self.df_gen.iterrows():
            truth_row = self.df_orig.iloc[idx]

            for rep_type, text in row.items():
                if isinstance(text, float) and math.isnan(text):
                    continue

                cols = self.plan_map.get(rep_type, [])
                if self.verbose:
                    print(f"Num Truth cols: {len(cols)}")
                    print(f"Truth cols: {truth_row.get(cols)}")
                # Run both evaluation methods
                num_metrics, num_fails = self._check_numeric_improved(
                    rep_type, text, truth_row
                )
                tmp_metrics, tmp_fails, _ = self._check_temporal_improved_with_llm(
                    rep_type, text, truth_row
                )

                if self.verbose:
                    print("xxx" * 10)
                    print(f"Numeric for Rep type: {rep_type}: {num_metrics}")
                    print(f"Temporal for Rep type: {rep_type}: {tmp_metrics}")

                # Structure the fails more clearly
                structured_fails = self._structure_fails(num_fails, tmp_fails, rep_type)
                # Collect metrics for summary (only if we have valid data)
                if num_metrics and isinstance(num_metrics, dict):
                    all_numeric_metrics.append(num_metrics)
                if (
                    tmp_metrics
                    and isinstance(tmp_metrics, dict)
                    and tmp_metrics.get("precision") is not None
                ):
                    all_temporal_metrics.append(tmp_metrics)

                # Build the result row
                result_row = {
                    "row_idx": idx,
                    "report_type": rep_type,
                    "generated_text": (
                        text[:100] + "..." if len(str(text)) > 100 else text
                    ),  # Truncate for readability
                    # Numeric metrics:
                    "num_precision": (
                        num_metrics.get("precision", None) if num_metrics else None
                    ),
                    "num_recall": (
                        num_metrics.get("recall", None) if num_metrics else None
                    ),
                    "num_f1": num_metrics.get("f1", None) if num_metrics else None,
                    "num_tp": (
                        num_metrics.get("true_positives", None) if num_metrics else None
                    ),
                    "num_fp": (
                        num_metrics.get("false_positives", None)
                        if num_metrics
                        else None
                    ),
                    "num_fn": (
                        num_metrics.get("false_negatives", None)
                        if num_metrics
                        else None
                    ),
                    "num_interpretation": (
                        num_metrics.get("interpretation", None) if num_metrics else None
                    ),
                    # Temporal metrics
                    "tmp_precision": (
                        tmp_metrics.get("precision", None) if tmp_metrics else None
                    ),
                    "tmp_recall": (
                        tmp_metrics.get("recall", None) if tmp_metrics else None
                    ),
                    "tmp_f1": tmp_metrics.get("f1", None) if tmp_metrics else None,
                    "tmp_tp": (
                        tmp_metrics.get("true_positives", None) if tmp_metrics else None
                    ),
                    "tmp_extracted_count": (
                        tmp_metrics.get("extracted_count", None)
                        if tmp_metrics
                        else None
                    ),
                    "tmp_ground_truth_count": (
                        tmp_metrics.get("ground_truth_count", None)
                        if tmp_metrics
                        else None
                    ),
                    "tmp_interpreation": (
                        tmp_metrics.get("interpretation", None) if tmp_metrics else None
                    ),
                    # Failure analysis
                    "fail_count": structured_fails["total_count"],
                    "numeric_fails": structured_fails["numeric"],
                    "temporal_fails": structured_fails["temporal"],
                    "fail_summary": structured_fails["summary"],
                    # Placeholder for LLM evaluation (you can enable this later)
                    "llm_verified": None,  # TODO: Implement LLM verification
                }
                results.append(result_row)
                if self.verbose:
                    print("==" * 10)

        # Create the results DataFrame
        df_results = pd.DataFrame(results)
        # Calculate summary statistics
        summary_stats = _calculate_summary_statistics(
            all_numeric_metrics, all_temporal_metrics
        )
        # summary_stats = []
        return df_results, summary_stats

    def parallel_run(self, log_file) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Parallel version of run() - will replace the original"""
        results: List[Dict[str, object]] = []
        all_numeric_metrics = []
        all_temporal_metrics = []

        # Process single row-report combination:
        def process_item(args):
            idx, rep_type, text, truth_row = args
            with capture_row():
                ROW_LOG.info("row-%d : start", idx)
                if isinstance(text, float) and math.isnan(text):
                    return None

                try:

                    num_metrics, num_fails = self._check_numeric_improved(
                        rep_type, text, truth_row
                    )
                    tmp_metrics, tmp_fails, masked_out = (
                        self._check_temporal_improved_with_llm(
                            rep_type, text, truth_row
                        )
                    )

                    # Structre that fails:
                    structured_fails = self._structure_fails(
                        num_fails, tmp_fails, rep_type
                    )

                    # build the result row (same as the serial one)
                    result_row = {
                        "row_idx": idx,
                        "report_type": rep_type,
                        "generated_text": (
                            text[:100] + "..." if len(str(text)) > 100 else text
                        ),
                        # Numeric metrics
                        "num_precision": (
                            num_metrics.get("precision", None) if num_metrics else None
                        ),
                        "num_recall": (
                            num_metrics.get("recall", None) if num_metrics else None
                        ),
                        "num_f1": num_metrics.get("f1", None) if num_metrics else None,
                        "num_tp": (
                            num_metrics.get("true_positives", None)
                            if num_metrics
                            else None
                        ),
                        "num_fp": (
                            num_metrics.get("false_positives", None)
                            if num_metrics
                            else None
                        ),
                        "num_fn": (
                            num_metrics.get("false_negatives", None)
                            if num_metrics
                            else None
                        ),
                        "num_interpretation": (
                            num_metrics.get("interpretation", None)
                            if num_metrics
                            else None
                        ),
                        # Temporal metrics
                        "tmp_precision": (
                            tmp_metrics.get("precision", None) if tmp_metrics else None
                        ),
                        "tmp_recall": (
                            tmp_metrics.get("recall", None) if tmp_metrics else None
                        ),
                        "tmp_f1": tmp_metrics.get("f1", None) if tmp_metrics else None,
                        "tmp_tp": (
                            tmp_metrics.get("true_positives", None)
                            if tmp_metrics
                            else None
                        ),
                        "tmp_extracted_count": (
                            tmp_metrics.get("extracted_count", None)
                            if tmp_metrics
                            else None
                        ),
                        "tmp_ground_truth_count": (
                            tmp_metrics.get("ground_truth_count", None)
                            if tmp_metrics
                            else None
                        ),
                        "tmp_interpreation": (
                            tmp_metrics.get("interpretation", None)
                            if tmp_metrics
                            else None
                        ),
                        # Failure analysis
                        "fail_count": structured_fails["total_count"],
                        "numeric_fails": structured_fails["numeric"],
                        "temporal_fails": structured_fails["temporal"],
                        "fail_summary": structured_fails["summary"],
                        "llm_verified": None,
                    }

                    ROW_LOG.info("row-%d : done", idx)

                    return result_row, num_metrics, tmp_metrics
                except Exception as e:
                    print(f"Error processing row {idx}, report {rep_type}: {str(e)}")
                    if self.verbose:
                        import traceback

                        traceback.print_exc()
                    return None

        # Prepare all work items
        work_items = []
        for idx, row in self.df_gen.iterrows():
            truth_row = self.df_orig.iloc[idx]
            for rep_type, text in row.items():
                work_items.append((idx, rep_type, text, truth_row))
        # Now process in paralale:
        print(f"Processing {len(work_items)} items with {max_workers} workers ...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # # submit all the jobs:
            # futures = [executor.submit(process_item, item) for item in work_items]

            # # Collect the results with progress bar:
            # for future in tqdm(
            #     as_completed(futures), total=len(futures), desc="processing"
            # ):
            ###########################################################################
            ## Use Pool.map i.e. parallel run but sequentail write so you get the output in the same order you inputted
            results_iter = executor.map(process_item, work_items)
            for i, result in enumerate(
                tqdm(results_iter, total=len(work_items), desc="processing")
            ):
                try:
                    # result = future.result()
                    if result is not None:

                        result_row, num_metrics, tmp_metrics = result

                        results.append(result_row)
                        if num_metrics and isinstance(num_metrics, dict):
                            all_numeric_metrics.append(num_metrics)
                        if (
                            tmp_metrics
                            and isinstance(tmp_metrics, dict)
                            and tmp_metrics.get("precision") is not None
                        ):
                            all_temporal_metrics.append(tmp_metrics)

                except Exception as e:
                    print(f"Future failed: {e}")

            # Sort results to maintain order:
            results.sort(key=lambda x: (x["row_idx"], x["report_type"]))

            df_results = pd.DataFrame(results)

            summary_stats = _calculate_summary_statistics(
                all_numeric_metrics, all_temporal_metrics
            )
            return df_results, summary_stats



# %% [markdown]
# # Parallel Processing

# %%
def process_unit_time_module(
    dataset_name,
    ground_truth_data,
    report_types_data,
    generated_reports_data,
    split_type,
    log_filename,
    rg_model_name,
    verbose=True,
):
    """Process a single dataset for LLM temporal numeric evaluation."""

    meta_csv_path = (
        Path(output_path).parent / f"meta_data_{rg_model_name}_{folder_name}.csv"
    )
    with filelock.FileLock(f"{output_lock_dir}/metadata.lock"):
        if run_from_localdir:
            meta_csv_path = (
                Path(output_path).parent
                / f"meta_data_{rg_model_name}_{folder_name}.csv"
            )
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
            meta_df.to_csv(meta_csv_path, index=False)
            # And change the rg_model_name to the default: rg_model_name = "Qwen2_5-72B-Instruct"

    try:
        # Get the row for this dataset
        dataset_row = meta_df[meta_df["dataset_name"] == dataset_name]
        if len(dataset_row) == 0:
            raise ValueError(f"Dataset '{dataset_name}' not found in metadata.")
        if len(dataset_row) > 1:
            raise ValueError(
                f"Multiple entries for dataset '{dataset_name}' in metadata."
            )

        # Get the single row
        row = dataset_row.iloc[0]
        # Define output path and column
        column_name = f"unit_time_csv_path_{out_name}"
        column_name_summary = f"unit_time_summary_{out_name}"

        output_csv = Path(eval_reports_path) / split_type
        Path(output_csv).mkdir(parents=True, exist_ok=True)

        output_csv = (
            output_csv / f"{dataset_name}_{rg_model_name}_unit_time_{out_name}.csv"
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

        if run_from_localdir:
            # Read data
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
            if randomize:
                sampled_indices = random.sample(range(total_rows), ROW_SAMPLE_SIZE)
                sampled_indices.sort()
            else:
                sampled_indices = list(range(0, ROW_SAMPLE_SIZE))

            orig_df_sampled = orig_df.iloc[sampled_indices].reset_index(drop=True)
            generated_df_sampled = generated_df.iloc[sampled_indices].reset_index(
                drop=True
            )
            print(f"Sampled {len(sampled_indices)} rows from {total_rows} total rows")
        else:
            # Use all the rows
            orig_df_sampled = orig_df
            generated_df_sampled = generated_df
            sampled_indices = list(range(total_rows))
            print(f"Processing all {total_rows} rows (no sampling)")

        evaluator = UnitTimeAccuracyEvaluator(
            original_csv=orig_df_sampled,
            plan_csv=planned_df,
            generated_csv=generated_df_sampled,
            numeric_tol=1e-3,
            temporal_tol_days=3,
            temporal_extractor=TemporalExtractor,
            verbose=verbose,
        )

        if exec_parallel_run:
            report_scores, summary = evaluator.parallel_run(log_filename)
        else:
            report_scores, summary = evaluator.run()

        report_scores.to_csv(output_csv, index=False)
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
            if column_name_summary not in meta_df.columns:
                meta_df[column_name_summary] = None  # Initialize with None for all rows

            # Update the metadata
            meta_df.at[row_idx, column_name] = str(output_csv)
            meta_df.at[row_idx, column_name_summary] = str(summary)

            # Save the metadata
            meta_df.to_csv(meta_csv_path, index=False)
    finally:
        # ALways restor stdout and close file:
        # sys.stdout = original_stdout
        # log_file.close()
        print(f"Debug log saved to : {log_filename}")
    return evaluator, report_scores, summary



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

            log_filename = f"{eval_reports_path}/{split_type}/{dataset_name}_{rg_model_name}_unit_time_{out_name}_debug.log"
            listener, print = setup_logging(log_filename, output_stream_to_console)

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

                process_unit_time_module(
                    dataset_name,
                    ground_truth_data,
                    report_types_data,
                    generated_reports_data,
                    split_type,
                    log_filename,
                    rg_model_name,
                    verbose=is_verbose,
                )

            listener.stop()

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
            log_filename = f"{eval_reports_path}/{split_type}/{dataset_name}_{rg_model_name}_unit_time_{out_name}_debug.log"
            listener, print = setup_logging(log_filename, output_stream_to_console)
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

                process_unit_time_module(
                    dataset_name,
                    ground_truth_data,
                    report_types_data,
                    generated_reports_data,
                    split_type,
                    log_filename,
                    rg_model_name,
                    verbose=is_verbose,
                )

            listener.stop()


# %% [markdown]
# # Shutdown SUTime client 

# %%
if sutime_client is not None:
    sutime_client.stop()
    sutime_client = None



import os
import json
import random
import pandas as pd
import io
from pathlib import Path
import numpy as np
from datetime import datetime
from tqdm.notebook import tqdm
import ast
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor


# Similarity metrics
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from strsimpy.jaro_winkler import JaroWinkler
from bert_score import score as bertscore
from ortools.linear_solver import pywraplp
from datasets import load_dataset

import litellm


def get_llm_response(prompt, API_KEY, completion_args):
    """Get a response from the LLM using litellm."""
    litellm.api_key = API_KEY
    try:
        response = litellm.completion(
            messages=[{"role": "user", "content": prompt}],
            **completion_args,
        )
        return response.choices[0].message.content
    except litellm.APIError as e:
        print(f"LiteLLM API Error during get_llm_response: \n{e}")
        raise


# Prompt templates
candidate_columns_prompt = """
You are an expert in data analytics and processing, and you are planning to extract and index insights from a column containing textual content in a table.

Your task is to process the content contained in the '{target_col}' column of rows in this table. Several examples of this column are shown below:
{sample_row_content}
----
Based on the examples above, your task is to identify a set of new columns that can be extracted from the text and added to the table.

You must be selective with your choice of new columns, only suggesting new columns that can be confidently extracted from the text and which can be assigned specific values or categories.

Do not add new columns that are already covered by the existing columns in the table. Keep in mind that these new columns may be added for other rows as well, so they should be general enough to be applied to other content that might be included in the '{target_col}' column.

First, briefly describe some candidate columns that might be extracted from the passage. Wrap this step with ```think```.

Second, think through each of your potential choices to justify whether or not they should be included in your final list. Choose which, if any, of the choices should be omitted for having overlaps with existing columns or being overly specific. Wrap this step with ```revise```.

Last, output your final choice in JSON format, as {"new_columns": column_list}, where column_list is a list of the new column names. Please wrap your JSON output with ```json```.
"""

val_extraction_prompt = """
You are an expert in data analytics and processing, and you are transforming textual data into discrete columns to be inserted into a SQL database.

Your task is to extract information from the content contained in textual columns of a row, in order to fill in additional column values. You will be provided with a summary of the table, textual content for a row in the table, and a list of candidates for columns that we want to extract from the text.

First, think through your answer, and identify which column candidates appear to have relevant information present in the text. Wrap this step with ```think```.

Second, write out a draft of new column value extractions. Format this draft as a JSON dict, wrapped with ```draft```. Extracted values must be a categorical or numeric values. 

Third, revise your draft by shortening and normalizing values. Wrap this entire step with ```revise```.

Last, output a JSON dict for the extracted values, wrapped with ```json```. If a value for a particular column cannot be confidently extracted, is unknown, is not applicable, or is not specified, do not include it in the final output.

Only extract information to fill in the values from the provided list of new column candidates. Do not output any other extractions.

Textual column contents:
----
{row_content}
----

New column candidates: {new_cols}

Extraction Output:
"""


def parse_wrapped_json(text: str):
    if (
        text.find("```json") == -1
        or text.rfind("```") == -1
        or text.find("```json") == text.rfind("```")
    ):
        return None
    try:
        if text.find("```json") + 7 == text.rfind("```"):
            text_to_parse = text[text.rfind("```") + 3 :].strip()
        else:
            text_to_parse = text[text.find("```json") + 7 : text.rfind("```")]
        parsed_json = json.loads(text_to_parse)
    except (SyntaxError, TypeError, ValueError):
        parsed_json = None
    return parsed_json


# Helper function for parallel processing
def multiproc_get_resp(query_id, query, api_key, completion_args):
    try:
        res = get_llm_response(query, api_key, completion_args)
        return {query_id: res}
    except Exception as e:
        # Return an error dict instead of raising, to prevent crashing the pool
        return {query_id: {"error": f"LLM call failed for query {query_id}: {e}"}}


def identify_candidate_new_columns_to_extract(
    *, table_df, target_col, api_key, completion_args
):
    output_collection = dict()
    for tc in [target_col]:
        prompts = []
        new_sample_count = 5
        new_sample_rows = table_df[tc][:5]
        sample_row_content = "".join(
            [
                f"----\nRow {i}: {new_sample_rows.iloc[i]}\n"
                for i in range(new_sample_count)
            ]
        )
        candidate_identification_prompt = candidate_columns_prompt.replace(
            "{sample_row_content}", sample_row_content
        ).replace("{target_col}", tc)

        result = get_llm_response(
            candidate_identification_prompt, api_key, completion_args
        )
        result_json = parse_wrapped_json(result)

        if result_json is None:
            candidate_content = []
        else:
            col_content = result_json["new_columns"]
            normalized_col_text = {c.lower().replace(" ", "_") for c in col_content}
            candidate_content = list(normalized_col_text)

        # put the output into a dictionary for each column:
        output_collection[tc] = candidate_content
    return output_collection


def extract_new_columns_for_table(
    *,
    table_df,
    target_col,
    new_column_candidates,
    api_key,
    completion_args,
    pool,
):
    content_to_prompt = []
    filter_new_column_names = set()
    for i in range(len(table_df)):
        row_content_strings = []
        target_col_combined = []
        row_content = table_df.iloc[i][target_col]
        if isinstance(row_content, list):
            row_content = " ".join(row_content)
        if not row_content or pd.isna(row_content):
            continue
        row_content_strings.append(f"{target_col}: {row_content}")
        target_col_combined.append(target_col)
        all_new_cols = [f"{nc}" for nc in new_column_candidates]
        row_content_strings = "\n".join(row_content_strings)
        target_col_combined = ".".join(target_col_combined)
        col_extraction_prompt = val_extraction_prompt
        col_extraction_prompt = col_extraction_prompt.replace(
            "{row_content}", row_content_strings
        )
        col_extraction_prompt = col_extraction_prompt.replace(
            "{new_cols}", str(all_new_cols)
        )

        content_to_prompt.append(
            (
                f"{str(i)}-{target_col_combined}",
                col_extraction_prompt,
                api_key,
                completion_args,
            )
        )

        filter_new_column_names |= set(all_new_cols)

    all_results = {}
    print(
        f"Performing extractions for {len(content_to_prompt)} rows/column combinations"
    )
    # results = [pool.apply_async(multiproc_get_resp, q) for q in content_to_prompt]
    # for r in tqdm(results):
    #     result_content = r.get()
    #     all_results |= result_content

    futures = [pool.submit(multiproc_get_resp, *q) for q in content_to_prompt]
    for future in tqdm(futures):
        result_content = future.result()
        all_results |= result_content

    for query_id, llm_resp in all_results.items():
        if isinstance(llm_resp, dict) and "error" in llm_resp:
            print(llm_resp["error"])  # Print the underlying API error
            extracted_cols = None
        else:
            extracted_cols = parse_wrapped_json(llm_resp)

        if extracted_cols is not None:
            extracted_cols = {
                k: v for k, v in extracted_cols.items() if k in filter_new_column_names
            }

        all_results[query_id] = extracted_cols
    return all_results


# Score column extraction
def score_col_extraction(GT, Pred, metric_type):
    score_matrix = compute_score_matrix(GT, Pred, metric_type)
    gt_pred_mapping, score = solve_optimal_mapping(
        score_matrix, GT, Pred, debug_solver=True
    )
    recall = extraction_recall(score, GT)
    precision = extraction_precision(score, Pred)
    # print(f"Precision: {precision:.3}, Recall: {recall:.3}")
    return gt_pred_mapping, recall, precision


# Compute score matrix
def compute_score_matrix(GT, Pred, metric):
    metric_sim = scoring_metric(metric)

    score_matrix = []
    for gt_col in GT:
        score_list = []
        for pred_col in Pred:
            score_list.append(metric_sim(gt_col, pred_col))
        score_matrix.append(score_list)
    return score_matrix


# Solve optimal mapping
def solve_optimal_mapping(score_matrix, gt_cols, pred_cols, debug_solver: bool = False):
    solver = pywraplp.Solver.CreateSolver("SCIP")

    var_list = []
    to_gt_constraints = []
    objective_terms = []
    p_list = [[] for _ in range(len(score_matrix[0]))]
    for i in range(len(score_matrix)):
        c_list = []
        for j, scoreval in enumerate(score_matrix[i]):
            new_var = solver.IntVar(0, 1, "")
            var_list.append((new_var, i, j))
            c_list.append(new_var)
            p_list[j].append(new_var)
            objective_terms.append(new_var * scoreval)
        solver.Add(sum(c_list) <= 1)
    for pl in p_list:
        solver.Add(sum(pl) <= 1)
    solver.Maximize(sum(objective_terms))

    status = solver.Solve()

    if debug_solver:
        if status == pywraplp.Solver.INFEASIBLE:
            print("INFEASIBLE")
        elif status == pywraplp.Solver.FEASIBLE:
            print("FEASIBLE")
        # elif status == pywraplp.Solver.OPTIMAL:
        # print("Solution:")
        # for var, gt_ind, pred_ind in var_list:
        #     if var.solution_value() == 1:
        #         print(f"Map {gt_cols[gt_ind]} to {pred_cols[pred_ind]}")

        # print(f"ObjectiveValue: {solver.Objective().Value():.5}")
    soln_gt_to_pred = {}
    for var, gt_ind, pred_ind in var_list:
        if var.solution_value() == 1:
            soln_gt_to_pred[gt_cols[gt_ind]] = pred_cols[pred_ind]
    soln_total_score = solver.Objective().Value()
    return soln_gt_to_pred, soln_total_score


# Calculate precision and recall
def extraction_precision(score, pred_cols):
    if len(pred_cols) == 0:
        return float("nan")
    return score / len(pred_cols)


def extraction_recall(score, gt_cols):
    if len(gt_cols) == 0:
        return float("nan")
    return score / len(gt_cols)


# Score row-level value extraction
def calc_row_level_prec_rec(
    GT_df, pred_data_dict, target_col, GT_columns, gt_pred_mapping, metric
):
    metric_sim = scoring_metric(metric)

    row_prec = []
    row_rec = []
    for index, row in GT_df.iterrows():
        row_score = 0
        pred_key = f"{index}-{target_col}"
        pred_row = pred_data_dict.get(pred_key, None)
        # pred_row = pred_data_dict[f"{index}-{target_col}"]
        if pred_row is None:
            print(
                f"Warning: No prediction found for key '{pred_key}'. Skipping this row. {pred_row}"
            )
            # continue
            pred_row = {}

        for gtc in GT_columns:
            # this scoring currently doesn't properly consider cases when the original data has null/empty values
            predc = gt_pred_mapping.get(gtc, None)
            if predc is None:
                continue
            pred_val = pred_row.get(predc, None)
            if pred_val is None:
                continue

            # row_score += metric_sim(row[gtc], pred_val)
            # Handle similarity metrics when both are floats/ Not the most ideal way but ok for now
            # print(f"GT value: {row[gtc]} (type: {type(row[gtc])})")
            # print(f"Pred value: {pred_val} (type: {type(pred_val)})")
            row_score += metric_sim(str(row[gtc]), str(pred_val))

        num_pred_cols = len(pred_row.keys())
        num_gt_cols = len(GT_columns)
        try:
            row_precision_score = row_score / num_pred_cols
            row_recall_score = row_score / num_gt_cols
            row_prec.append(row_precision_score)
            row_rec.append(row_recall_score)
        except ZeroDivisionError as zde:
            print(
                f"Zero division at {pred_key}: len(pred_row)={num_pred_cols}, len(GT_columns)={num_gt_cols}"
            )
            # Append default values or skip
            continue

    # Handle cases where all the rwos were skipped ude to errors:
    if not row_prec or not row_rec:
        avg_prec = 0.0
        avg_rec = 0.0
    else:
        avg_prec = np.mean(row_prec)
        avg_rec = np.mean(row_rec)

    print(f"Average precision: {avg_prec:.3}")
    print(f"Average recall: {avg_rec:.3}")

    return row_rec, row_prec, avg_rec, avg_prec


# Scoring metric
def scoring_metric(metric):
    if metric == "levenshtein":
        normalized_levenshtein = NormalizedLevenshtein()
        metric_sim = lambda x, y: 1 - normalized_levenshtein.distance(x, y)
    elif metric == "jaro-winkler":
        jaro_winkler = JaroWinkler()
        metric_sim = lambda x, y: jaro_winkler.similarity(x, y)
    elif metric == "bertscore":
        metric_sim = lambda x, y: bertscore([x], [y], lang="en")[2].item()
    elif metric == "exact":
        metric_sim = lambda x, y: 1 if x.lower() == y.lower() else 0
    else:
        print("Not implemented")
        return None
    return metric_sim


def process_target_columns(
    *,
    target_cols,
    planned_df,
    orig_df,
    generated_df,
    max_workers,
    api_key,
    completion_args,
):

    results = {
        "Report_type": [],
        "Truth_columns": [],
        "Predicted_columns": [],
        "Levenshtein_Precision": [],
        "Levenshtein_Recall": [],
        "Jaro_Winkler_Precision": [],
        "Jaro_Winkler_Recall": [],
        "Exact_Match_Precision": [],
        "Exact_Match_Recall": [],
        "Row_Level_Levenshtein_Precision": [],
        "Row_Level_Levenshtein_Recall": [],
    }

    # with ThreadPool(max_workers) as pool:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for target in target_cols:
            print(target)
            pred_cols = identify_candidate_new_columns_to_extract(
                table_df=generated_df,
                target_col=target,
                api_key=api_key,
                completion_args=completion_args,
            )
            pred_cols = list(pred_cols.values())[0]  # Extract the list of columns
            GT_columns = planned_df[[target]].iloc[0].item()
            GT_columns = ast.literal_eval(GT_columns)

            # Score table-level column extraction
            print(f"GT Columns: {GT_columns}")
            print(f"Predictions: {pred_cols}")
            results["Report_type"].append(target)
            results["Truth_columns"].append(GT_columns)
            results["Predicted_columns"].append(pred_cols)

            # print("---")
            # print("Levenshtein")
            col, levenshtein_prec, levenshtein_recall = score_col_extraction(
                GT_columns, pred_cols, "levenshtein"
            )
            # print("---")
            # print("Jaro-winkler")
            col, jaro_winkler_prec, jaro_winkler_recall = score_col_extraction(
                GT_columns, pred_cols, "jaro-winkler"
            )
            # print("---")
            # print("Exact Match")
            exact_gt_pred_mapping, exact_match_prec, exact_match_recall = (
                score_col_extraction(GT_columns, pred_cols, "exact")
            )
            # print("---")

            # Extract new columns for the table
            extraction_res = extract_new_columns_for_table(
                table_df=generated_df,
                target_col=target,
                new_column_candidates=pred_cols,
                api_key=api_key,
                completion_args=completion_args,
                pool=pool,
            )
            # print(extraction_res)
            GT_data = orig_df[GT_columns]

            # Calculate row-level precision and recall
            (
                row_recall,
                row_precision,
                avg_levenshtein_recall,
                avg_levenshtein_precision,
            ) = calc_row_level_prec_rec(
                GT_data,
                extraction_res,
                target,
                GT_columns,
                exact_gt_pred_mapping,
                "levenshtein",
            )

            # # Append results to the dictionary
            results["Levenshtein_Precision"].append(levenshtein_prec)
            results["Levenshtein_Recall"].append(levenshtein_recall)
            results["Jaro_Winkler_Precision"].append(jaro_winkler_prec)
            results["Jaro_Winkler_Recall"].append(jaro_winkler_recall)
            results["Exact_Match_Precision"].append(exact_match_prec)
            results["Exact_Match_Recall"].append(exact_match_recall)
            results["Row_Level_Levenshtein_Precision"].append(avg_levenshtein_precision)
            results["Row_Level_Levenshtein_Recall"].append(avg_levenshtein_recall)

    return results


def process_dataset(
    *,
    dataset_name,
    split_type,
    ground_truth_data,
    report_types_data,
    generated_reports_data,
    folder_name,
    output_path,
    run_from_localdir,
    out_name,
    eval_reports_path,
    rg_model_name,
    error_log,
    API_KEY,
    completion_args,
    ROW_SAMPLE_SIZE,
    ROW_RANDOM_SEED,
    max_workers,
):
    """Process a single dataset for KV extraction evaluation."""

    # Get the row for this dataset
    meta_csv_path = (
        Path(output_path).parent / f"meta_data_{rg_model_name}_{folder_name}.csv"
    )
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
    try:
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
        kv_column_name = f"kv_extraction_quality_{out_name}"
        output_path = eval_reports_path / split_type
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_path = (
            output_path
            / f"{dataset_name}_{rg_model_name}_metrics_extract_{out_name}.csv"
        )

        # Skip if already processed
        existing_path = row.get(kv_column_name, pd.NA)
        if (
            not pd.isna(existing_path)
            and os.path.exists(existing_path)
            and os.path.getsize(existing_path) > 0
        ):
            print(f"Skipping KV extraction for {dataset_name} - already processed")
            return
        # Read data

        if run_from_localdir:
            original_csv = row["ground_truth_csv_path"]
            planned_csv = row["report_types_csv_path"]
            generated_csv = row["generated_reports_csv_path"]
            print(f"Processing KV extraction for {dataset_name}")
            print(f"Loading data from: {original_csv}, {planned_csv}, {generated_csv}")
            orig_df = pd.read_csv(original_csv)
            planned_df = pd.read_csv(planned_csv)
            generated_df = pd.read_csv(generated_csv)

        else:
            # process from the hf datasets. The dataset_row should have that info:
            orig_df = pd.read_csv(io.StringIO(ground_truth_data["csv_text"]))
            planned_df = pd.read_csv(io.StringIO(report_types_data["csv_text"]))
            generated_df = pd.read_csv(io.StringIO(generated_reports_data["csv_text"]))

        # Identify target columns
        target_cols = list(planned_df.columns)

        total_rows = len(generated_df)
        if ROW_SAMPLE_SIZE is not None and total_rows > ROW_SAMPLE_SIZE:
            random.seed(f"{ROW_RANDOM_SEED}_{dataset_name}")
            sampled_indices = random.sample(range(total_rows), ROW_SAMPLE_SIZE)
            sampled_indices.sort()
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

        # Process using target columns function
        results = process_target_columns(
            target_cols=target_cols,
            planned_df=planned_df,
            orig_df=orig_df_sampled,
            generated_df=generated_df_sampled,
            api_key=API_KEY,
            completion_args=completion_args,
            max_workers=max_workers,
        )

        # Add sampling metadata
        sampling_info = {
            "dataset_name": dataset_name,
            "total_rows": len(generated_df),
            "sampled_rows": (
                min(ROW_SAMPLE_SIZE, len(generated_df))
                if ROW_SAMPLE_SIZE
                else len(generated_df)
            ),
            "sampling_method": (
                "random"
                if ROW_SAMPLE_SIZE and len(generated_df) > ROW_SAMPLE_SIZE
                else "full"
            ),
            "sample_seed": (
                f"{ROW_RANDOM_SEED}_{dataset_name}" if ROW_SAMPLE_SIZE else None
            ),
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save sampling info
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(sampling_info, f, indent=2)

        # Convert results to DataFrame and save
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)

        print(f"Results saved to {output_path}")
        # Update metadata with a file lock to prevent race conditions
        # Ensure the dataset still exists
        if dataset_name not in meta_df["dataset_name"].values:
            print(
                f"Warning: Dataset {dataset_name} no longer in metadata. Skipping update."
            )
            return

        # Find the row again (index might have changed)
        row_idx = meta_df[meta_df["dataset_name"] == dataset_name].index[0]

        # Update the metadata
        meta_df.at[row_idx, kv_column_name] = str(output_path)
        # Save the metadata
        meta_df.to_csv(meta_csv_path, index=False)
        print(f"Metadata updated for {dataset_name}")

    except Exception as e:
        error_message = f"Error processing dataset {dataset_name}: {str(e)}"
        print(error_message)
        with open(error_log, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {error_message}\n")

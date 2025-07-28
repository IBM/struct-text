import os
import io
import threading
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import dspy
import pandas as pd
import filelock
from tqdm import tqdm


# Granular Columns: This method first looked at the combined the `report_types` and the `column_names` to control the eval as well.
class ReportPlannerWithSamples(dspy.Signature):
    """
    Given a list of available columns and sample data as a python list of dictionaries, suggest insightful report types for analysis or summarization.
    For EACH suggested report type, specify columns from the original list to generate that specific report.
    Ensure related columns (e.g., units, dates accompanying metrics) are included where appropriate for each report type.
    """

    sample_list: List[Dict[str, Any]] = dspy.InputField(
        desc="A list of sample data as dictionary showing example values for a few subset samples of the column names to help understand how to aggregate the report types"
    )
    column_list: str = dspy.InputField(
        desc="A string containing a list of all available column names."
    )
    planned_reports: List[Dict[str, List[str]]] = dspy.OutputField(
        desc="""Respond ONLY with a Python list of dictionaries using the few shot examples of how the rows of the data look.
EXAMPLE OUTPUTS: Here are a few representative examples of key:value pairs of report_type and column names 

[{'Publication Overview': ['ArticleLabel', 'ArticleType', 'ArticleTitle', 'JournalName', 'JournalVolume', 'JournalIssue', 'PublicationDate']}, {'Citation Analysis': ['ArticleLabel', 'ArticleTitle', 'CitedWorkTitle', 'SubjectArea', 'AuthorReference']}, {'Subject Area Trends': ['ArticleLabel', 'ArticleTitle', 'SubjectArea', 'PublicationDate']}]
[{'Regulation_Summary_Report': ['InstrumentTitle', 'InstrumentType', 'OfficialTitle', 'ApplicableCountry', 'DateOfPublication', 'Jurisdiction', 'Series']}, {'Authorship_Report': ['InstrumentTitle', 'InstrumentAuthor', 'DateOfPublication']}, {'Foundational_Text_Report': ['InstrumentTitle', 'RelatedFoundationalText', 'DateOfPublication']}]
[{'Financial Performance': ['company_holding_name', 'stock_ticker', 'revenue', 'revenue_period', 'revenue_filing_date', 'revenue_unit', 'net_income', 'net_income_period', 'net_income_filing_date', 'net_income_unit', 'eps_basic', 'eps_basic_period', 'eps_basic_filing_date', 'eps_basic_unit', 'eps_diluted', 'eps_diluted_period', 'eps_diluted_filing_date', 'eps_diluted_unit']}, {'Balance Sheet': ['company_holding_name', 'stock_ticker', 'total_assets', 'total_assets_period', 'total_assets_filing_date', 'total_assets_unit', 'current_assets', 'current_assets_period', 'current_assets_filing_date', 'current_assets_unit', 'total_liabilities', 'total_liabilities_period', 'total_liabilities_filing_date', 'total_liabilities_unit', 'current_liabilities', 'current_liabilities_period', 'current_liabilities_filing_date', 'current_liabilities_unit', 'stockholders_equity', 'stockholders_equity_period', 'stockholders_equity_filing_date', 'stockholders_equity_unit']}, {'Operating Metrics': ['company_holding_name', 'stock_ticker', 'operating_income', 'operating_income_period', 'operating_income_filing_date', 'operating_income_unit', 'cash_equivalents', 'cash_equivalents_period', 'cash_equivalents_filing_date', 'cash_equivalents_unit']}]

IMPORTANT GUIDELINES: 
1. Choose between 1-5 report types based on the column structure and data patterns.
2. DO NOT force yourself to create reports if the columns don't naturally group into meaningful categories.
3. Quality over quantity - each report should be genuinely useful and not artificially created.
4. Each report should serve a distinct analytical purpose and use a coherent set of related columns.
5. Each dictionary must have the key:value with the key being the name of the report_type as key and the values as a list of associated_columns of strings representing the specific column names needed for that report type."""
    )


class ReportTextGenerator(dspy.Signature):
    """
    For each planned report provided, write a coherent detailed paragraph.
    Strictly use ONLY the 'associated_columns' specified for that specific report type when analyzing the provided data (column_KV_pairs).
    Base the paragraph ONLY on information present in the structured input for those columns. Avoid hallucinating numbers or details.
    Ensure consistency in units and dates found in the data.
    """

    column_KV_pairs: List[Dict[str, str]] = dspy.InputField(
        desc="A list of dictionaries, where each dictionary represents a data record (e.g., a row) with column names as keys and values as strings. This is the raw data."
    )
    planned_reports: List[Dict[str, List[str]]] = dspy.InputField(
        desc="The list of planned reports, where each dictionary specifies a 'report_type' and its specific 'associated_columns'."
    )

    generated_reports: List[Dict[str, str]] = dspy.OutputField(
        desc="Respond ONLY with a Python list of dictionaries. Each dictionary must have exactly key:value with the key being the report_type as key and the generated_text as values in string format."
    )


def process_report_generation(
    *,
    data_row: Dict[str, Any],
    split_type: str,
    output_path: str,
    out_name: str,
    folder_name: str,
    output_lock_dir: str,
    force_reprocess: bool,
    report_planner: dspy.Module,
    report_text_generator: dspy.Module,
    ROW_SAMPLE_SIZE: int,
    ROW_RANDOM_SEED: int,
    randomize: bool,
    max_workers: int,
):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Define the output path for the meta CSV file
    meta_csv_path = Path(output_path).parent / f"meta_data_{out_name}_{folder_name}.csv"
    # in this dataset itwill always have the _ground_truth.csv pattern in the filename
    csv_file = data_row["file_name"].replace("_ground_truth.csv", ".csv")
    dataset_name = Path(csv_file).stem

    # Initialize a list to store the meta data
    meta_data = []

    error_log = f"{output_path}/{split_type}/report_generator_errors_{out_name}.txt"

    # Check if the meta CSV file already exists
    lock_path = f"{output_lock_dir}/report_metadata.lock"
    with filelock.FileLock(lock_path, timeout=10):  # Add a timeout
        try:
            if meta_csv_path.exists():
                meta_df = pd.read_csv(meta_csv_path)
            else:
                meta_df = pd.DataFrame(
                    columns=[
                        "dataset_name",
                        "ground_truth_csv_path",
                        "report_types_csv_path",
                        "generated_reports_csv_path",
                        "generation_model",
                    ]
                )
        except filelock.Timeout:
            print(f"Could not acquire lock for {lock_path} within the timeout period.")
            return

    # convert the original csv file ground_truth into a pandas df:
    output_filename = dataset_name + f"_generated_reports_{out_name}.csv"
    output_file_path = Path(output_path) / split_type
    Path(output_file_path).mkdir(parents=True, exist_ok=True)
    output_file_path = output_file_path / output_filename

    planned_reports_filename = dataset_name + f"_report_types_{out_name}.csv"
    planned_reports_filepath = Path(output_path) / split_type / planned_reports_filename

    # check if the dataset exists and has already been processed with the given model:
    matching_row = meta_df[
        (meta_df["dataset_name"] == dataset_name)
        & (meta_df["generation_model"] == out_name)
    ]
    # Check if the out_name in the matching row matches the current out_name
    if not matching_row.empty and not force_reprocess:
        print(
            f"Skipping {dataset_name} as it already exists in the meta CSV for {out_name}"
        )
        return

    try:
        wiki_df = pd.read_csv(io.StringIO(data_row["csv_text"]))

    except pd.errors.EmptyDataError as e:
        print(f"Error reading in {data_row['csv_text']}: {str(e)}")
        return

    # sample rows for a subset run:
    total_rows = len(wiki_df)
    if ROW_SAMPLE_SIZE is not None and total_rows > ROW_SAMPLE_SIZE:
        # Random can take unique strings.
        random.seed(f"{ROW_RANDOM_SEED}_{dataset_name}")

        if randomize:
            sampled_indices = random.sample(range(total_rows), ROW_SAMPLE_SIZE)
            sampled_indices.sort()
        else:
            sampled_indices = list(range(0, ROW_SAMPLE_SIZE))

        orig_df_sampled = wiki_df.iloc[sampled_indices].reset_index(drop=True)
        print(f"Sampled {len(sampled_indices)} rows from {total_rows} total rows")
    else:
        orig_df_sampled = wiki_df

    # Always sample a small number of rows for the planner to avoid context overflow.
    if len(orig_df_sampled) > 10:
        sampled_rows = orig_df_sampled.iloc[:10]
    else:
        sampled_rows = orig_df_sampled

    list_of_dict = sampled_rows.to_dict(orient="records")
    all_planned_reports = report_planner(
        sample_list=list_of_dict, column_list=wiki_df.columns
    ).planned_reports

    print(f"{all_planned_reports}\n")

    # Given the column names and report types generate it for all the rows:
    # Initialize a list of size equal to orig_df's length, filled with None or placeholders.
    all_data = [None] * len(orig_df_sampled)

    #############################################################
    def process_report_generator(idx):
        try:
            # run the individual idx run from the loop:
            row = orig_df_sampled.iloc[idx]
            result = report_text_generator(
                column_KV_pairs=row.to_dict(), planned_reports=all_planned_reports
            )
            combined_data = {}
            # Append generated reports to the combined data:
            for report in result.generated_reports:
                for key, value in report.items():
                    combined_data[key] = value
            return (
                idx,
                # result,
                combined_data,
                None,
            )  # Return (row_idx, result, None for no result, error)

        except Exception as e:
            print(f"Error processing {dataset_name}, {idx}: {str(e)}")
            return idx, {}, str(e)

    #############################################################
    # use the threadpool executor to parallelize the processing:
    result_lock = threading.Lock()  # threadsafe for updating the results;
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit all the tasks to the executor:
        future_to_idx = {
            executor.submit(process_report_generator, idx): idx
            for idx in range(len(orig_df_sampled))
        }

        with tqdm(
            total=len(orig_df_sampled),
            desc=f"Procssing rowise report for {dataset_name}",
        ) as pbar:
            # process completed task as they finish (not neccarily in order)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    # get the result:
                    idx, combined_data, error = future.result()
                    # handle the succesfuly processing:
                    if combined_data is not None:
                        with result_lock:
                            all_data[idx] = combined_data
                    if error is not None:
                        with open(error_log, "a") as f:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            f.write(f"{timestamp} - {idx} - {error}\n")
                except Exception as e:
                    print(f"Error Handling result for {idx}: {str(e)}")
                pbar.update(1)

    ############################################
    output_df = pd.DataFrame(all_data)
    output_df.to_csv(output_file_path, index=False)
    print(f"Generated reports saved to: {output_file_path}")

    # Flatten the data:
    flattened_data = {}
    for report in all_planned_reports:
        for key, values in report.items():
            flattened_data[key] = [values]

    planned_df = pd.DataFrame(flattened_data)
    planned_df.to_csv(planned_reports_filepath, index=False)

    print(f"Planned reports saved to: {planned_reports_filepath}")

    # also write it out because you have read this directly from the hugging face repo.
    #  This will ensure replication of the original folder structure
    ground_truth_path = Path(output_path) / split_type / data_row["file_name"]
    # wiki_df.to_csv(ground_truth_path)
    # We only save those that are sampled from. Because that's what you want to analyze in the pipeline.
    # Doens't make sense to save erything.
    orig_df_sampled.to_csv(ground_truth_path)

    print("--" * 10)
    print(f"{output_file_path.stem}: Exist={os.path.isfile(output_file_path)}")
    # Collect meta data for the current CSV file
    # Update metadata with a file lock to prevent race conditions
    with filelock.FileLock(lock_path, timeout=10):  # Add a timeout
        try:
            # Re-read metadata to get the latest version
            if meta_csv_path.exists():
                meta_df = pd.read_csv(meta_csv_path)
            else:
                meta_df = pd.DataFrame(
                    columns=[
                        "dataset_name",
                        "ground_truth_csv_path",
                        "report_types_csv_path",
                        "generated_reports_csv_path",
                        "generation_model",
                    ]
                )
            meta_data.append(
                {
                    "dataset_name": Path(dataset_name).stem,
                    "ground_truth_csv_path": ground_truth_path,
                    "report_types_csv_path": planned_reports_filepath,
                    "generated_reports_csv_path": output_file_path,
                    "generation_model": out_name,
                }
            )

            # Append new meta data to the existing meta CSV file
            new_meta_df = pd.DataFrame(meta_data)
            meta_df = pd.concat([meta_df, new_meta_df], ignore_index=True)
            meta_df.to_csv(meta_csv_path, index=False)
        except filelock.Timeout:
            print(f"Could not acquire lock for {lock_path} within the timeout period.")
            return

    print(f"Meta data saved to: {meta_csv_path}")

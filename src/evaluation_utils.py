from __future__ import annotations

import os
import datetime as dt
from datetime import timedelta
from typing import Optional

import pint
import stanza
from stanza.server import CoreNLPClient, StartServer
import socket
from contextlib import closing
import dspy
import json
import re
import warnings
import inspect
import os
import io
import sys
import time
import random
import math
import builtins


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, PydanticUndefinedAnnotation
from collections import defaultdict

import pandas as pd
import numpy as np
import pint
import quantulum3
import dateparser
import parsedatetime
import dateutil
from dateutil import parser as duparser
import datetime as dt
from datetime import timedelta
from stanza.server import CoreNLPClient
import stanza
import spacy
import timexy

# LangChain and DSPy
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import dspy
import litellm

# Execution and Multithreading
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging, logging.handlers, queue, threading, uuid
from tqdm import tqdm
import filelock
import pickle
import subprocess
import ast


def getenv_bool(name: str, default=False):
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {val!r}")


def setup_logging(log_filename, output_stream_to_console):
    Path(log_filename).parent.mkdir(exist_ok=True, parents=True)

    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)

    log_q = queue.Queue(maxsize=10_000)
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    if output_stream_to_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        listener = logging.handlers.QueueListener(log_q, stream_handler, file_handler)
    else:
        listener = logging.handlers.QueueListener(log_q, file_handler)

    listener.start()

    csv_log = logging.getLogger()
    csv_log.setLevel(logging.INFO)
    csv_log.addHandler(logging.handlers.QueueHandler(log_q))
    csv_log.propagate = False

    patched_print = lambda *a, **k: csv_log.info(" ".join(map(str, a)))

    return listener, patched_print


def pick_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------- #
# 1.  Public-facing constants & registry
# ---------------------------------------------------------------------- #

ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

# Default properties passed to CoreNLP
_STANZA_PROPS = {
    "annotators": "tokenize,ssplit,pos,lemma,ner,entitymentions",
    "ner.useSUTime": "true",
    "sutime.includeRange": "true",
    "ner.docdate.usePresent": "true",
    "ner.applyNumericClassifiers": "true",
}

# ---------------------------------------------------------------------- #
# 2.  Internal cache for the single running client
# ---------------------------------------------------------------------- #

_sutime_client: Optional[CoreNLPClient] = None


def ensure_sutime_client(
    corenlp_home: str | None = None,
    port: int = pick_free_port(),
    memory: str = "6G",
    timeout: int = 30_000,
) -> CoreNLPClient:
    """
    Lazily start a local CoreNLP server (once) and return a SUTime-enabled client.
    Call this at the top of any notebook / script that needs temporal parsing.

    Parameters
    ----------
    corenlp_home : str | None
        Path to your CoreNLP installation.  If ``None`` we look at
        ``$CORENLP_HOME`` or raise.
    port : int
        Port on localhost where the server should listen.
    memory : str
        JVM -Xmx value, e.g. "6G".
    timeout : int
        Per-request timeout in *milliseconds*.

    Returns
    -------
    stanza.server.CoreNLPClient
        A ready-to-use Stanza client whose ``annotate`` method supports SUTime.
    """
    global _sutime_client

    if _sutime_client is not None:
        return _sutime_client  # ⇐ already running

    # Resolve CoreNLP installation directory
    corenlp_home = corenlp_home or os.getenv("CORENLP_HOME")
    if corenlp_home is None:
        raise RuntimeError(
            "CoreNLP path not provided. Pass `corenlp_home=` or "
            "set $CORENLP_HOME to your installation directory."
        )
    os.environ["CORENLP_HOME"] = corenlp_home

    endpoint = f"http://localhost:{port}"

    # First call FORCE_START to spin the server up and ping it
    _sutime_client = CoreNLPClient(
        endpoint=endpoint,
        properties=_STANZA_PROPS,
        be_quiet=True,
        timeout=timeout,
        memory=memory,
        start_server=StartServer.FORCE_START,
    )
    test_text = "Test on January 1, 2025"
    result = _sutime_client.annotate(test_text)
    print("CoreNLP server started successfully")

    return _sutime_client


# ---------------------------------------------------------------------- #
# 3.  Helper for normalising SUTime TIMEX3 values
# ---------------------------------------------------------------------- #

_ANCHOR_DATE = dt.date.today()


def concretise_timex(timex: str) -> str:
    """
    Re-map SUTime relative placeholders (PRESENT_REF, PAST_REF, FUTURE_REF)
    onto real ISO dates anchored on *today*.

    Adjust the ±90-day convention if you need something else.
    """
    if timex == "PRESENT_REF":
        return _ANCHOR_DATE.isoformat()
    if timex == "PAST_REF":
        return (_ANCHOR_DATE - timedelta(days=90)).isoformat()
    if timex == "FUTURE_REF":
        return (_ANCHOR_DATE + timedelta(days=90)).isoformat()
    return timex


# ---------------------------------------------------------------------- #
# 4.  Helper for DSPY Driven prompts for Temporal parsing.
# ---------------------------------------------------------------------- #
class TemporalSpan(BaseModel):
    original_text: str
    type: Literal["date", "quarter", "year", "fiscal_period"]
    normalized_value: str
    year: Optional[int] = Field(None, ge=0)
    quarter: Optional[int] = Field(None, ge=1, le=4)
    context: Optional[str]  # Free-form


try:
    TemporalSpan.model_rebuild()
except PydanticUndefinedAnnotation as exc_info:
    print(exc_info)
    assert exc_info.code == "undefined-annotation"


class TemporalExtractor(dspy.Signature):
    """Extract and normalize temporal information from text.     ONLY extract actual dates, not numeric values that look like dates."""

    text = dspy.InputField(
        desc="Text containing dates, quarters or time periods from a variety of sources"
    )
    temporal_json: List[TemporalSpan] = dspy.OutputField(
        desc="""JSON array of temporal entities. Each entitiy should have:
                                     - original_text: the exact text from the input
                                     - type: 'date'|'quarter'|'year'|'fiscal_period')
                                     - normalized_value: standardized format (YYYY-MM-DD for dates, YYYY-QN for quarters)
                                     - year: integer year if applicable
                                     - quarter: integer 1-4 if applicable
                                     - context: any relevant context about the temporal reference

                                    Examples of valid temporal expressions:
                                    - "fourth quarter of 2024" → {"original_text": "fourth quarter of 2024", "type": "quarter", "normalized_value": "2024-Q4", "year": 2024, "quarter": 4}
                                    - "December 28, 2024" → {"original_text": "December 28, 2024", "type": "date", "normalized_value": "2024-12-28", "year": 2024}
                                    - "CY2024Q4" → {"original_text": "CY2024Q4", "type": "fiscal_period", "normalized_value": "2024-Q4", "year": 2024, "quarter": 4}
                                    - Dates: "December 28, 2024", "2024-12-28", "28/12/2024"
                                    - Holidays: "New Year's Day", "Christmas"
                                    - Quarters: "Q4 2024", "fourth quarter of 2024"
                                    - Years: "2024", "1990s"
                                    - Historical periods: "19th century", "Bronze Age"

                                    DO NOT Extract:
                                    - Decimal numbers (9.27, 13.4, 27.7)
                                    - Ages (18 years old)
                                    - Percentages (9.0%)
                                    - Index scores (0.94)
                                    - Any standalone number that isn't explicitly a year

                                    Return empty array [] if no temporal information found."""
    )


# ---------------------------------------------------------------------- #
# 5.  LLM as a judge prompts
# ---------------------------------------------------------------------- #


class FactualityEvaluator(dspy.Signature):
    """Judge factual accuracy of a generated report against the structured source.

    The model MUST:
    1. Extract each explicit claim.
    2. For each claim output Supported / Contradicted / Not‑Present.
    3. Give a single integer score 1‑5 following the rubric.
    4. Reasoning: Provide a detailed human understandable reasoning section for the score.

    Scoring rubric (select ONE integer 1–5):
    1 Fundamentally incorrect – Most claims are contradicted; the report gives a false impression of the source.
    2 Largely incorrect       – Several important claims are contradicted; the report misrepresents core information.
    3 Mixed accuracy          – A few claims have errors but the main narrative is mostly correct; minor distortions.
    4 Mostly correct          – Only minor or peripheral errors; all critical facts are faithful to the source.
    5 Fully correct           – No errors; every claim is directly supported by the source.
    """

    # source_data: Dict[str, Any] = dspy.InputField(
    #     description="Source data to evaluate against"
    # )
    # generated_report: str = dspy.InputField(description="Report text to evaluate")
    source_data: Dict[str, Any] = dspy.InputField(
        desc="The structured data column name/value pairs"
    )
    generated_report: Dict[str, Any] = dspy.InputField(
        desc="A dictionary of generated unstructured text reports as key:value corresponding to the report_type:generated_reports to evaluate"
    )
    claim_analysis: List[Dict[str, str]] = dspy.OutputField(
        description="List of claims with their support status"
    )
    score: int = dspy.OutputField(
        description="Integer score from 1-5 based on the rubric"
    )
    reasoning: str = dspy.OutputField(desc="Reasoning behind the factuality score")


class HallucinationEvaluator(dspy.Signature):
    """Detect ungrounded content relative to structured source.

    The model MUST:
    1. Extract each explicit claim. Label each claim sentence as Grounded / External‑Fact / Hallucination.
    2. Score: Apply the rubric to produce an integer score 1‑5.
    3. Reasoning: Provide a detailed human understandable reasoning section for the score.

    Scoring rubric (select ONE integer 1–5):
    1 Heavy hallucination     – The report contains numerous invented details; critical information appears fabricated.
    2 Frequent hallucination  – Multiple significant statements lack grounding; key points are unverifiable.
    3 Occasional hallucination– Some secondary details are ungrounded, but main points are supported by the source.
    4 Rare hallucination      – Only minor or peripheral details lack grounding; core content is well-supported.
    5 No hallucination        – All content is either directly grounded in the source or explicitly attributed.
    """

    source_data: Dict[str, Any] = dspy.InputField(
        desc="The structured data column name/value pairs"
    )
    generated_report: Dict[str, Any] = dspy.InputField(
        desc="A dictionary of generated unstructured text reports as key:value corresponding to the report_type:generated_reports to evaluate"
    )
    statement_analysis: List[Dict[str, str]] = dspy.OutputField(
        description="Analysis of each statement's grounding"
    )
    score: int = dspy.OutputField(
        description="Integer score from 1-5 based on the rubric"
    )
    reasoning: str = dspy.OutputField(desc="Reasoning behind the score")


class CoherenceEvaluator(dspy.Signature):
    """Assess English flow & logical ordering of the generated report.
    The model MUST:
     1. Identify any flow or logical issues.
     2. Apply the rubric to produce an integer score 1‑5.
     3. Provide brief reasoning for the score.

     Scoring rubric (select ONE integer 1–5):
     1 Incoherent  – The text is difficult to follow; ideas jump randomly; contradictory statements or broken logical flow.
     2 Poor flow   – Multiple jarring transitions; sections feel disconnected; the reader must work to follow the narrative.
     3 Acceptable  – Generally understandable but with some awkward transitions; organization could be improved.
     4 Smooth      – Clear progression of ideas with only minor flow issues; easy to follow the author's thinking.
     5 Seamless    – Exceptionally well-organized; ideas build naturally upon one another; transitions feel effortless.
    """

    generated_report: Dict[str, Any] = dspy.InputField(
        desc="A dictionary of generated unstructured text reports as key:value corresponding to the report_type:generated_reports to evaluate"
    )
    issues: List[str] = dspy.OutputField(
        description="List of identified coherence issues"
    )
    score: int = dspy.OutputField(
        description="Integer score from 1-5 based on the rubric"
    )
    reasoning: str = dspy.OutputField(description="Reasoning behind the score")


class TextQualityEvaluator(dspy.Module):
    """Comprehensive text quality evaluation module that assesses factuality,
    hallucination, and coherence of generated reports.

    This module chains together specialized evaluators to produce a complete
    quality assessment.
    """

    def __init__(self):
        super().__init__()
        # Options for incorporating the rubrics into DSPy prompts:

        # Option 1: The rubrics are already incorporated into the signature docstrings above
        self.factuality_evaluator = dspy.ChainOfThought(FactualityEvaluator)
        self.hallucination_evaluator = dspy.ChainOfThought(HallucinationEvaluator)
        self.coherence_evaluator = dspy.ChainOfThought(CoherenceEvaluator)

    def forward(
        self, source_data: Dict[str, Any], generated_report: str
    ) -> dspy.Prediction:
        """Evaluate the quality of a generated report.

        Args:
            source_data: The source data that the report should be based on
            generated_report: The text of the report to evaluate

        Returns:
            A prediction containing all evaluation metrics
        """
        factuality_eval = self.factuality_evaluator(
            source_data=source_data, generated_report=generated_report
        )

        hallucination_eval = self.hallucination_evaluator(
            source_data=source_data, generated_report=generated_report
        )

        coherence_eval = self.coherence_evaluator(generated_report=generated_report)

        return dspy.Prediction(
            source_data=source_data,
            generated_report=generated_report,
            # Factuality results
            claim_analysis=factuality_eval.claim_analysis,
            factuality_score=factuality_eval.score,
            factuality_reasoning=factuality_eval.reasoning,
            # Hallucination results
            statement_analysis=hallucination_eval.statement_analysis,
            hallucination_score=hallucination_eval.score,
            hallucination_reasoning=hallucination_eval.reasoning,
            # Coherence results
            coherence_issues=coherence_eval.issues,
            coherence_score=coherence_eval.score,
            coherence_reasoning=coherence_eval.reasoning,
            # Overall scores for easy access
            overall_quality_score=(
                factuality_eval.score + hallucination_eval.score + coherence_eval.score
            )
            / 3.0,
        )

from __future__ import annotations

import asyncio
import datetime
import os
import time
from enum import auto
from typing import Awaitable, Callable, Dict, List, Optional, Union

from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import TestResult
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.utils import get_or_create_event_loop
from gary.debug.formatter import Formatter
from pydantic import BaseModel
from strenum import UppercaseStrEnum


class RetrievalEvaluationType(UppercaseStrEnum):
    TRIAD = auto()
    FAITHFULNESS_ANSWER_RELEVANCE = auto()


class RAGEvaluator(BaseModel):
    name: str
    evaluation_type: RetrievalEvaluationType
    metrics: List[BaseMetric]
    test_results: List[TestResult] = []

    class Config:
        arbitrary_types_allowed = True

    # Public methods
    # ---

    def clean(self):
        self.test_results = []

    def run_evals(
        self,
        invoker: Union[
            Callable[[Dict], LLMTestCase],
            Callable[[Dict], LLMTestCase],
            Callable[[Dict], Awaitable[LLMTestCase]],
            Callable[[Dict], Awaitable[LLMTestCase]],
        ],
        eval_questions: List[Dict[str, str]],
        cleanup_previous_results: bool = True,
        run_async: bool = True,
        batch_size: int = 1,
        batch_interval: float = 0.2,
        store_dir: Optional[str] = None,
    ) -> List[TestResult]:
        if cleanup_previous_results:
            self.clean()

        self.test_results = [] if self.test_results is None else self.test_results
        for metric in self.metrics:
            metric.async_mode = run_async

        file_path: Optional[str] = None
        if store_dir:
            file_path = self._create_test_results_file(store_dir)

        dataset = EvaluationDataset()
        for use_case in eval_questions:
            test_case: LLMTestCase
            if asyncio.iscoroutinefunction(invoker):
                loop = get_or_create_event_loop()
                test_case = loop.run_until_complete(invoker(use_case))
            else:
                test_case = invoker(use_case)  # type: ignore

            dataset.add_test_case(test_case)

            if len(dataset.test_cases) > batch_size and file_path:
                self._evaluate(dataset, file_path)
                dataset = EvaluationDataset()
                time.sleep(batch_interval)

        # Case where not enter depending of the batch, still data
        if dataset.test_cases and file_path:
            self._evaluate(dataset, file_path)

        if file_path:
            self._write_metric_pass_rates(self.test_results, file_path)
            self._write_summary(self.test_results, file_path)

        return self.test_results

    def _evaluate(
        self,
        dataset: EvaluationDataset,
        file_path: str,
    ):
        evaluation_results = dataset.evaluate(self.metrics)
        for test_result in evaluation_results:
            self._write_test_result(test_result, file_path)

        self.test_results += evaluation_results

    def _create_test_results_file(self, store_dir: str) -> str:
        ext = ".txt"
        file_path = store_dir + "/" + self.name + ext

        # Set the initial suffix to 1
        suffix = 1

        # Loop until we find a file name that does not exist
        while os.path.isfile(file_path):
            # Increment the suffix
            suffix += 1

            # Generate a new file name with the suffix
            file_name = self.name + "_{}".format(suffix) + ext

            # Generate a new file path with the new file name
            file_path = store_dir + "/" + file_name

        # Get the current date and time
        current_datetime = datetime.datetime.now()

        # Format the date and time as a string
        current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Create the file and open it for writing
        with open(file_path, "a") as file_object:
            file_object.write(current_datetime_str)
            file_object.write("\n")
        return file_path

    def _write_summary(self, test_results: List[TestResult], file_path: str) -> None:
        # Initialize counters
        total_success = 0
        total_error = 0
        total_results = len(test_results)
        # Prepare data for the table
        for test in test_results:
            total_success += test.success

            for metric in test.metrics:
                if metric.error:
                    total_error += 1
                break

        # Calculate means
        if total_results != 0:
            mean_success = total_success / total_results
        else:
            mean_success = 0

        headers = ["Metric", "Value"]
        data = [
            ["Total tests", total_results],
            ["Mean success", "{:.2%}".format(mean_success)],
            ["Total error", total_error],
        ]
        with open(file_path, "a") as file_object:
            file_object.write("\n")
            file_object.write("Summary")
            row_format = "{:<20} {:<15}"
            if headers is not None:
                file_object.write(row_format.format(*headers) + "\n")
                file_object.write("-" * 35 + "\n")

            for row in data:
                file_object.write(row_format.format(*row) + "\n")
            file_object.write("\n" + "=" * 70 + "\n")

    def _write_test_result(self, test_result: TestResult, file_path: str) -> None:
        with open(file_path, "a") as file_object:
            file_object.write("\n")
            file_object.write("=" * 70 + "\n")
            file_object.write("\n")
            file_object.write("Metrics Summary\n")
            file_object.write("\n")
            for metric in test_result.metrics:
                successful = True
                if metric.error is not None:
                    successful = False
                else:
                    # This try block is for user defined custom metrics,
                    # which might not handle the score == undefined case elegantly
                    try:
                        if not metric.is_successful():
                            successful = False
                    except Exception:
                        successful = False

                if not successful:
                    file_object.write(
                        (
                            f"❌ {metric.__name__}:\n"
                            f"  - score: {metric.score}\n"
                            f"  - threshold: {metric.threshold}\n"
                            f"  - strict: {metric.strict_mode}\n"
                            f"  - evaluation model: {metric.evaluation_model}\n"
                            f"  - reason: {metric.reason}\n"
                            f"  - error: {metric.error}\n\n"
                        )
                    )
                else:
                    file_object.write(
                        (
                            f"✅ {metric.__name__}:\n"
                            f"  - score: {metric.score}\n"
                            f"  - threshold: {metric.threshold}\n"
                            f"  - strict: {metric.strict_mode}\n"
                            f"  - evaluation model: {metric.evaluation_model}\n"
                            f"  - reason: {metric.reason}\n"
                            f"  - error: {metric.error}\n\n"
                        )
                    )
                if metric.score_breakdown:
                    for metric_name, score in metric.score_breakdown.items():
                        file_object.write(f"      - {metric_name} (score: {score})\n")

            formatter = Formatter()
            file_object.write("\n")
            file_object.write("For test case:\n")
            file_object.write(f"  - input: {test_result.input}\n")
            file_object.write(f"  - actual output: {test_result.actual_output}\n")
            file_object.write(f"  - expected output: {test_result.expected_output}\n")
            file_object.write(f"  - context: {formatter.format(test_result.context)}\n")
            file_object.write(f"  - retrieval context: {formatter.format(test_result.retrieval_context)}\n")

    def _write_metric_pass_rates(self, test_results: List[TestResult], file_path: str) -> None:
        metric_counts = {}
        metric_successes = {}

        for result in test_results:
            for metric in result.metrics:
                metric_name = metric.__class__.__name__
                if metric_name not in metric_counts:
                    metric_counts[metric_name] = 0
                    metric_successes[metric_name] = 0
                metric_counts[metric_name] += 1
                if metric.success:
                    metric_successes[metric_name] += 1

        metric_pass_rates = {metric: (metric_successes[metric] / metric_counts[metric]) for metric in metric_counts}

        with open(file_path, "a") as file_object:
            file_object.write("\n" + "=" * 70 + "\n")
            file_object.write("\n")
            file_object.write("Overall Metric Pass Rates\n")
            for metric, pass_rate in metric_pass_rates.items():
                file_object.write(f"{metric}: {pass_rate:.2%} pass rate\n")
            file_object.write("\n" + "=" * 70 + "\n")

    # Instance
    # ---

    @classmethod
    def from_rag_triad_evaluation(
        cls,
        name: str,
        eval_llm: Union[DeepEvalBaseLLM, str],
        context_relevance_threshold=0.7,
        faithfulness_threshold=0.7,
        answer_relevance_threshold=0.7,
    ) -> RAGEvaluator:
        """
        Build RetrievalEvalBuilder for RAG triad Evaluation

        The RAG triad is made up of 3 evaluations:
        1. Contextual Recall
        2. Faithfulness
        3. Answer Relevance

        1. Contextual Recall:
        Is the retrieved context capture the context general or general information necessary to generate the expected output?

        A higher contextual recall score indicates that the retrieval system is better at capturing all general relevant information from input.
        It evaluates the retrieved context aligns with the expected output.
        Any missing or irrelevant information in the context could lead to hallucinations or incorrect responses.

        2. Faithfulness:
        Is the response supported by the context?

        Evaluating whether the Response factually aligns with the contents of the Retrieval

        After the context is retrieved, it is then formed into an answer by an LLM.
        LLMs are often prone to straying from the facts provided, exaggerating or expanding to a correct-sounding answer.

        3. Answer Relevance:
        Is the response relevant to the query?

        Evaluating how relevant the Response of the LLM is compared to the provided Input

        Lastly, the response still needs to helpfully answer the original question.
        Evaluating the relevance of the final response to the user input.

        Args:
        - name (str): Name of the test
        - eval_llm (str | DeepEvalBaseLLM): Provider for language model, str for GPT or custom model like mistral
        - app_id (str): Application ID.
        - context_relevance_threshold (float): threshold for test success
        - faithfulness_threshold (float): threshold for test success
        - answer_relevance_threshold (float): threshold for test success

        Returns:
        - RetrievalEvalBuilder: RetrievalEvalBuilder instance configured for RAG triad evaluation.
        """  # noqa: E501

        contextual_metric = ContextualRecallMetric(threshold=context_relevance_threshold, model=eval_llm, include_reason=True)
        faithfulness_metric = FaithfulnessMetric(threshold=faithfulness_threshold, model=eval_llm, include_reason=True)
        answer_relevancy_metric = AnswerRelevancyMetric(threshold=answer_relevance_threshold, model=eval_llm, include_reason=True)

        metrics = [contextual_metric, faithfulness_metric, answer_relevancy_metric]

        return cls(name=name, evaluation_type=RetrievalEvaluationType.TRIAD, metrics=metrics)

    @classmethod
    def from_faithfulness_and_answer_relevance_evaluation(
        cls,
        name: str,
        eval_llm: Union[DeepEvalBaseLLM, str],
        faithfulness_threshold=0.7,
        answer_relevance_threshold=0.7,
    ) -> RAGEvaluator:
        """
        Build RetrievalEvalBuilder for RAG triad Evaluation

        The RAG triad is made up of 3 evaluations:
        1. Contextual Recall
        2. Faithfulness
        3. Answer Relevance

        2. Faithfulness:
        Is the response supported by the context?

        Evaluating whether the Response factually aligns with the contents of the Retrieval

        After the context is retrieved, it is then formed into an answer by an LLM.
        LLMs are often prone to straying from the facts provided, exaggerating or expanding to a correct-sounding answer.

        3. Answer Relevance:
        Is the response relevant to the query?

        Evaluating how relevant the Response of the LLM is compared to the provided Input

        Lastly, the response still needs to helpfully answer the original question.
        Evaluating the relevance of the final response to the user input.

        Args:
        - name (str): Name of the test
        - eval_llm (str | DeepEvalBaseLLM): Provider for language model, str for GPT or custom model like mistral
        - app_id (str): Application ID.
        - context_relevance_threshold (float): threshold for test success
        - faithfulness_threshold (float): threshold for test success
        - answer_relevance_threshold (float): threshold for test success

        Returns:
        - RetrievalEvalBuilder: RetrievalEvalBuilder instance configured for RAG triad evaluation.
        """

        faithfulness_metric = FaithfulnessMetric(threshold=faithfulness_threshold, model=eval_llm, include_reason=True)
        answer_relevancy_metric = AnswerRelevancyMetric(threshold=answer_relevance_threshold, model=eval_llm, include_reason=True)

        metrics = [faithfulness_metric, answer_relevancy_metric]

        return cls(
            name=name,
            evaluation_type=RetrievalEvaluationType.FAITHFULNESS_ANSWER_RELEVANCE,
            metrics=metrics,
        )

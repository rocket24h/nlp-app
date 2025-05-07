import pandas as pd
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import bootstrap
import os
import time
from deepeval.metrics import GEval
from deepeval.models.llms import GeminiModel
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset

from modules.kg_index import load_KG_from_config

# KGIndex instance
kg_index = load_KG_from_config()

# File paths
# Replace with the actual path if needed
input_file = "test/qa_test_results.txt"

# Initialize an empty list to store triplets
triplets = []

# Read the text file and extract triplets
with open(input_file, "r", encoding="utf-8") as file:
    lines = file.readlines()
    question, predicted, ground_truth = None, None, None
    for line in lines:
        line = line.strip()
        if line.startswith("Question:"):
            question = line.replace("Question:", "").strip()
        elif line.startswith("Predicted:"):
            predicted = line.replace("Predicted:", "").strip()
        elif line.startswith("Ground Truth:"):
            ground_truth = line.replace("Ground Truth:", "").strip()
            if question and predicted and ground_truth:
                triplets.append((question, predicted, ground_truth))
                question, predicted, ground_truth = None, None, None

print(f"Loaded {len(triplets)} triplets from the file.")

# LLM Judge
prompt = f"You are a judge with profound knowledge spanning across multiple domains. Given the following question and two answers: one is from our system, one is the ground truth \
        Please rate the answers with either 0 or 1, with 0 meaning the answer is wrong, ambiguous or the system can't answer from the given context; and 1 meaning the answer is correct, and/or can answer the question. \
        Question: {question} \nAnswer: {predicted} \nGround truth: {ground_truth}\
        Please only return the score, without any explanation.\
        \
        In all cases, if the system states that they cannot answer based on the given context, or the answer is dead wrong. You return the score = 0   \
        \n"

# Initialize GEval with the Gemini model
model_name = "gemini-2.0-flash"
geval = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also lightly penalize omission of detail, and focus on the main idea",
        "Vague language, or contradicting OPINIONS, are OK",
        "Completely wrong answers are not OK",
        "No answers given based on context is also not OK",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT,
                       LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=GeminiModel(model_name=model_name,
                      api_key=os.environ.get("GOOGLE_API_KEY")),
)

# Sequentially take 10 triplets, create LLMTestCase, and evaluate them
batch_size = 10
for i in range(0, len(triplets), batch_size):
    batch = triplets[i:i + batch_size]
    test_cases = [
        LLMTestCase(
            input=question,
            actual_output=predicted,
            expected_output=ground_truth,
        )
        for question, predicted, ground_truth in batch
    ]
    dataset = EvaluationDataset(test_cases)
    results = dataset.evaluate([geval])
    for test_result in results.test_results:
        write_content = ""
        write_content += f"Question: {test_result.input}\n"
        write_content += f"Predicted: {test_result.actual_output}\n"
        write_content += f"Ground Truth: {test_result.expected_output}\n"
        for metric in test_result.metrics_data:
            write_content += f"{metric.name}: {metric.score}\n"
        write_content += "-" * 80 + "\n"
        with open("test/qa_test_llm_results.txt", "a", encoding="utf-8") as f:
            f.write(write_content)
    # Sleep for 60 seconds to avoid hitting the rate limit
    time.sleep(60)
    print(
        f"Processed {i + batch_size} triplets, saved results to 'test/qa_test_llm_results.txt'")

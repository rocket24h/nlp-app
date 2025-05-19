import pandas as pd
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import bootstrap
import time

from modules.kg_index import load_KG_from_config

# KGIndex instance
kg_index = load_KG_from_config()

# Load datasets
in_topic_df = pd.read_csv("test/data/300_basic.csv")
cross_topic_df = pd.read_csv("test/data/cross_topic_qa.csv")

datasets = {
    "in_topic": in_topic_df,
    "cross_topic": cross_topic_df,
}

print("Loaded datasets.")

# Initialize models
sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
smoothing = SmoothingFunction()


def normalize(text):
    return text.strip()


def evaluate_response(predicted, ground_truth):
    # Normalize
    predicted = normalize(predicted)
    ground_truth = normalize(ground_truth)

    # Cosine similarity
    cosine_sim = util.cos_sim(
        sbert_model.encode(predicted, convert_to_tensor=True),
        sbert_model.encode(ground_truth, convert_to_tensor=True)
    ).item()

    # BLEU score (uses token overlap)
    reference = [ground_truth.split()]
    candidate = predicted.split()
    bleu_score = sentence_bleu(
        reference, candidate, smoothing_function=smoothing.method1)

    return cosine_sim, bleu_score


def kg_query(query, retry_limit=5, retry_delay=10, backoff=True):
    attempt = 0
    delay = retry_delay
    response = None

    while attempt < retry_limit:
        try:
            response = kg_index.query(query)
            return response.response
        except Exception as e:
            if "resource_exhausted" in str(e).lower():
                attempt += 1
                print(
                    f"\n[Attempt {attempt}] Resource exhausted. Retrying in {delay} seconds...")
                time.sleep(delay)
                if backoff:
                    delay *= 2  # exponential backoff
            else:
                print(f"[Error] {e}")
                break

    return "No response found."


results = {}
for name, df in datasets.items():
    cosine_scores = []
    bleu_scores = []
    predicted_answers = []
    ground_truth_answers = []
    print(f"Evaluating {name} dataset...")
    for _, row in tqdm(df.iterrows()):
        question = row["question"]
        ground_truth = row["answer"]
        predicted = kg_query(question)

        # Write to file for manual inspection
        with open("test/qa_test_results.txt", "a", encoding="utf-8") as f:
            f.write(f"Question: {question}\n")
            f.write(f"Predicted: {predicted}")
            f.write(f"Ground Truth: {ground_truth}\n")
            f.write("-" * 80 + "\n")

        cosine, bleu = evaluate_response(predicted, ground_truth)
        cosine_scores.append(cosine)
        bleu_scores.append(bleu)

        predicted_answers.append(predicted)
        ground_truth_answers.append(ground_truth)
        time.sleep(2)

    # BERTScore batch eval
    P, R, F1 = bert_score(predicted_answers, ground_truth_answers,
                          lang="en", model_type="bert-base-uncased")

    results[name] = {
        "Cosine Similarity": sum(cosine_scores) / len(cosine_scores),
        "BLEU Score": sum(bleu_scores) / len(bleu_scores),
        "BERTScore (F1)": float(F1.mean()),
    }

for dataset, metrics in results.items():
    print(f"\nResults for {dataset}:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.4f}")

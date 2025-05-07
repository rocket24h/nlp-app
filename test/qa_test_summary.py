import matplotlib.pyplot as plt

# File path
input_file = "test/qa_test_llm_results.txt"

# Initialize lists to store scores
first_300_scores = []
last_50_scores = []

# Read the file and extract scores
with open(input_file, "r", encoding="utf-8") as file:
    lines = file.readlines()
    scores = []
    for line in lines:
        line = line.strip()
        if line.startswith("Correctness (GEval):"):
            score = float(line.replace("Correctness (GEval):", "").strip())
            scores.append(score)

# Calculate averages for the first 300 and last 50 scores
first_300_scores = scores[:300]
last_50_scores = scores[-50:]

average_first_300 = sum(first_300_scores) / \
    len(first_300_scores) if first_300_scores else 0
average_last_50 = sum(last_50_scores) / \
    len(last_50_scores) if last_50_scores else 0

# Print the results
print(
    f"Average score for the first 300 question-answers: {average_first_300:.2f}")
print(f"Average score for the last 50 question-answers: {average_last_50:.2f}")

# Plot the score distributions for the two splits
plt.figure(figsize=(12, 6))
plt.hist(first_300_scores, bins=20, alpha=0.7,
         label="First 300 Scores", color="blue", edgecolor="black")
plt.hist(last_50_scores, bins=10, alpha=0.7,
         label="Last 50 Scores", color="orange", edgecolor="black")
plt.title("Score Distribution (Correctness GEval)")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

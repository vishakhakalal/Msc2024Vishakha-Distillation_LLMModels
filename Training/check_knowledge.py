import json
import numpy as np
from scipy.special import rel_entr

# Paths to the JSON files
student_normal_path = '/nfs/primary/distillation/Training/datasets/student_scores_CAT_normal.json'
student_poison_path = '/nfs/primary/distillation/Training/datasets/poison_student_scores_10k_CAT.json'
teacher_normal_path = '/nfs/primary/distillation/Training/datasets/teacher_scores_10k_normal.json'
teacher_poison_path = '/nfs/primary/distillation/Training/datasets/poison_teacher_scores_10k.json'


# Load the JSON files
def load_scores(path):
    with open(path, 'r') as f:
        return json.load(f)


student_normal_scores = load_scores(student_normal_path)
student_poison_scores = load_scores(student_poison_path)
teacher_normal_scores = load_scores(teacher_normal_path)
teacher_poison_scores = load_scores(teacher_poison_path)


# Print the number of queries and documents in each set
def print_summary(scores, label):
    num_queries = len(scores)
    num_docs = len(set(docno for qid in scores for docno in scores[qid]))
    print(f"{label} - Number of queries: {num_queries}, Number of documents: {num_docs}")


print_summary(student_normal_scores, "Student Normal")
print_summary(teacher_normal_scores, "Teacher Normal")
print_summary(student_poison_scores, "Student Poison")
print_summary(teacher_poison_scores, "Teacher Poison")


# Calculate KL Divergence
def calculate_kl_divergence(student_scores, teacher_scores):
    kl_div_values = []
    matching_pairs = 0
    for qid in student_scores.keys():
        if qid in teacher_scores:
            student_probs = []
            teacher_probs = []
            for docno in student_scores[qid].keys():
                if docno in teacher_scores[qid]:
                    student_probs.append(student_scores[qid][docno])
                    teacher_probs.append(teacher_scores[qid][docno])

            if student_probs and teacher_probs:
                student_probs = np.array(student_probs)
                teacher_probs = np.array(teacher_probs)

                # Normalize to form probability distributions
                student_probs_sum = student_probs.sum()
                teacher_probs_sum = teacher_probs.sum()

                if student_probs_sum == 0 or teacher_probs_sum == 0:
                    print(f"Warning: Zero sum encountered for query {qid}.")
                    continue

                student_probs = np.clip(student_probs / student_probs_sum, 1e-10, 1)
                teacher_probs = np.clip(teacher_probs / teacher_probs_sum, 1e-10, 1)

                kl_div = np.sum(rel_entr(student_probs, teacher_probs))
                kl_div_values.append(kl_div)
                matching_pairs += 1
    if matching_pairs == 0:
        print("No matching scores found for KL divergence calculation.")
        return float('nan')
    return np.mean(kl_div_values)


kl_div_normal = calculate_kl_divergence(student_normal_scores, teacher_normal_scores)
kl_div_poison = calculate_kl_divergence(student_poison_scores, teacher_poison_scores)

print(f"KL Divergence Normal: {kl_div_normal}")
print(f"KL Divergence Poison: {kl_div_poison}")

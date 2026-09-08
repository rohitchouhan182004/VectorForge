import time
import numpy as np

from data import generate_dataset
from vector_index import ExactVectorIndex
from ivf_index import IVFIndex


# ==========================================
# 1. Generate Dataset
# ==========================================

print("Generating dataset...")

vectors, centers, cluster_ids = generate_dataset(
    num_vectors=50000,
    dimension=64,
    num_clusters=100
)

print("Dataset ready!")


# ==========================================
# 2. Build Exact Index
# ==========================================

print("\nBuilding Exact Index...")

exact_index = ExactVectorIndex(
    dimension=64
)

for vector_id, vector in enumerate(vectors):
    exact_index.insert(vector_id, vector)

print("Exact index vectors:", exact_index.count())


# ==========================================
# 3. Build IVF Index
# ==========================================

print("\nBuilding IVF Index...")

ivf_index = IVFIndex(
    vectors,
    centers,
    cluster_ids
)

print("IVF index ready!")


# ==========================================
# 4. Create 500 Queries
# ==========================================

np.random.seed(123)

query_ids = np.random.choice(
    len(vectors),
    size=500,
    replace=False
)

queries = vectors[query_ids]


# ==========================================
# 5. Calculate Exact Ground Truth
# ==========================================

print("\nCalculating exact ground truth...")

ground_truth = []

for i, query in enumerate(queries):

    results = exact_index.search(
        query,
        k=10
    )

    ground_truth.append(results)

    if (i + 1) % 100 == 0:
        print(f"Ground truth: {i + 1}/500")


# ==========================================
# 6. Test Different nprobe Values
# ==========================================

nprobe_values = [1, 5, 10, 20, 50]

print("\n")
print("=" * 75)
print("             IVF nprobe EXPERIMENT")
print("=" * 75)

print(
    f"{'nprobe':<10}"
    f"{'IVF Time(ms)':<18}"
    f"{'Recall@10':<15}"
    f"{'Speedup':<15}"
    f"{'Candidates':<15}"
)

print("-" * 75)


for nprobe in nprobe_values:

    ivf_times = []
    recalls = []
    candidate_counts = []


    # --------------------------------------
    # Run all 500 queries
    # --------------------------------------

    for query_number, query in enumerate(queries):

        start = time.perf_counter()

        ivf_results = ivf_index.search(
            query,
            k=10,
            nprobe=nprobe
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        ivf_times.append(elapsed)


        # ----------------------------------
        # Recall@10
        # ----------------------------------

        exact_ids = {
            result["id"]
            for result in ground_truth[query_number]
        }

        ivf_ids = {
            result["id"]
            for result in ivf_results
        }

        common = exact_ids & ivf_ids

        recall = len(common) / 10

        recalls.append(recall)


        # ----------------------------------
        # Candidate count
        # ----------------------------------

        count = ivf_index.get_candidate_count(
            query,
            nprobe=nprobe
        )

        candidate_counts.append(count)


    # --------------------------------------
    # Average metrics
    # --------------------------------------

    avg_time = np.mean(ivf_times)

    avg_recall = np.mean(recalls) * 100

    avg_candidates = np.mean(candidate_counts)

    # Exact time from previous benchmark
    exact_time = 19.829

    speedup = exact_time / avg_time


    print(
        f"{nprobe:<10}"
        f"{avg_time:<18.3f}"
        f"{avg_recall:<15.2f}%"
        f"{speedup:<15.2f}x"
        f"{avg_candidates:<15.0f}"
    )


print("=" * 75)

print("\nExperiment completed!")
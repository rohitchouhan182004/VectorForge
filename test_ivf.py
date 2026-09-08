import numpy as np

from data import generate_dataset
from ivf_index import IVFIndex


# Generate dataset
# 
vectors, centers, cluster_ids = generate_dataset(
    num_vectors=50000,
    dimension=64,
    num_clusters=100
)


# Build IVF index
# cluster_ids = np.random.randint(
#     0,
#     100,
#     size=50000
# )


index = IVFIndex(
    vectors,
    centers,
    cluster_ids
)


# Random query
query = vectors[100]


# Search
results = index.search(
    query,
    k=10,
    nprobe=5
)


print("\nIVF Search Results:")

for result in results:
    print(result)


# Candidate count
candidate_count = index.get_candidate_count(
    query,
    nprobe=5
)


print("\nTotal vectors:", len(vectors))

print(
    "Vectors searched by IVF:",
    candidate_count
)

print(
    "Search percentage:",
    round(
        candidate_count / len(vectors) * 100,
        2
    ),
    "%"
)
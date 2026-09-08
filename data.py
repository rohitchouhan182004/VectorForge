import numpy as np


def generate_dataset(
    num_vectors=50000,
    dimension=64,
    num_clusters=100,
    seed=42
):
    np.random.seed(seed)

    # Create cluster centers
    centers = np.random.randn(
        num_clusters,
        dimension
    ).astype(np.float32)

    # Normalize centers
    centers = centers / np.linalg.norm(
        centers,
        axis=1,
        keepdims=True
    )

    # Randomly assign every vector to a cluster
    cluster_ids = np.random.randint(
        0,
        num_clusters,
        size=num_vectors
    )

    # Create vectors around their cluster center
    vectors = centers[cluster_ids] + (
        np.random.randn(
            num_vectors,
            dimension
        ).astype(np.float32) * 0.15
    )

    # Normalize all vectors
    vectors = vectors / np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    # return vectors.astype(np.float32), centers.astype(np.float32)
    return (
    vectors.astype(np.float32),
    centers.astype(np.float32),
    cluster_ids
)

if __name__ == "__main__":

    vectors, centers = generate_dataset()

    print("Dataset created!")
    print("Number of vectors:", len(vectors))
    print("Vector dimension:", vectors.shape[1])
    print("Number of clusters:", len(centers))
    print("Memory:", round(vectors.nbytes / (1024 * 1024), 2), "MB")
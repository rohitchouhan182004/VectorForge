import numpy as np


class IVFIndex:

    def __init__(self, vectors, centers, cluster_ids):

        self.vectors = vectors
        self.centers = centers
        self.cluster_ids = cluster_ids

        self.num_clusters = len(centers)

        # Inverted index
        self.inverted_lists = [[] for _ in range(self.num_clusters)]

        # Deleted vector IDs
        self.deleted_ids = set()

        # Next ID for new vectors
        self.next_id = len(vectors)

        # Build inverted lists
        for vector_id, cluster_id in enumerate(cluster_ids):
            self.inverted_lists[cluster_id].append(vector_id)


    def insert(self, vector):

        vector = np.array(vector, dtype=np.float32)

        if len(vector) != self.vectors.shape[1]:
            raise ValueError(
                f"Expected dimension {self.vectors.shape[1]}, "
                f"but got {len(vector)}"
            )

        norm = np.linalg.norm(vector)

        if norm == 0:
            raise ValueError("Zero vector is not allowed")

        # Normalize
        vector = vector / norm

        # Find nearest cluster
        cluster_scores = self.centers @ vector

        cluster_id = int(np.argmax(cluster_scores))

        # New vector ID
        vector_id = self.next_id
        self.next_id += 1

        # Add vector
        self.vectors = np.vstack([
            self.vectors,
            vector
        ])

        # Add cluster ID
        self.cluster_ids = np.append(
            self.cluster_ids,
            cluster_id
        )

        # Add to inverted list
        self.inverted_lists[cluster_id].append(vector_id)

        return vector_id


    def delete(self, vector_id):

        if vector_id < 0 or vector_id >= self.next_id:
            return False

        if vector_id in self.deleted_ids:
            return False

        self.deleted_ids.add(vector_id)

        return True


    def search(self, query, k=10, nprobe=5):

        query = np.array(query, dtype=np.float32)

        if len(query) != self.vectors.shape[1]:
            raise ValueError(
                f"Expected dimension {self.vectors.shape[1]}, "
                f"but got {len(query)}"
            )

        norm = np.linalg.norm(query)

        if norm == 0:
            raise ValueError("Zero query vector is not allowed")

        # Normalize query
        query = query / norm

        # Find nearest clusters
        cluster_scores = self.centers @ query

        nprobe = min(
            nprobe,
            self.num_clusters
        )

        nearest_clusters = np.argsort(
            cluster_scores
        )[::-1][:nprobe]

        # Collect candidates
        candidates = []

        for cluster_id in nearest_clusters:

            for vector_id in self.inverted_lists[cluster_id]:

                # Ignore deleted vectors
                if vector_id not in self.deleted_ids:
                    candidates.append(vector_id)

        if not candidates:
            return []

        # Get candidate vectors
        candidate_vectors = self.vectors[candidates]

        # Cosine similarity
        scores = candidate_vectors @ query

        # Top K
        k = min(k, len(scores))

        top_indices = np.argsort(
            scores
        )[::-1][:k]

        results = []

        for index in top_indices:

            vector_id = candidates[index]

            results.append({
                "id": int(vector_id),
                "score": float(scores[index])
            })

        return results


    def get_candidate_count(self, query, nprobe=5):

        query = np.array(
            query,
            dtype=np.float32
        )

        norm = np.linalg.norm(query)

        if norm == 0:
            raise ValueError("Zero query vector is not allowed")

        query = query / norm

        # Cluster similarity
        cluster_scores = self.centers @ query

        nprobe = min(
            nprobe,
            self.num_clusters
        )

        nearest_clusters = np.argsort(
            cluster_scores
        )[::-1][:nprobe]

        count = 0

        for cluster_id in nearest_clusters:

            for vector_id in self.inverted_lists[cluster_id]:

                if vector_id not in self.deleted_ids:
                    count += 1

        return count


    def count(self):

        return len(self.vectors) - len(self.deleted_ids)
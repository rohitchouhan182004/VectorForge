import numpy as np


class ExactVectorIndex:

    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = []
        self.ids = []

    def insert(self, vector_id, vector):
        vector = np.array(vector, dtype=np.float32)

        if len(vector) != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, "
                f"but got {len(vector)}"
            )

        # Normalize the vector
        norm = np.linalg.norm(vector)

        if norm == 0:
            raise ValueError("Zero vector is not allowed")

        vector = vector / norm

        self.ids.append(vector_id)
        self.vectors.append(vector)

    def search(self, query_vector, k=5):

        if len(self.vectors) == 0:
            return []

        query_vector = np.array(
            query_vector,
            dtype=np.float32
        )

        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, "
                f"but got {len(query_vector)}"
            )

        # Normalize query
        norm = np.linalg.norm(query_vector)

        if norm == 0:
            raise ValueError("Zero query vector is not allowed")

        query_vector = query_vector / norm

        # Convert list to NumPy matrix
        matrix = np.array(self.vectors)

        # Cosine similarity
        scores = matrix @ query_vector

        # Get top-k indices
        k = min(k, len(scores))

        top_indices = np.argsort(scores)[::-1][:k]

        results = []

        for index in top_indices:
            results.append({
                "id": self.ids[index],
                "score": float(scores[index])
            })

        return results

    def delete(self, vector_id):

        if vector_id not in self.ids:
            return False

        index = self.ids.index(vector_id)

        self.ids.pop(index)
        self.vectors.pop(index)

        return True

    def count(self):
        return len(self.ids)
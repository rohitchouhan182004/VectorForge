import hashlib
import numpy as np


# 20 technical topics.
TOPICS = [
    "machine learning",
    "artificial intelligence",
    "deep learning",
    "computer vision",
    "natural language processing",
    "data science",
    "cloud computing",
    "cyber security",
    "software engineering",
    "database systems",
    "distributed systems",
    "web development",
    "mobile applications",
    "internet of things",
    "robotics",
    "blockchain",
    "financial technology",
    "healthcare technology",
    "education technology",
    "e commerce",
]


# Technical scenarios add variety while keeping the main topic explicit.
CONTEXTS = [
    "customer analytics",
    "fraud detection",
    "recommendation systems",
    "real time monitoring",
    "document processing",
    "predictive maintenance",
    "resource allocation",
    "anomaly detection",
    "workflow automation",
    "quality assurance",
    "search and retrieval",
    "performance optimization",
    "access control",
    "data integration",
    "scalable services",
    "decision support",
    "model evaluation",
    "event processing",
    "personalized applications",
    "operational reporting",
]


METHODS = [
    "batch processing",
    "stream processing",
    "feature engineering",
    "model training",
    "distributed execution",
    "caching",
    "indexing",
    "load balancing",
    "data validation",
    "continuous integration",
    "observability",
    "automated testing",
    "secure authentication",
    "schema management",
    "fault tolerance",
    "parallel processing",
    "latency optimization",
    "access auditing",
    "data partitioning",
    "API integration",
]


OUTCOMES = [
    "reduce processing time",
    "improve prediction quality",
    "handle larger workloads",
    "detect unusual behavior",
    "reduce manual effort",
    "increase system reliability",
    "improve retrieval accuracy",
    "control infrastructure cost",
    "support low latency queries",
    "maintain consistent data quality",
    "simplify deployment",
    "strengthen security controls",
    "improve resource utilization",
    "scale services horizontally",
    "make technical decisions faster",
]


TEMPLATES = [
    "A {topic} system can support {context} by using {method} to {outcome}.",
    "Engineers apply {topic} techniques to {context}, where {method} helps teams {outcome}.",
    "In production, {topic} solutions are often designed for {context}; {method} can {outcome}.",
    "A practical {topic} project combines {method} with {context} to {outcome}.",
    "Teams building {topic} applications use {method} when solving {context} problems and aiming to {outcome}.",
    "Reliable {topic} platforms need {method} for {context}, helping organizations {outcome}.",
    "During system development, {topic} models can improve {context} through {method} and {outcome}.",
    "Modern {topic} workflows connect {context} with {method} so that teams can {outcome}.",
    "An engineering pipeline for {topic} may include {method} while handling {context} to {outcome}.",
    "Organizations use {topic} for {context}, applying {method} as a way to {outcome}.",
]


DEPLOYMENT_TARGETS = [
    "edge devices",
    "cloud services",
    "backend APIs",
    "data pipelines",
    "production workloads",
]


def text_to_vector(text, dimension=64):
    """
    Deterministic lightweight text vectorization.

    This is NOT a neural semantic embedding model.
    It converts words into a fixed-size vector using SHA-256 hashing.
    """
    vector = np.zeros(dimension, dtype=np.float32)

    words = text.lower().split()

    for word in words:
        word = word.strip(".,!?;:")

        digest = hashlib.sha256(
            word.encode("utf-8")
        ).digest()

        for i in range(0, len(digest), 4):
            value = int.from_bytes(
                digest[i:i + 4],
                byteorder="little",
                signed=False
            )

            index = value % dimension

            contribution = (
                (value / 4294967295.0) * 2.0
            ) - 1.0

            vector[index] += contribution

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def generate_text_dataset(num_documents=5000):
    """
    Generate unique technical documents with diverse wording.

    The default 5,000 documents are spread across 20 topics.
    Each topic receives 250 documents with different scenarios,
    methods, outcomes, templates, and deployment targets.
    """
    texts = []

    for i in range(num_documents):
        topic = TOPICS[i % len(TOPICS)]

        topic_document_number = i // len(TOPICS)

        context = CONTEXTS[
            topic_document_number % len(CONTEXTS)
        ]

        method = METHODS[
            (topic_document_number // len(CONTEXTS)) % len(METHODS)
        ]

        outcome = OUTCOMES[
            (topic_document_number // (len(CONTEXTS) * len(METHODS)))
            % len(OUTCOMES)
        ]

        template = TEMPLATES[
            (topic_document_number // 5) % len(TEMPLATES)
        ]

        deployment_target = DEPLOYMENT_TARGETS[
            topic_document_number % len(DEPLOYMENT_TARGETS)
        ]

        formatted_text = template.format(
            topic=topic,
            context=context,
            method=method,
            outcome=outcome
        )

        text = (
            f"{formatted_text} "
            f"The workload targets {deployment_target} "
            f"and uses document reference {i}."
        )

        texts.append(text)

    vectors = np.array(
        [
            text_to_vector(text)
            for text in texts
        ],
        dtype=np.float32
    )

    return texts, vectors


if __name__ == "__main__":
    texts, vectors = generate_text_dataset()

    print("Documents:", len(texts))
    print("Vector shape:", vectors.shape)

    print("\nFirst 5 documents:")

    for i in range(5):
        print(f"\n{i}: {texts[i]}")

    print("\nVector first 10 values:")
    print(vectors[0][:10])

    print("\nUnique documents:", len(set(texts)))

VectorForge

VectorForge is a lightweight vector search engine built from scratch in Python. It compares exact brute-force cosine similarity search with a custom IVF-Flat approximate nearest-neighbor index.

Features

50,000 synthetic 64-dimensional vectors

100 vector clusters

Exact brute-force cosine similarity search

Custom IVF-Flat index

Configurable nprobe

Recall@10 evaluation against exact ground truth

500-query benchmark

Speedup and candidate-count measurements

Recall/QPS Pareto visualization

Insert, search, and delete APIs

5,000 short technical documents

Text-to-vector search layer using deterministic 64-dimensional vectors

FastAPI REST API

Browser dashboard for live demonstrations

Architecture

                    VectorForge
                        |
        +---------------+----------------+
        |                                |
   Vector Dataset                   Text Dataset
   50,000 vectors                   5,000 documents
   64 dimensions                    64D vectors
        |                                |
   +----+-----+                    +-----+------+
   |          |                    |            |
Exact       IVF-Flat            Exact       IVF-Flat
Search      Search              Search      Search
   |          |                    |            |
   +----------+--------------------+------------+
                        |
                   FastAPI API
                        |
                  Web Dashboard

IVF-Flat

IVF-Flat (Inverted File with Flat storage) reduces the amount of data searched for each query.

The index contains:

Cluster centers

An inverted list for every cluster

The original vectors stored without product quantization

For a query:

Normalize the query vector.

Compare it with all cluster centers.

Select the closest nprobe clusters.

Collect vectors from those clusters.

Compute exact cosine similarity only for those candidates.

Return the top-k candidates.

This creates a speed-versus-recall trade-off controlled by nprobe.

nprobe

A small nprobe searches fewer candidates and is faster.

A larger nprobe searches more candidates and generally improves Recall@10.

The dashboard evaluates:

nprobe = 1

nprobe = 5

nprobe = 10

nprobe = 20

nprobe = 50

Exact Ground Truth

The exact index checks every vector using cosine similarity.

The benchmark uses:

50,000 vectors

64 dimensions

500 randomly selected query vectors

top-10 exact results as ground truth

For every IVF configuration, Recall@10 is calculated as:

Recall@10 = common results between IVF top-10 and exact top-10 / 10

Speedup is calculated as:

Speedup = average exact search time / average IVF search time

QPS shown in the Pareto graph is calculated as:

QPS = 1000 / search_time_ms

The measured timings can vary between runs because they depend on the local machine and current system load.

Text Search

VectorForge also generates 5,000 short technical documents.

Each document is converted into a deterministic 64-dimensional vector using a lightweight hashing-based text vectorization method. The resulting vectors are indexed using the same Exact and IVF-Flat search implementations.

Important: this text layer is intentionally lightweight and deterministic. It is not a transformer-based semantic embedding model. It demonstrates how a text-to-vector layer can feed the custom vector index without adding a third-party vector-search library.

API Endpoints

GET /

Opens the VectorForge dashboard.

GET /stats

Returns dataset and index statistics.

Example:

{
  "vectors": 50000,
  "dimension": 64,
  "clusters": 100,
  "ivf_vectors": 50000,
  "exact_vectors": 50000,
  "deleted_vectors": 0,
  "text_documents": 5000,
  "text_dimension": 64,
  "text_clusters": 50
}

POST /search

Search the vector dataset.

Example request:

{
  "vector": [0.1, 0.1, 0.1],
  "k": 10,
  "nprobe": 5,
  "algorithm": "ivf"
}

The vector must contain exactly 64 values.

Supported algorithms:

ivf

exact

POST /insert

Insert a new 64-dimensional vector.

{
  "vector": [0.1, 0.1, 0.1, "..."]
}

The API returns the assigned vector ID.

DELETE /delete/{vector_id}

Marks a vector as deleted from the indexes.

Example:

DELETE /delete/50000

GET /sample/{vector_id}

Returns a stored sample vector by ID. The dashboard uses this endpoint to perform live vector searches.

POST /text-search

Search the 5,000 technical documents.

Example:

{
  "query": "machine learning applications",
  "k": 10,
  "nprobe": 5,
  "algorithm": "ivf"
}

GET /benchmark

Runs the 500-query exact-ground-truth benchmark and returns measurements for all configured nprobe values.

Installation

Python 3.12 was used during development.

Create/activate a virtual environment if desired, then install dependencies:

pip install -r requirements.txt

Run

From the project directory:

uvicorn app:app --reload

Then open:

http://127.0.0.1:8000

On startup, VectorForge:

Generates the 50,000-vector dataset.

Builds the exact index.

Builds the IVF-Flat index.

Generates 5,000 text documents.

Builds exact and IVF text indexes.

Starts the FastAPI server.

Project Structure

vectorforge/
│
├── app.py                 # FastAPI application and web dashboard
├── data.py                # 50K clustered synthetic vector dataset
├── vector_index.py        # Exact brute-force vector index
├── ivf_index.py           # Custom IVF-Flat implementation
├── text_data.py           # 5K documents and deterministic text vectors
├── benchmark.py            # Standalone benchmark utilities
├── test_index.py          # Exact index tests
├── test_ivf.py            # IVF index tests
├── requirements.txt       # Python dependencies
│
└── README.md              # Project documentation

Design Constraints

The vector-search implementation intentionally avoids external vector-search indexes such as:

FAISS

Chroma

Pinecone

sklearn nearest-neighbor indexes

The core nearest-neighbor logic is implemented directly with NumPy arrays, cosine similarity, cluster selection, and inverted lists.

Demonstration Flow

For a project demonstration:

Open the dashboard.

Show 50,000 Vectors, 64 Dimensions, and 100 Clusters.

Show Exact Brute-Force average search time.

Show IVF-Flat with nprobe=5.

Explain Recall@10 and speedup.

Run the live vector search using both Exact and IVF-Flat.

Search text such as machine learning applications.

Show the 5,000-document results and candidate count.

Run the 500-query benchmark.

Explain the Recall/QPS Pareto graph.

Demonstrate insert, search, and delete using the API if required.

Limitations

The benchmark measures CPU/runtime performance on the machine where the application is executed, so timings vary.

The synthetic vector dataset is clustered specifically to evaluate IVF behavior.

The text vectorizer is a deterministic hashing-based representation rather than a learned semantic embedding model.

The IVF implementation uses flat exact cosine scoring inside the selected clusters.

Deleted vectors are handled through a deleted-ID set rather than physically compacting the stored arrays.

Summary

VectorForge demonstrates the core ideas behind approximate nearest-neighbor vector search without relying on a prebuilt vector-search library. It provides an exact baseline, a custom IVF-Flat implementation, configurable nprobe, quantitative Recall@10 evaluation, speed measurements, a Pareto visualization, REST APIs, and a text-search demonstration layer.
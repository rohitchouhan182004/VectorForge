VectorForge

A from-scratch vector search engine implementing Exact Brute-Force Cosine Search and a custom IVF-Flat Approximate Nearest Neighbor (ANN) index.

The project is built without Pinecone, FAISS, Chroma, sklearn.neighbors, or other pre-built vector indexing libraries.

Highlights

50,000 synthetic 64-dimensional vectors

100 IVF clusters

Exact brute-force cosine similarity baseline

IVF-Flat implemented from scratch

Configurable nprobe for speed/recall trade-offs

500 query vectors with exact Top-10 ground truth

Recall@10 benchmarking

FastAPI REST API

Insert, search, delete, and statistics endpoints

5,000 generated technical text documents for a text-search demo

Browser dashboard with benchmark visualizations

Architecture

                 VectorForge
                     |
          +----------+----------+
          |                     |
    Exact Index             IVF-Flat Index
          |                     |
   Brute-force cosine     Cluster assignment
          |                     |
     Ground truth        nprobe clusters
          |                     |
          +----------+----------+
                     |
                Top-K results

Exact Search

The exact index compares the query against every stored vector using cosine similarity. This provides the ground truth used to evaluate ANN recall.

IVF-Flat

The IVF-Flat implementation:

Creates cluster centers.

Assigns every vector to its nearest center.

Stores vector IDs in inverted lists.

At query time, selects the nearest nprobe clusters.

Computes exact cosine similarity only for vectors in those clusters.

Returns the Top-K candidates.

Benchmark

The benchmark uses 50,000 vectors, 64 dimensions, 100 clusters, and 500 query vectors. Exact brute-force Top-10 results are used as ground truth.

One benchmark run produced:

nprobe

Recall@10

Avg Candidates

Speedup

1

99.56%

501

133.58x

5

99.90%

2,503

29.66x

10

99.98%

5,002

11.55x

20

100.00%

10,001

6.06x

50

100.00%

25,001

2.48x

Timing can vary slightly between runs depending on the machine and system load. A useful operating point is nprobe=5, which achieved approximately 99.9% Recall@10 while examining about 5% of the dataset in this benchmark run.

Data and What Is Mocked

The benchmark data is intentionally synthetic so the experiment is reproducible.

The main 50,000-vector dataset is generated programmatically as clustered synthetic vectors.

The 5,000 technical documents are generated programmatically from technical topics, contexts, methods, outcomes, and deployment targets.

Text documents are converted into deterministic 64-dimensional vectors using lightweight hashing-based vectorization.

The text vectorizer is not a transformer-based semantic embedding model.

No external vector database is used.

No pre-built ANN/vector-index library is used.

Exact search and IVF-Flat indexing logic are implemented in this project.

Requirements

Python 3.12+

pip

Dependencies are pinned in requirements.txt.

Installation

1. Clone the repository

git clone https://github.com/rohitchouhan182004/VectorForge.git
cd VectorForge

2. Create a virtual environment

Windows PowerShell:

python -m venv .venv
.venv\Scripts\Activate.ps1

3. Install dependencies

pip install -r requirements.txt

Run the Application

Start the FastAPI server:

uvicorn app:app --reload

Open the dashboard:

http://127.0.0.1:8000

API Endpoints

Search

POST /search

Search using the exact index or IVF-Flat. The vector must contain 64 dimensions.

Example:

{
  "vector": [0.1, 0.1, 0.1],
  "k": 10,
  "index": "ivf",
  "nprobe": 5
}

Insert

POST /insert

Adds a new 64-dimensional vector and returns its assigned ID.

Delete

DELETE /delete/{vector_id}

Deletes a vector by ID.

Statistics

GET /stats

Returns vector counts, dimensions, cluster counts, deleted vectors, and text-document statistics.

Benchmark

GET /benchmark

Runs the exact-vs-IVF benchmark using 500 query vectors and reports Recall@10, latency, candidate counts, and speedup.

Text Search

POST /text-search

Searches the generated technical-document collection using the same vector-search infrastructure.

Run Tests

python -m pytest

If pytest is not installed:

pip install pytest

Project Structure

VectorForge/
├── app.py              # FastAPI API and browser dashboard
├── benchmark.py        # Exact vs IVF benchmark
├── data.py             # Synthetic 50K vector dataset generation
├── ivf_index.py        # Custom IVF-Flat implementation
├── vector_index.py     # Exact brute-force cosine index
├── text_data.py        # 5K technical text dataset and vectorization
├── test_index.py       # Exact index tests
├── test_ivf.py         # IVF index tests
├── requirements.txt    # Python dependencies
└── README.md           # Documentation

Design Constraints

This project intentionally avoids:

Pinecone

FAISS

Chroma

sklearn.neighbors

Other pre-built vector database/indexing solutions

NumPy is used for vector arithmetic and matrix operations.

Why IVF-Flat?

Exact search gives perfect recall but must compare the query with all vectors. IVF-Flat reduces the search space by routing the query to nearby clusters and then performing exact similarity calculations inside those selected clusters.

nprobe controls the trade-off:

Lower nprobe → fewer candidates, faster search, potentially lower recall

Higher nprobe → more candidates, slower search, higher recall

Demo Flow

Start the FastAPI application.

Open the VectorForge dashboard.

Show the 50K-vector dataset statistics.

Run the benchmark.

Compare Exact Brute-Force with IVF-Flat.

Show Recall@10 and speedup.

Run an IVF search with nprobe=5.

Demonstrate text search over 5,000 technical documents.

Demonstrate Insert → Search → Delete using the API.

License

This project is created as a technical assignment and portfolio/interview demonstration.
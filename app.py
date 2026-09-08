from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import numpy as np
import time

from data import generate_dataset
from vector_index import ExactVectorIndex
from ivf_index import IVFIndex
from text_data import generate_text_dataset, text_to_vector


# ============================================================
# CONFIGURATION
# ============================================================

NUM_VECTORS = 50000
DIMENSION = 64
NUM_CLUSTERS = 100


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="VectorForge",
    description="Brute-force and IVF-Flat vector search engine",
    version="1.0"
)


# ============================================================
# DATASET
# ============================================================

print("Generating 50,000 vectors...")

vectors, centers, cluster_ids = generate_dataset(
    num_vectors=NUM_VECTORS,
    dimension=DIMENSION,
    num_clusters=NUM_CLUSTERS
)

print("Dataset ready!")


# ============================================================
# EXACT INDEX
# ============================================================

print("Building exact index...")

exact_index = ExactVectorIndex(
    dimension=DIMENSION
)

for vector_id, vector in enumerate(vectors):
    exact_index.insert(vector_id, vector)

print("Exact index ready!")


# ============================================================
# IVF INDEX
# ============================================================

print("Building IVF index...")

ivf_index = IVFIndex(
    vectors.copy(),
    centers,
    cluster_ids.copy()
)

print("IVF index ready!")


print("Generating 5,000 text documents...")
text_documents, text_vectors = generate_text_dataset(num_documents=5000)

# Lightweight manual clustering for the text layer.
# We intentionally avoid sklearn / FAISS / other vector-search libraries.
TEXT_CLUSTERS = 50
rng = np.random.default_rng(42)
text_center_indices = rng.choice(len(text_vectors), size=TEXT_CLUSTERS, replace=False)
text_centers = text_vectors[text_center_indices].copy()
text_cluster_ids = np.argmax(text_vectors @ text_centers.T, axis=1)

print("Building exact text index...")
text_exact_index = ExactVectorIndex(dimension=DIMENSION)
for text_id, vector in enumerate(text_vectors):
    text_exact_index.insert(text_id, vector)
print("Exact text index ready!")

print("Building IVF text index...")
text_ivf_index = IVFIndex(
    text_vectors.copy(),
    text_centers,
    text_cluster_ids.copy()
)
print("IVF text index ready!")



# ============================================================
# REQUEST MODELS
# ============================================================

class SearchRequest(BaseModel):
    vector: list[float] = Field(
        ...,
        description="64-dimensional query vector"
    )
    k: int = Field(default=10, ge=1, le=100)
    nprobe: int = Field(default=5, ge=1, le=100)
    algorithm: str = Field(default="ivf")


class InsertRequest(BaseModel):
    vector: list[float]


class TextSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Short text query"
    )
    k: int = Field(default=10, ge=1, le=100)
    nprobe: int = Field(default=5, ge=1, le=50)
    algorithm: str = Field(default="ivf")


# ============================================================
# VALIDATION
# ============================================================

def validate_vector(vector):
    if len(vector) != DIMENSION:
        raise ValueError(
            f"Vector must have exactly {DIMENSION} dimensions"
        )


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>VectorForge</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: #0b1020;
    color: #f5f7ff;
}

.container {
    max-width: 1200px;
    margin: auto;
    padding: 35px 24px 60px;
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
}

.logo {
    font-size: 36px;
    font-weight: 800;
}

.subtitle {
    color: #8f9ab4;
    margin-top: 7px;
}

.badge {
    display: inline-block;
    padding: 6px 10px;
    margin-left: 6px;
    border-radius: 20px;
    font-size: 11px;
    background: #202942;
    color: #8fdcff;
}

.status {
    padding: 9px 14px;
    border-radius: 20px;
    background: #14251d;
    color: #61f2a3;
    font-size: 12px;
    font-weight: 700;
}

.grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}

.card {
    background: #131a2e;
    border: 1px solid #252e48;
    border-radius: 16px;
    padding: 22px;
}

.section {
    margin-top: 22px;
}

.stat-value {
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 6px;
}

.stat-label {
    color: #8f9ab4;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.hero {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

.hero-card {
    background: #131a2e;
    border: 1px solid #252e48;
    border-radius: 16px;
    padding: 28px;
}

.hero-title {
    color: #8f9ab4;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.big-number {
    font-size: 48px;
    font-weight: 900;
    margin: 10px 0;
}

.green {
    color: #61f2a3;
}

.blue {
    color: #6dc7ff;
}

.muted {
    color: #8f9ab4;
}

.search-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr auto;
    gap: 12px;
    align-items: end;
}

label {
    color: #9ba5bd;
    font-size: 13px;
}

input,
select {
    width: 100%;
    padding: 12px;
    margin-top: 7px;
    background: #0d1427;
    color: white;
    border: 1px solid #303a57;
    border-radius: 8px;
    outline: none;
}

button {
    padding: 12px 18px;
    border: none;
    border-radius: 8px;
    background: #5b8cff;
    color: white;
    font-weight: 700;
    cursor: pointer;
}

button:hover {
    opacity: 0.85;
}

.result {
    margin-top: 18px;
    background: #090d18;
    border-radius: 10px;
    padding: 18px;
    overflow-x: auto;
    color: #8fffc1;
    font-family: Consolas, monospace;
    font-size: 13px;
    white-space: pre-wrap;
    min-height: 100px;
}

.text-summary {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 18px;
}

.text-stat {
    background: #0d1427;
    border: 1px solid #283451;
    border-radius: 10px;
    padding: 13px;
}

.text-stat-value {
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 4px;
}

.text-stat-label {
    color: #8f9ab4;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.text-result-card {
    margin-top: 12px;
    padding: 16px;
    background: #0d1427;
    border: 1px solid #283451;
    border-radius: 12px;
}

.text-result-top {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 9px;
}

.text-rank {
    font-weight: 800;
    color: #6dc7ff;
}

.text-score {
    font-weight: 800;
    color: #61f2a3;
}

.text-content {
    color: #e7ebf7;
    line-height: 1.55;
    font-size: 13px;
}

.text-meta {
    color: #8f9ab4;
    font-size: 11px;
    margin-top: 9px;
}

.algorithm-note {
    margin-top: 16px;
    padding: 12px 14px;
    background: #111a31;
    border-left: 3px solid #5b8cff;
    border-radius: 8px;
    color: #aeb9d2;
    font-size: 12px;
    line-height: 1.5;
}

.pareto-wrap {
    margin-top: 22px;
    background: #090d18;
    border: 1px solid #283451;
    border-radius: 12px;
    padding: 18px;
    overflow-x: auto;
}

.pareto-title {
    font-size: 16px;
    font-weight: 800;
    margin-bottom: 4px;
}

.pareto-subtitle {
    color: #8f9ab4;
    font-size: 12px;
    margin-bottom: 14px;
}

#paretoChart {
    width: 100%;
    min-width: 720px;
    height: 430px;
    display: block;
}

.pareto-legend {
    display: flex;
    gap: 22px;
    align-items: center;
    margin-top: 8px;
    color: #aeb9d2;
    font-size: 12px;
}

.legend-dot {
    display: inline-block;
    width: 11px;
    height: 11px;
    border-radius: 50%;
    margin-right: 6px;
}

.legend-frontier {
    background: #d92f2f;
}

.legend-dominated {
    background: #858585;
}

.pareto-note {
    margin-top: 12px;
    color: #8f9ab4;
    font-size: 11px;
    line-height: 1.5;
}

@media(max-width: 850px) {
    .text-summary {
        grid-template-columns: 1fr;
    }
}

.loading {
    color: #ffd36d;
    margin-top: 14px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 18px;
}

th,
td {
    padding: 13px;
    text-align: left;
    border-bottom: 1px solid #28324d;
}

th {
    color: #8f9ab4;
    font-size: 11px;
    text-transform: uppercase;
}

td {
    color: #e7ebf7;
    font-size: 14px;
}

.bar-row {
    margin-top: 16px;
}

.bar-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 7px;
    font-size: 13px;
}

.bar-bg {
    width: 100%;
    height: 10px;
    background: #252d43;
    border-radius: 20px;
    overflow: hidden;
}

.bar {
    height: 100%;
    background: #5b8cff;
    border-radius: 20px;
}

.footer {
    text-align: center;
    color: #68738d;
    font-size: 12px;
    margin-top: 35px;
}

@media(max-width: 850px) {

    .grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .hero {
        grid-template-columns: 1fr;
    }

    .search-grid {
        grid-template-columns: 1fr;
    }

    .header {
        flex-direction: column;
        align-items: flex-start;
        gap: 15px;
    }
}

@media(max-width: 500px) {

    .grid {
        grid-template-columns: 1fr;
    }

    .logo {
        font-size: 28px;
    }
}

</style>
</head>


<body>

<div class="container">

    <div class="header">

        <div>

            <div class="logo">
                ⚡ VectorForge

                <span class="badge">IVF-Flat</span>
                <span class="badge">Python</span>
            </div>

            <div class="subtitle">
                High-performance vector search engine
            </div>

        </div>

        <div class="status">
            ● SYSTEM ONLINE
        </div>

    </div>


    <!-- DATASET STATS -->

    <div class="grid">

        <div class="card">
            <div class="stat-value" id="vectors">--</div>
            <div class="stat-label">Vectors</div>
        </div>

        <div class="card">
            <div class="stat-value" id="dimension">--</div>
            <div class="stat-label">Dimensions</div>
        </div>

        <div class="card">
            <div class="stat-value" id="clusters">--</div>
            <div class="stat-label">Clusters</div>
        </div>

        <div class="card">
            <div class="stat-value">500</div>
            <div class="stat-label">Benchmark Queries</div>
        </div>

    </div>


    <!-- PERFORMANCE -->

    <div class="section">

        <div class="hero">

            <div class="hero-card">

                <div class="hero-title">
                    EXACT BRUTE-FORCE
                </div>

                <div class="big-number blue" id="exactTime">
                    --
                </div>

                <div class="muted">
                    Average search time
                </div>

                <br>

                <div class="muted">
                    Candidates searched:
                    <b id="exactCandidates" style="color:white;">
                        --
                    </b>
                </div>

            </div>


            <div class="hero-card">

                <div class="hero-title">
                    IVF-FLAT · NPROBE 5
                </div>

                <div class="big-number green" id="speedup">
                    --
                </div>

                <div class="muted">
                    Faster than brute-force
                </div>

                <br>

                <div>
                    Recall@10:
                    <b class="green" id="recall">--</b>
                </div>

                <br>

                <div class="muted">
                    Candidates searched:
                    <b id="candidates" style="color:white;">
                        --
                    </b>
                </div>

            </div>

        </div>

    </div>


    <!-- LIVE SEARCH -->

    <div class="section">

        <div class="card">

            <h2>🔎 Live Vector Search</h2>

            <p class="muted">
                Compare Exact Brute-Force and IVF-Flat
                on the 50,000-vector dataset.
            </p>


            <div class="search-grid">

                <div>

                    <label>Vector ID</label>

                    <input
                        id="vectorId"
                        type="number"
                        value="100"
                        min="0"
                        max="49999"
                    >

                </div>


                <div>

                    <label>Algorithm</label>

                    <select id="algorithm">

                        <option value="ivf">
                            IVF-Flat
                        </option>

                        <option value="exact">
                            Exact Brute-Force
                        </option>

                    </select>

                </div>


                <div>

                    <label>nprobe</label>

                    <input
                        id="nprobe"
                        type="number"
                        value="5"
                        min="1"
                        max="100"
                    >

                </div>


                <div>

                    <button onclick="searchVector()">
                        Run Search
                    </button>

                </div>

            </div>


            <div id="searchStatus" class="loading"></div>

            <div id="searchResult" class="result">
Ready for search...
            </div>

        </div>

    </div>



    <!-- TEXT SEARCH -->

    <div class="section">

        <div class="card">

            <h2>📝 Text Search</h2>

            <p class="muted">
                Search 5,000 short technical documents using deterministic
                64-dimensional text vectors and IVF-Flat.
            </p>

            <div class="search-grid">

                <div style="grid-column: span 2;">

                    <label>Search Query</label>

                    <input
                        id="textQuery"
                        type="text"
                        value="machine learning applications"
                        placeholder="e.g. machine learning, cloud computing"
                    >

                </div>

                <div>

                    <label>Algorithm</label>

                    <select id="textAlgorithm">

                        <option value="ivf">
                            IVF-Flat
                        </option>

                        <option value="exact">
                            Exact Brute-Force
                        </option>

                    </select>

                </div>

                <div>

                    <label>nprobe</label>

                    <input
                        id="textNprobe"
                        type="number"
                        value="5"
                        min="1"
                        max="50"
                    >

                </div>

            </div>

            <br>

            <button onclick="searchText()">
                Search Documents
            </button>

            <div id="textSearchStatus" class="loading"></div>

            <div id="textSearchResult">
                <div class="result">Ready for text search...</div>
            </div>

        </div>

    </div>


    <!-- BENCHMARK -->

    <div class="section">

        <div class="card">

            <h2>📊 IVF Performance Experiment</h2>

            <p class="muted">
                500-query exact ground truth benchmark
                across different nprobe values.
            </p>

            <button onclick="runBenchmark()">
                Run 500-Query Benchmark
            </button>

            <div id="benchmarkStatus" class="loading"></div>


            <div id="benchmarkTable"></div>

            <div class="pareto-wrap">

                <div class="pareto-title">
                    IVF-Flat Recall/QPS Pareto
                </div>

                <div class="pareto-subtitle">
                    Higher Recall@10 and higher QPS are better. The frontier
                    shows the best speed-versus-accuracy trade-offs.
                </div>

                <svg id="paretoChart"
                     viewBox="0 0 900 430"
                     preserveAspectRatio="xMidYMid meet">
                </svg>

                <div class="pareto-legend">
                    <span>
                        <span class="legend-dot legend-frontier"></span>
                        Pareto frontier
                    </span>

                    <span>
                        <span class="legend-dot legend-dominated"></span>
                        Dominated
                    </span>
                </div>

                <div class="pareto-note">
                    QPS is calculated from measured average search time:
                    QPS = 1000 / search_time_ms. A point is dominated when
                    another configuration is at least as good in both
                    Recall@10 and QPS and strictly better in one.
                </div>

            </div>

        </div>

    </div>


    <div class="footer">
        VectorForge · Exact Search + IVF-Flat
        · 50K vectors · 5K text documents · Recall@10 evaluation
    </div>

</div>


<script>


// ============================================================
// LOAD STATS
// ============================================================

async function loadStats() {

    try {

        const response = await fetch("/stats");

        const data = await response.json();

        document.getElementById("vectors").textContent =
            data.vectors.toLocaleString();

        document.getElementById("dimension").textContent =
            data.dimension;

        document.getElementById("clusters").textContent =
            data.clusters;

    }
    catch (error) {

        console.error("Stats error:", error);

    }

}


// ============================================================
// LIVE SEARCH
// ============================================================

async function searchVector() {

    const vectorId = Number(
        document.getElementById("vectorId").value
    );

    const algorithm =
        document.getElementById("algorithm").value;

    const nprobe = Number(
        document.getElementById("nprobe").value
    );

    const status =
        document.getElementById("searchStatus");

    const output =
        document.getElementById("searchResult");


    status.textContent = "Searching...";

    output.textContent = "";


    try {

        const sampleResponse =
            await fetch("/sample/" + vectorId);


        if (!sampleResponse.ok) {

            throw new Error("Vector ID not found");

        }


        const sample =
            await sampleResponse.json();


        const searchResponse =
            await fetch(
                "/search",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        vector: sample.vector,

                        k: 10,

                        nprobe: nprobe,

                        algorithm: algorithm

                    })
                }
            );


        const result =
            await searchResponse.json();


        if (!searchResponse.ok) {

            throw new Error(
                result.detail || "Search failed"
            );

        }


        output.textContent =
            JSON.stringify(
                result,
                null,
                2
            );


        status.textContent =
            "✓ Search completed successfully.";

    }
    catch (error) {

        output.textContent =
            error.message;

        status.textContent =
            "✗ Search failed.";

    }

}



// ============================================================
// TEXT SEARCH
// ============================================================

async function searchText() {

    const query =
        document.getElementById("textQuery").value.trim();

    const algorithm =
        document.getElementById("textAlgorithm").value;

    const nprobe = Number(
        document.getElementById("textNprobe").value
    );

    const status =
        document.getElementById("textSearchStatus");

    const output =
        document.getElementById("textSearchResult");

    if (!query) {
        status.textContent = "✗ Enter a search query.";
        output.textContent = "";
        return;
    }

    status.textContent = "Searching documents...";
    output.textContent = "";

    try {

        const response = await fetch("/text-search", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                query: query,
                k: 10,
                nprobe: nprobe,
                algorithm: algorithm
            })

        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Text search failed"
            );
        }

        let html = `
            <div class="text-summary">
                <div class="text-stat">
                    <div class="text-stat-value">${data.results.length}</div>
                    <div class="text-stat-label">Results</div>
                </div>

                <div class="text-stat">
                    <div class="text-stat-value green">${data.search_time_ms} ms</div>
                    <div class="text-stat-label">Search Time</div>
                </div>

                <div class="text-stat">
                    <div class="text-stat-value blue">${data.candidates_searched.toLocaleString()}</div>
                    <div class="text-stat-label">Candidates / ${data.total_documents.toLocaleString()}</div>
                </div>
            </div>

            <div class="algorithm-note">
                <b>${data.algorithm.toUpperCase()} search:</b>
                Query converted to a ${data.vector_dimension}D vector.
                IVF-Flat searches only the selected cluster candidates and
                ranks them using cosine similarity.
            </div>
        `;

        data.results.forEach((item, index) => {

            html += `
                <div class="text-result-card">

                    <div class="text-result-top">

                        <span class="text-rank">
                            #${index + 1} · Document ${item.id}
                        </span>

                        <span class="text-score">
                            Score ${item.score.toFixed(4)}
                        </span>

                    </div>

                    <div class="text-content">
                        ${item.text}
                    </div>

                    <div class="text-meta">
                        Cosine similarity · 64-dimensional vector
                    </div>

                </div>
            `;

        });

        output.innerHTML = html;

        status.textContent =
            "✓ Text search completed successfully.";

    }
    catch (error) {

        output.textContent = error.message;

        status.textContent =
            "✗ Text search failed.";

    }

}


// ============================================================
// BENCHMARK
// ============================================================


function drawParetoChart(experiments) {

    const svg = document.getElementById("paretoChart");

    if (!svg || !experiments || experiments.length === 0) {
        return;
    }

    const width = 900;
    const height = 430;

    const margin = {
        left: 75,
        right: 35,
        top: 45,
        bottom: 62
    };

    const plotWidth =
        width - margin.left - margin.right;

    const plotHeight =
        height - margin.top - margin.bottom;

    const points = experiments.map(item => ({
        nprobe: item.nprobe,
        recall: item.recall_at_10,
        qps: item.search_time_ms > 0
            ? 1000 / item.search_time_ms
            : 0
    }));

    // A point is dominated if another point has >= recall and >= QPS,
    // with at least one strict improvement.
    const dominated = points.map((point, i) => {

        return points.some((other, j) => {

            if (i === j) {
                return false;
            }

            return (
                other.recall >= point.recall &&
                other.qps >= point.qps &&
                (
                    other.recall > point.recall ||
                    other.qps > point.qps
                )
            );

        });

    });

    const frontier = points
        .filter((_, i) => !dominated[i])
        .sort((a, b) => a.qps - b.qps);

    const minQps = Math.min(...points.map(p => p.qps));
    const maxQps = Math.max(...points.map(p => p.qps));

    const minRecall = Math.min(...points.map(p => p.recall));
    const maxRecall = Math.max(...points.map(p => p.recall));

    const qpsPad =
        Math.max((maxQps - minQps) * 0.10, maxQps * 0.05);

    const recallPad = 1.5;

    const qpsMin = Math.max(0, minQps - qpsPad);
    const qpsMax = maxQps + qpsPad;

    const recallMin = Math.max(0, minRecall - recallPad);
    const recallMax = Math.min(100, maxRecall + recallPad);

    function x(qps) {
        return margin.left +
            ((qps - qpsMin) / (qpsMax - qpsMin)) *
            plotWidth;
    }

    function y(recall) {
        return margin.top +
            plotHeight -
            ((recall - recallMin) / (recallMax - recallMin)) *
            plotHeight;
    }

    function fmt(value) {
        if (value >= 1000) {
            return (value / 1000).toFixed(1) + "K";
        }
        return Math.round(value).toString();
    }

    let html = "";

    // Background and axes.
    html += `
        <rect x="0" y="0" width="${width}" height="${height}"
              fill="#090d18"></rect>
    `;

    const gridCount = 5;

    for (let i = 0; i <= gridCount; i++) {

        const yy =
            margin.top +
            (plotHeight / gridCount) * i;

        const recallValue =
            recallMax -
            ((recallMax - recallMin) / gridCount) * i;

        html += `
            <line x1="${margin.left}"
                  y1="${yy}"
                  x2="${width - margin.right}"
                  y2="${yy}"
                  stroke="#252d43"
                  stroke-width="1"></line>

            <text x="${margin.left - 12}"
                  y="${yy + 4}"
                  text-anchor="end"
                  fill="#8f9ab4"
                  font-size="11">
                ${recallValue.toFixed(1)}%
            </text>
        `;
    }

    for (let i = 0; i <= gridCount; i++) {

        const xx =
            margin.left +
            (plotWidth / gridCount) * i;

        const qpsValue =
            qpsMin +
            ((qpsMax - qpsMin) / gridCount) * i;

        html += `
            <line x1="${xx}"
                  y1="${margin.top}"
                  x2="${xx}"
                  y2="${height - margin.bottom}"
                  stroke="#1d2539"
                  stroke-width="1"></line>

            <text x="${xx}"
                  y="${height - margin.bottom + 22}"
                  text-anchor="middle"
                  fill="#8f9ab4"
                  font-size="11">
                ${fmt(qpsValue)}
            </text>
        `;
    }

    // Axes.
    html += `
        <line x1="${margin.left}"
              y1="${height - margin.bottom}"
              x2="${width - margin.right}"
              y2="${height - margin.bottom}"
              stroke="#66718a"
              stroke-width="1.5"></line>

        <line x1="${margin.left}"
              y1="${margin.top}"
              x2="${margin.left}"
              y2="${height - margin.bottom}"
              stroke="#66718a"
              stroke-width="1.5"></line>

        <text x="${width / 2}"
              y="${height - 13}"
              text-anchor="middle"
              fill="#d7deee"
              font-size="13"
              font-weight="700">
            QPS (higher is better)
        </text>

        <text x="18"
              y="${height / 2}"
              text-anchor="middle"
              transform="rotate(-90 18 ${height / 2})"
              fill="#d7deee"
              font-size="13"
              font-weight="700">
            Recall@10 (higher is better)
        </text>
    `;

    // Frontier line.
    if (frontier.length > 1) {

        const path = frontier.map((point, i) => {

            const command = i === 0 ? "M" : "L";

            return `${command} ${x(point.qps)} ${y(point.recall)}`;

        }).join(" ");

        html += `
            <path d="${path}"
                  fill="none"
                  stroke="#d92f2f"
                  stroke-width="2.5"
                  stroke-linejoin="round"
                  stroke-linecap="round"></path>
        `;
    }

    // Points and labels.
    points.forEach((point, i) => {

        const isFrontier = !dominated[i];

        const cx = x(point.qps);
        const cy = y(point.recall);

        // Stagger labels so nearby benchmark points do not overlap.
        const labelOffsets = [22, -18, 36, -32, 50];
        const offset = labelOffsets[i % labelOffsets.length];

        let labelX = cx + 10;
        let anchor = "start";

        // Keep labels inside the chart boundaries.
        if (labelX > width - margin.right - 70) {
            labelX = cx - 10;
            anchor = "end";
        }

        let labelY = cy + offset;

        if (labelY < margin.top + 8) {
            labelY = cy + 22;
        }

        if (labelY > height - margin.bottom - 8) {
            labelY = cy - 18;
        }

        html += `
            <circle cx="${cx}"
                    cy="${cy}"
                    r="7"
                    fill="${isFrontier ? "#d92f2f" : "#858585"}"
                    stroke="#090d18"
                    stroke-width="2">
                <title>
                    nprobe=${point.nprobe}
                    | Recall@10=${point.recall.toFixed(2)}%
                    | QPS=${point.qps.toFixed(0)}
                </title>
            </circle>

            <text x="${labelX}"
                  y="${labelY}"
                  text-anchor="${anchor}"
                  fill="${isFrontier ? "#e7ebf7" : "#8f9ab4"}"
                  font-size="11"
                  font-weight="${isFrontier ? "700" : "500"}">
                nprobe=${point.nprobe}
            </text>
        `;
    });

    svg.innerHTML = html;
}


async function runBenchmark() {

    const status =
        document.getElementById(
            "benchmarkStatus"
        );

    const table =
        document.getElementById(
            "benchmarkTable"
        );


    status.textContent =
        "Running 500 queries... Please wait.";

    table.innerHTML = "";


    try {

        const response =
            await fetch("/benchmark");


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail || "Benchmark failed"
            );

        }


        const exactTime =
            data.exact_ground_truth
                .average_search_time_ms;


        document.getElementById(
            "exactTime"
        ).textContent =
            exactTime.toFixed(3) + " ms";


        document.getElementById(
            "exactCandidates"
        ).textContent =
            data.dataset.vectors
                .toLocaleString();


        const best =
            data.ivf_experiments.find(
                item => item.nprobe === 5
            );


        if (best) {

            document.getElementById(
                "speedup"
            ).textContent =
                best.speedup.toFixed(2) + "×";


            document.getElementById(
                "recall"
            ).textContent =
                best.recall_at_10.toFixed(2) + "%";


            document.getElementById(
                "candidates"
            ).textContent =
                best.average_candidates
                    .toLocaleString();

        }


        let html = `

            <table>

                <tr>

                    <th>nprobe</th>
                    <th>Search Time</th>
                    <th>Recall@10</th>
                    <th>Speedup</th>
                    <th>Candidates</th>

                </tr>

        `;


        for (
            const item of data.ivf_experiments
        ) {

            html += `

                <tr>

                    <td>
                        <b>${item.nprobe}</b>
                    </td>

                    <td>
                        ${item.search_time_ms} ms
                    </td>

                    <td>
                        ${item.recall_at_10}%
                    </td>

                    <td class="green">
                        <b>${item.speedup}×</b>
                    </td>

                    <td>
                        ${item.average_candidates.toLocaleString()}
                    </td>

                </tr>

            `;

        }


        html += "</table>";


        html += `
            <br>
            <h3>Recall@10</h3>
        `;


        for (
            const item of data.ivf_experiments
        ) {

            html += `

                <div class="bar-row">

                    <div class="bar-label">

                        <span>
                            nprobe ${item.nprobe}
                        </span>

                        <span>
                            ${item.recall_at_10}%
                        </span>

                    </div>


                    <div class="bar-bg">

                        <div
                            class="bar"
                            style="
                                width:
                                ${item.recall_at_10}%;
                            "
                        >
                        </div>

                    </div>

                </div>

            `;

        }


        table.innerHTML = html;

        drawParetoChart(data.ivf_experiments);


        status.textContent =
            "✓ Benchmark completed successfully.";

    }
    catch (error) {

        status.textContent =
            "✗ Benchmark failed: " +
            error.message;

    }

}


// ============================================================
// START DASHBOARD
// ============================================================

loadStats();

runBenchmark();


</script>

</body>

</html>
"""



# ============================================================
# TEXT SEARCH API
# ============================================================

@app.post("/text-search")
def text_search(request: TextSearchRequest):

    query_vector = text_to_vector(
        request.query,
        dimension=DIMENSION
    )

    if np.linalg.norm(query_vector) == 0:
        raise HTTPException(
            status_code=400,
            detail="Query could not be converted into a non-zero vector"
        )

    start = time.perf_counter()

    try:

        algorithm = request.algorithm.lower()

        if algorithm == "exact":

            results = text_exact_index.search(
                query_vector,
                k=request.k
            )

            candidates = text_exact_index.count()

        elif algorithm == "ivf":

            results = text_ivf_index.search(
                query_vector,
                k=request.k,
                nprobe=request.nprobe
            )

            candidates = text_ivf_index.get_candidate_count(
                query_vector,
                nprobe=request.nprobe
            )

        else:

            raise HTTPException(
                status_code=400,
                detail="Algorithm must be 'exact' or 'ivf'"
            )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        formatted_results = []

        for result in results:
            doc_id = result["id"]

            formatted_results.append({
                "id": doc_id,
                "score": result["score"],
                "text": text_documents[doc_id]
            })

        return {
            "query": request.query,
            "algorithm": algorithm,
            "results": formatted_results,
            "search_time_ms": round(elapsed, 3),
            "candidates_searched": candidates,
            "total_documents": len(text_documents),
            "vector_dimension": DIMENSION
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ============================================================
# SEARCH API
# ============================================================

@app.post("/search")
def search(request: SearchRequest):

    try:

        validate_vector(request.vector)

        start = time.perf_counter()

        algorithm = request.algorithm.lower()


        if algorithm == "exact":

            results = exact_index.search(
                request.vector,
                k=request.k
            )

            candidates = exact_index.count()


        elif algorithm == "ivf":

            results = ivf_index.search(
                request.vector,
                k=request.k,
                nprobe=request.nprobe
            )

            candidates = ivf_index.get_candidate_count(
                request.vector,
                nprobe=request.nprobe
            )


        else:

            raise HTTPException(
                status_code=400,
                detail="Algorithm must be 'exact' or 'ivf'"
            )


        elapsed = (
            time.perf_counter() - start
        ) * 1000


        return {

            "algorithm": algorithm,

            "results": results,

            "search_time_ms":
                round(elapsed, 3),

            "candidates_searched":
                candidates,

            "total_vectors":
                exact_index.count()

        }


    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ============================================================
# INSERT API
# ============================================================

@app.post("/insert")
def insert(request: InsertRequest):

    try:

        validate_vector(request.vector)

        vector = np.array(
            request.vector,
            dtype=np.float32
        )


        vector_id = ivf_index.insert(
            vector
        )


        exact_index.insert(
            vector_id,
            vector
        )


        return {

            "message":
                "Vector inserted successfully",

            "id":
                vector_id

        }


    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ============================================================
# DELETE API
# ============================================================

@app.delete("/delete/{vector_id}")
def delete(vector_id: int):

    exact_deleted = exact_index.delete(vector_id)

    ivf_deleted = ivf_index.delete(vector_id)


    if not exact_deleted and not ivf_deleted:

        raise HTTPException(
            status_code=404,
            detail="Vector not found"
        )


    return {

        "message":
            "Vector deleted successfully",

        "id":
            vector_id

    }


# ============================================================
# STATS API
# ============================================================

@app.get("/stats")
def stats():

    return {

        "vectors":
            exact_index.count(),

        "dimension":
            DIMENSION,

        "clusters":
            NUM_CLUSTERS,

        "ivf_vectors":
            ivf_index.count(),

        "exact_vectors":
            exact_index.count(),

        "deleted_vectors":
            len(ivf_index.deleted_ids),

        "text_documents":
            len(text_documents),

        "text_dimension":
            DIMENSION,

        "text_clusters":
            TEXT_CLUSTERS

    }


# ============================================================
# SAMPLE VECTOR API
# ============================================================

@app.get("/sample/{vector_id}")
def sample(vector_id: int):

    if (
        vector_id < 0
        or vector_id >= len(vectors)
    ):

        raise HTTPException(
            status_code=404,
            detail="Vector ID not found"
        )


    return {

        "id":
            vector_id,

        "vector":
            vectors[vector_id].tolist()

    }


# ============================================================
# BENCHMARK API
# ============================================================

@app.get("/benchmark")
def benchmark():

    np.random.seed(123)


    query_ids = np.random.choice(
        len(vectors),
        size=500,
        replace=False
    )


    queries = vectors[query_ids]


    # --------------------------------------------------------
    # Exact ground truth
    # --------------------------------------------------------

    ground_truth = []

    exact_times = []


    for query in queries:

        start = time.perf_counter()

        results = exact_index.search(
            query,
            k=10
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000


        exact_times.append(elapsed)

        ground_truth.append(results)


    avg_exact_time = float(
        np.mean(exact_times)
    )


    # --------------------------------------------------------
    # IVF experiments
    # --------------------------------------------------------

    nprobe_values = [
        1,
        5,
        10,
        20,
        50
    ]


    experiments = []


    for nprobe in nprobe_values:

        ivf_times = []

        recalls = []

        candidate_counts = []


        for i, query in enumerate(queries):

            start = time.perf_counter()

            results = ivf_index.search(
                query,
                k=10,
                nprobe=nprobe
            )

            elapsed = (
                time.perf_counter() - start
            ) * 1000


            ivf_times.append(elapsed)


            exact_ids = {
                result["id"]
                for result in ground_truth[i]
            }


            ivf_ids = {
                result["id"]
                for result in results
            }


            common = exact_ids & ivf_ids


            recall = len(common) / 10

            recalls.append(recall)


            candidate_count = ivf_index.get_candidate_count(
                query,
                nprobe=nprobe
            )


            candidate_counts.append(
                candidate_count
            )


        avg_ivf_time = float(
            np.mean(ivf_times)
        )


        avg_recall = float(
            np.mean(recalls) * 100
        )


        avg_candidates = float(
            np.mean(candidate_counts)
        )


        speedup = avg_exact_time / avg_ivf_time


        experiments.append({

            "nprobe":
                nprobe,

            "search_time_ms":
                round(
                    avg_ivf_time,
                    3
                ),

            "recall_at_10":
                round(
                    avg_recall,
                    2
                ),

            "speedup":
                round(
                    speedup,
                    2
                ),

            "average_candidates":
                round(
                    avg_candidates
                )

        })


    return {

        "dataset": {

            "vectors":
                len(vectors),

            "dimension":
                vectors.shape[1],

            "clusters":
                len(centers)

        },

        "queries":
            500,

        "exact_ground_truth": {

            "average_search_time_ms":
                round(
                    avg_exact_time,
                    3
                )

        },

        "ivf_experiments":
            experiments

    }

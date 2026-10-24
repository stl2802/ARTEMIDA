# ARTEMIDA — Hybrid Search Engine

## About the System

ARTEMIDA (Advanced Retrieval Through Enhanced Multi-method Intelligent Document Architecture) is a hybrid search engine that combines multiple approaches to information retrieval to achieve maximum relevance of results.

Traditional search engines use a single search method:
- **Lexical search** (BM25) finds documents with exact word matches but doesn't understand synonyms or semantics
- **Vector search** (TF-IDF) considers word frequency but doesn't capture semantics
- **Neural search** understands meaning but may miss precise terminology matches

### Solution

ARTEMIDA combines all these methods and adds machine learning for intelligent ranking:

1. **Lexical search (BM25)** — finds documents with exact terms
2. **Vector search (TF-IDF + SVD)** — cosine similarity in vector space
3. **Neural search (multilingual-e5)** — semantic understanding, including Russian language
4. **HNSW index** — fast approximate vector search
5. **Hybrid search** — combines BM25 and cosine similarity with configurable weight α
6. **Learning to Rank (LTR)** — gradient boosting model (LightGBM) for reranking results
7. **Smart Hybrid** — automatic optimal method selection based on query analysis

### Architecture

```
User Query
    ↓
[Query Analyzer] — analyzes language, length, query type
    ↓
┌─────────────────────────────────────────────┐
│  Automatic Method Selection                 │
│  - Russian text      → Neural (multilingual)│
│  - Short term        → Lexical (BM25)       │
│  - Meaning-based     → LTR or Hybrid        │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  Search Methods:                            │
│  1. Lexical: inverted index + BM25         │
│  2. TF-IDF: cosine similarity of vectors   │
│  3. Neural: sentence-transformers          │
│  4. HNSW: approximate vector search        │
│  5. Hybrid: BM25 + Cosine (parameter α)    │
│  6. LTR: LightGBM on multiple features     │
└─────────────────────────────────────────────┘
    ↓
[Ranking] — combines and sorts results
    ↓
Top-5 relevant documents
```

### Document Corpus

The document text used is the abstract (abstract) from an arXiv scientific article. The abstract contains key terms and the semantic core of the work, which is sufficient for information retrieval tasks, while avoiding licensing issues with CC-BY licenses on full PDF texts.

## Getting Started with Your Own Queries

### Working with Custom Queries

To use the system with your own search queries, you need to create or modify the file:

```
generated_data/json/queries.json
```

### Query File Structure

```json
{
    "queries": {
        "exact_terms": [
            "neural networks",
            "machine learning", 
            "deep learning",
            "natural language processing"
        ],
        "detailed_queries": [
            "What are the applications of neural networks in computer vision?",
            "How does gradient boosting algorithm work in ensemble methods?",
            "Explain transformer architecture in natural language processing"
        ],
        "short_queries": [
            "ML algorithms",
            "AI ethics",
            "data science"
        ],
        "error_queries": [
            "optimization techniques for training large scale models",
            "comparison of supervised and unsupervised learning"
        ]
    }
}
```

## Usage

All commands:

```bash
# Full project initialization from scratch
python main.py init

# Create literature documents from your sources
python main.py create-data

# Generate test queries
python main.py create-queries

# Process corpus to generated_data/json/corpus.json
python main.py corpus

# Preprocess corpus
python main.py preprocess

# Build vocabulary
python main.py vocabulary

# Lexical search (BM25)
python main.py lexical build    # build index
python main.py lexical search   # search

# TF-IDF search
python main.py tfidf build     # build vectors
python main.py tfidf svd -k 100  # apply SVD
python main.py tfidf search    # search

# Neural search
python main.py neural build    # build vectors
python main.py neural search   # search

# HNSW index
python main.py hnsw build      # build index
python main.py hnsw search     # search

# Hybrid search
python main.py hybrid build    # build indices
python main.py hybrid search -a 0.7  # search with alpha=0.7

# Smart hybrid search (auto-select: neural/lexical/hybrid/ltr/hnsw)
python main.py smart-hybrid

# Learning to Rank
python main.py ltr prepare   # prepare dataset
python main.py ltr train     # train model
python main.py ltr search    # search

# Build all indices at once
python main.py index-all -k 100

# Benchmarks
python main.py benchmark alpha    # alpha parameter impact
python main.py benchmark hnsw     # HNSW vs exact search
python main.py benchmark ltr      # LTR vs hybrid
```

## LTR (Learning to Rank)

LTR uses machine learning to rerank search results by learning from relevance labels.

### Features Used

1. **bm25_score** — BM25 relevance score
2. **tfidf_cosine** — TF-IDF cosine similarity
3. **neural_cosine** — Neural embedding cosine similarity
4. **word_overlap_text** — Word overlap in text
5. **word_overlap_title** — Word overlap in title
6. **doc_length** — Document length

### Training Process

1. **Dataset Preparation** — Generate feature vectors for query-document pairs
2. **Train/Test Split** — Split data for training and evaluation
3. **Model Training** — LightGBM gradient boosting on features
4. **Evaluation** — Precision@5, nDCG@5 on test set

### LTR Advantages

- **Multiple factor consideration** — model considers all features simultaneously
- **Learnable** — quality improves with more labeled data
- **Interpretable** — can see feature importance

## Project Structure

```
.
├── literature/                  # Source documents (doc_*.txt)
├── create_data/                 # Corpus generation and processing
│   ├── literature_installer.py # Generate test documents
│   └── process_corpus.py       # Parse documents
├── preprocess_corpus.py        # Text preprocessing
├── searches/                    # Search methods
│   ├── lecsika_search.py       # Lexical search (BM25)
│   ├── tfidf_search.py         # TF-IDF + SVD
│   ├── neural_search.py        # Neural search
│   ├── hnsw_search.py          # HNSW index
│   ├── hybrid_search.py        # Hybrid search
│   ├── ltr_search.py           # Learning to Rank
│   └── smart_hybrid_search.py  # Smart hybrid search
├── vectoryzing/                 # Vectorization
│   ├── vectorize_corpus.py     # TF-IDF vectorization
│   └── query_vectorizer.py     # Query vectorization
├── neural_train_process_data/   # LTR data preparation
│   ├── prepare_ltr_dataset.py  # Generate dataset
│   ├── split_dataset.py        # Train/test split
│   └── feature_extractor.py    # Feature computation
├── benchmarks/                  # Benchmarks
│   ├── benchmark_alpha.py      # Impact of parameter α
│   ├── benchmark_hnsw.py       # HNSW vs exact search
│   └── benchmark_ltr.py        # LTR vs hybrid
├── main.py                     # CLI interface
└── generated_data/              # Generated data
    ├── json/                   # JSON index files
    ├── npy/                    # NumPy vector arrays
    ├── model/                  # ML models
    └── bin/                    # HNSW index
```

## Why This System?

### Limitations of Traditional Approaches

| Method | Advantages | Disadvantages |
|--------|-----------|---------------|
| **BM25** | Finds exact word matches, fast | Doesn't understand synonyms, no semantics |
| **TF-IDF** | Considers word frequency | Doesn't understand meaning, ignores word order |
| **Neural** | Understands semantics, works with synonyms | May miss exact terms, requires resources |
| **HNSW** | Fast search on large vectors | Approximate method, accuracy loss |

### What ARTEMIDA Provides

1. **Combining strengths** — each method compensates for others' weaknesses
2. **Adaptivity** — system selects best method for each query
3. **Machine learning** — LTR model learns from data and improves over time
4. **Multilingual** — works with Russian and English languages
5. **Research potential** — benchmarks for comparing methods

### Applications

- **Scientific search** — search through scientific articles and documents
- **Corporate knowledge bases** — intelligent search within organization documents
- **Educational systems** — search for learning materials
- **IR research** — platform for experimenting with search methods

## Dependencies

- `nltk` — tokenization and stemming
- `sentence-transformers` — neural vectorization (multilingual-e5)
- `hnswlib` — HNSW index for fast vector search
- `lightgbm` — gradient boosting for LTR
- `numpy` — vector operations
- `scikit-learn` — TF-IDF, SVD, metrics
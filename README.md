# Offer retrieval: Fetch take-home

A take-home NLP implementation that ranks retail offers from query text using TF-IDF and cosine similarity. Category and brand metadata are joined before indexing; a string-similarity heuristic removes near-duplicate offer text.

## Verified state, 2026-09-29

The existing core ran in a controlled Python 3.12 environment with corpus downloads prepared separately and the script's runtime package-upgrade command suppressed. The Dash page returned HTTP 200. A Target query returned 20 candidates. Empty input raised an empty-vocabulary exception. Huggies and diapers returned no candidates in this snapshot; this is not evidence of retrieval quality.

This is a rehabilitation candidate, not one of the completed decision-system releases. The original notebook also contains a stored NameError for an undefined df. No precision/recall or latency benchmark is claimed.

## Current execution path

The historical entry point is final.py. It installs/upgrades packages and downloads NLTK data at launch, so its environment is uncontrolled. The pinned audit environment used Dash 2.14.1, pandas 2.1.3, NumPy 1.26.4, NLTK 3.8.1, scikit-learn 1.5.2 and dash-bootstrap-components 1.5.0. NLTK punkt, stopwords and wordnet corpora were required.

The source and CSVs are retained without changing their existing rights. Dataset provenance and code licensing must be established before packaging this as a redistributable library.

## Repair sequence

1. Remove runtime dependency installation; add a controlled environment and an app factory.
2. Make CSV paths relative to the source file and initialize the index for WSGI startup.
3. Handle empty queries, add input/schema checks, and test deduplication.
4. Add a labeled query set with Recall@k and ranking-quality evaluation.
5. Index metadata once, measure query latency, and retain the smallest architecture supported by measurements.

The original implementation is in [final.py](final.py), notebook in [final.ipynb](final.ipynb), and demonstration in [Walkthrough.mp4](Walkthrough.mp4). This README describes verified behavior; it does not claim prompt engineering or transformer retrieval is implemented.

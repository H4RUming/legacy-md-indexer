# legacy-md-indexer

## Overview
A high-throughput **Hierarchical RAG Pipeline** designed for metadata extraction and intelligent querying across 9,000+ unstructured legacy documents. To resolve the inference bottlenecks of 32B+ VLM architectures, this system utilizes a lightweight text-LLM setup with pre-parsed Markdown (`processed_md/`), maximizing tokens per second (tk/s) while maintaining high retrieval precision.

---

## Core Pipeline

### Phase 0: Document ETL (Pre-requisite)
Transforms raw, multi-format legacy documents (PDF, DOCX, XLSX, etc.) into **structured Markdown** optimized for LLM ingestion.
- **Strict Layout Preservation:** Minimizes data loss by preserving table structures and document hierarchies.
- **OCR Integration:** Handles scanned documents via `EasyOCR` and `docling` with multi-language support (KO/EN).
- **Resource Isolation:** Prevents memory leaks during massive batch conversions using `multiprocessing.Pool` with isolated worker limits.

### Phase 1: Metadata Cataloging
Extracts metadata from all documents to generate a `file_catalog.json` for high-speed pre-filtering.
- **Hybrid Extraction:** A 2-track system combining **Regex-based fast-track** for patterns (dates, IDs) and **LLM-based dynamic taxonomy** for complex category classification.

### Phase 2: Agentic Router (Pre-filtering)
Analyzes natural language queries to dynamically extract search parameters.
- **Parameter Extraction:** Isolates specific timeframes (`years`, `months`) and the core `search_query` from the user's input.
- **Metadata Filtering:** Drastically narrows the search space from 9,000+ files to a handful of target documents before the heavy retrieval phase.

### Phase 3: Integrated Retrieval & RAG
Executes full-text search on filtered documents and generates the final response.
- **Lexical Search (BM25):** Utilizes `rank_bm25` for keyword-based ranking across the target document subset.
- **Morphological Analysis:** Integrated **Kiwi (kiwipiepy)** tokenizer to remove Korean particles (Josa) and symbols, focusing on substantive morphemes to boost retrieval accuracy.
- **128k Context Window:** Supports up to 131,072 tokens with intelligent truncation logic to prevent OOM.
- **Streaming Inference:** Real-time response generation via Ollama API for an optimized user experience.

---

## System Engineering & Stability

* **Crash-safe State Management:** Executes synchronous state saves immediately after processing each file, ensuring zero data loss and instant resume capability.
* **Context Defense:** Hard-coded 160,000-character truncation guardrail to prevent context overflow and memory exhaustion.
* **Reasoning Model (CoT) Defense:** Pre-emptive regex stripping of `<think>...</think>` tags to ensure consistent JSON parsing when using reasoning models like `qwen2.5`.
* **Gradio 5.x Integration:** Full support for the latest Gradio message formats, including streaming outputs and detailed interaction logging.

---

## Tech Stack
- **Language:** Python 3.10+
- **LLM Engine:** Ollama (GPT-OSS 120B)
- **NLP / Retrieval:** Kiwipiepy (Kiwi), Rank-BM25
- **ETL:** Docling, EasyOCR, Pandas
- **Frontend:** Gradio 5.x

---

## Future Scope
- **Multi-modal Re-ranking:** Implementing a VLM-based secondary verification for documents containing complex diagrams or blueprints.
- **Hybrid Vector Search:** Combining BM25 lexical search with Dense Vector embeddings to capture semantic nuances in unstructured text.

---
title: DocFinder AI
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---
# DocFinder AI

### RAG Based AI Assistant

DocFinder AI is a Retrieval-Augmented Generation (RAG) application
that allows users to upload documents and ask questions about their
content using natural language.

## Features

- PDF document support
- DOCX document support
- TXT document support
- OCR for scanned PDFs
- Semantic search
- BM25 keyword retrieval
- Hybrid retrieval
- Reciprocal Rank Fusion
- Optional document reranking
- RAG-based question answering
- Document summarization
- Source/page references
- Enter-to-search interface
- Modern Gradio UI

## Architecture

```text
Document
   |
   v
Text Extraction / OCR
   |
   v
Text Splitting
   |
   v
Embeddings
   |
   +----------------+
   |                |
   v                v
 FAISS             BM25
 Semantic          Keyword
 Search            Search
   |                |
   +-------+--------+
           |
           v
   Reciprocal Rank Fusion
           |
           v
       Reranking
           |
           v
   Relevant Context
           |
           v
          LLM
           |
           v
    Grounded Answer
           |
           v
    Source References
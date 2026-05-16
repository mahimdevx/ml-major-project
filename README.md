# Shakespeare SLM RAG Project

A Retrieval-Augmented Generation (RAG) based Small Language Model system for Shakespeare domain adaptation.

## Project Objective

This project implements a Shakespeare-aware assistant capable of:

- beginner-friendly explanations
- contextual question answering
- evidence retrieval
- Shakespearean style generation

The system is designed for the CSCI433/933 Machine Learning Algorithms and Applications Assignment 2.

## Repository Structure

```text
ml-major-project/
├── data/
├── src/
├── prompts/
├── results/
├── report/
├── docs/
└── demo/
```

## Core Technologies

- Python
- FAISS
- sentence-transformers
- HuggingFace Transformers
- Retrieval-Augmented Generation (RAG)

## Setup

```bash
pip install -r requirements.txt
```

## Recommended Workflow

```bash
git checkout -b develop
```

Create feature branches:

```bash
feature/preprocessing
feature/retrieval
feature/rag-pipeline
feature/evaluation
feature/report
```

## Team Roles

- Dataset Engineering
- Retrieval & Embeddings
- Generation & Prompting
- Evaluation & Reporting

## Assignment Requirements

The project follows the assignment requirements including:

- structured Shakespeare dataset
- embedding-based retrieval
- grounded evidence generation
- beginner-friendly explanations
- evaluation and failure analysis


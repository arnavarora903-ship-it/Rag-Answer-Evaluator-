# RAG Answer Evaluator

## Overview
A Retrieval-Augmented Generation (RAG) system that answers questions from documents and evaluates answer reliability.

## Features
- Document-based Q&A using embeddings  
- Cosine similarity retrieval  
- LLM-based evaluation layer  
- Multi-step pipeline using LangChain LCEL  

## Workflow Design
This project is structured as a multi-step pipeline:
1. Document ingestion and chunking  
2. Embedding and retrieval  
3. Answer generation  
4. Answer evaluation  

## Tech Stack
Python, LangChain, Chroma, FAISS, Streamlit  

## Status
Completed and fully functional

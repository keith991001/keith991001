# KGSynX: Knowledge Graph and SHAP-Driven Synthetic Data Generation

This repository provides the core implementation of the KGSynX pipeline, a framework combining knowledge graph embeddings and explainable feedback for improving tabular synthetic data.

## Modules

- `data_loader.py`: loads and preprocesses UCI heart data.
- `kg_builder.py`: builds a graph representation of structured data.
- `embedding.py`: computes Node2Vec-based node embeddings.
- `shap_analysis.py`: evaluates real vs synthetic SHAP feature attributions.
- `prompt_utils.py`: constructs generation prompts from graph structure.
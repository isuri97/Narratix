# LiLADH: An Open Retrieval Resource for Digital Humanities Archival Corpora

This module contains the retrieval adapter component of **LiLADH: An Open Retrieval Resource for Digital Humanities Archival Corpora**. It implements a lightweight bottleneck adapter that transforms *query* embeddings to better align with a domain-specific archival corpus (Holocaust survivor testimonies), while document embeddings from the frozen base model remain untouched.

## Contents

```
Adapter/
├── code/
│   └── DQola adapter.py        # Adapter model, training loop, evaluation, experiment runner
├── prompt/
│   └── query_generation_prompt.txt   # Prompt used to synthetically generate benchmark queries
├── data/
│   └── cleaned_transformed_data.json # Cleaned query/passage benchmark used for adapter training & eval
└── Readme.md                  

```
## What this adapter does

```
Query → Base Embedding Model (frozen) → Linear Adapter (trainable) → Adapted query embedding
                                          384 → 128 → 384
                                          (bottleneck, residual connection,
                                           GELU, LayerNorm, L2-normalised output)
```

Only the query side is adapted; the document/passage corpus is embedded once with the frozen base model and never modified. This keeps the adapter
lightweight and makes it usable as a drop-in layer in front of any of the evaluated base models.

**Training objective:** hybrid loss combining InfoNCE (in-batch negatives, τ = 0.05) and a weighted Triplet Margin Loss (hard negatives, margin = 0.5,
weight = 0.3). Optimised with AdamW (lr = 1e-4), linear warmup (100 steps), gradient clipping (max norm 1.0), for 10 epochs (10/30/40 were tested; no
gain observed beyond 10).

**Base models evaluated:**
- `all-MiniLM-L6-v2`
- `all-MiniLM-L12-v2`
- `multi-qa-mpnet-base-dot-v1`
- `stsb-roberta-base`

**Evaluation metrics:** Hit@1, Hit@5, Hit@10, MRR — reported for each base model, comparing baseline retrieval against adapter-transformed queries.

## Data format

The training/validation files are JSONL, one record per passage chunk:

```json
{
  "chunk_index": 0,
  "input_chunk": "testimony passage text ...",
  "questions": {
    "question_1": "Who was deported from ...",
    "question_2": "When did the witness ...",
    "...": "... up to question_20"
  }
}
```
## Usage

```bash
python "code/DQola adapter.py" \
    --train_path  data/filtered_chunks_train.json \
    --val_path    data/filtered_chunks_test.json \
    --neg_path    data/nivdia.txt \
    --output_dir  outputs/
```

Queries were synthetically generated per chunk using the prompt in `prompt/query_generation_prompt.txt`, targeting named entities (dates, people, places, organisations, camps, ghettos, etc.) mentioned in the passage. A subset was subsequently checked by human annotators for grammaticality, factual grounding, and usefulness as a retrieval query.


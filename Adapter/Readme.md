# LiLADH: An Open Retrieval Resource for Digital Humanities Archiva# Adapter — DsQoLA (Domain-specific Query-Only Linear Adapter)

This module contains the retrieval adapter component of **LiLADH: An Open Retrieval Resource for Digital Humanities Archival Corpora**. It implements
a lightweight bottleneck adapter that transforms *query* embeddings to better align with a domain-specific archival corpus (Holocaust survivor testimonies), while document embeddings from the frozen base model remain untouched.


## Contents

```
Adapter/
├── code/
│   └── DQola adapter.py        # Adapter model, training loop, evaluation, experiment runner
├── prompt/
│   └── query_generation_prompt.txt   # Prompt used to synthetically generate benchmark queries
├── Data/
│   └── knowledge_modelling/    # Per-testimony extracted entity/relationship JSON (1.json–98.json)
├── data/
│   └── cleaned_transformed_data.json # Cleaned query/passage benchmark used for adapter training & eval
└── Readme.md                   # This file
```

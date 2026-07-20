# Dataset Recovery Report

This report summarizes the recovery of conversations from the pipeline and details of the final merged dataset.

## 1. Recovered Conversations & Lineage
* **August 2026 Audit findings**:
  - The clean training dataset (`apps_train_cleaned.json`) contains **3,155** valid DSA-related coding questions, which were completely ignored in the previous run.
  - The previous run used a tiny placeholder of only **2** samples.
* **Lineage & Merge Sources**:
  - **Source 1: Transformed & Augmented SFT** (`data/transformed/train_sft_augmented.json`)
    - Cleaned coding tasks: **3,155**
    - Expanded dialogues (wrong approach, bug diagnosis, pattern recognition, etc.): **25,148**
    - Total SFT records: **28,303**
  - **Source 2: Tutor Corpus** (`dataset/tutor_corpus/dsa_tutor_v1.jsonl`)
    - Procedurally generated tutoring templates covering 21 topics: **210** conversations
  - **Merged Pool**: **28,513** unique, valid tutoring conversations.

## 2. Strategic Stratification (90% / 10% Split)
To evaluate the model's tutoring quality accurately during training, the merged pool is split into a **90% train** and **10% validation** split, ensuring perfect balance across topics, difficulties, and conversation types:
* **Final Train size (`train_v1.jsonl`)**: **25,662** conversations
* **Final Validation size (`validation_v1.jsonl`)**: **2,851** conversations
* **Validation Ratio**: **10.00%**

The merge logic has been corrected in [execute_release.py](file:///C:/Users/Web-wizrd/Desktop/Github/LLM-Dataset-Training/scripts/execute_release.py) to prevent silent data discarding and to maintain precise stratification down to the stratum level.

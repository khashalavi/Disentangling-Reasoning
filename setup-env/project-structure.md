

---

## 📁 Repository Tree Overview

```text
Disentangling-Memory-and-Reasoning/
│
├── load_data/
│   ├── data_agent.py
│   ├── preprocess.py
│   └── README.md
│
├── model/
│   └── (Contains model utilities and architecture definitions)
│
├── train.py
├── train.sh
├── eval.sh
├── requirements.txt
└── README.md
```

---

## 📄 Root Directory Files

### `train.py`
The main Python script for training the model. It handles:
* Loading the base LLM (primarily optimized for Llama models).
* Applying Parameter-Efficient Fine-Tuning (PEFT), specifically combining Prompt Tuning with LoRA.
* Setting up the training loop and loss calculations tailored to explicitly enforce the `{memory}` and `{reason}` generation phases.
* Dynamic accuracy evaluation during the training process.

### `train.sh`
A shell script acting as the entry point for training. 
* Contains the configuration and hyperparameters (e.g., learning rate, batch size, LoRA rank).
* Executes `train.py` with the appropriate command-line arguments.
* *Note: The authors mention that while it works flawlessly for Llama, applying it to Qwen models currently has unresolved bugs.*

### `eval.sh`
The evaluation shell script.
* Used post-training (or during validation) to benchmark the model.
* Triggers inference scripts to test if the model successfully recalls memory first before generating reasoning steps, and calculates standard accuracy metrics.

### `requirements.txt`
The standard Python dependencies file. Defines the required libraries such as `transformers`, `peft`, `torch`, and `datasets` needed to run the environment.

### `README.md`
The main documentation file for the repository, containing setup instructions, conceptual overviews, and citation information for the ACL 2025 paper.

---

## 📂 Directories in Detail

### `load_data/`
This directory is responsible for all data ingestion, formatting, and preprocessing. The method requires specialized training data where Chain-of-Thought (CoT) is explicitly divided into memory recall and reasoning steps.

* **`data_agent.py`**: Interacts with the Hugging Face Hub (requires HF Token). It downloads the raw datasets and acts as the orchestrator for loading training and evaluation sets.
* **`preprocess.py`**: The core data manipulation script. It takes standard QA/CoT datasets and restructures them. It injects the special `{memory}` and `{reason}` tokens, ensuring the inputs and labels are correctly formatted for the specialized fine-tuning task.
* **`README.md`**: An internal guide specifically for the data pipeline, explaining how to add new datasets or modify the preprocessing logic.

### `model/`
This directory contains the custom architectural code that wraps around standard Hugging Face models.
* Includes the logic required to implement the hybrid Prompt Tuning + LoRA approach.
* Defines how the special tokens (`{memory}` and `{reason}`) manipulate the model's forward pass or generation sequence to enforce the disentangled cognitive steps.

---

## 🚀 How It All Connects (The Workflow)

1. **Environment Setup:** Users start by installing `requirements.txt`.
2. **Data Prep:** The workflow moves to `load_data/` where `data_agent.py` and `preprocess.py` fetch and transform the data.
3. **Training:** `train.sh` is executed. It calls `train.py`, which loads the custom architecture from `model/`, consumes the formatted data, and trains the Llama model.
4. **Evaluation:** `eval.sh` is run to measure the model's new disentangled reasoning capabilities.


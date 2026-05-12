# PeptideMTR

This work was developed as a collaboration between **Novo Nordisk** and the **Wilke lab** at **The University of Texas at Austin**.

### Authors
* **Aaron L. Feller** [1,2]* (aaron.feller@utexas.edu)
* **Maxim Secor** [1]
* **Sebastian Swanson** [1]
* **Claus O. Wilke** [2]
* **Kristine Deibler** [1]

**[1] Molecular AI, Novo Nordisk** **[2]; Integrative Biology, The University of Texas at Austin** 

---

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Models](#models)
  - [Tokenizer](#tokenizer)
  - [Example Script](#usage)
  - [Datasets](#datasets)
- [Repository Installation](#repository-installation)
- [Contributing](#contributing)
- [License](#license)

## Introduction
PeptideMTR is transformer-based representation learning suite for therapeutic peptides. The project investigates how explicit physicochemical information (99 RDKit descriptors) used during training can enhance the predictive power of peptide models.

The framework benchmarks three distinct architectural approaches:
1. **MLM (masked language modeling):** Purely sequence-based learning via amino acid tokens.
2. **MTR (multi-task regression)** Regression models trained using a curated set of 99 RDKit physicochemical descriptors.
3. **MLM-MTR (Hybrid):** A dual-objective architecture that leverages both latent sequence patterns and explicit chemical descriptors during the training phase.



## Getting Started

### Models
All 9 model variants associated with the forthcoming paper are hosted on Hugging Face: [huggingface.co/aaronfeller](https://huggingface.co/aaronfeller).

| Model Variant | Strategy | Training Features |
| :--- | :--- | :--- |
| **PeptideCLM-2 MLM** | Sequence Pre-training | Masked SMILES tokens |
| **PeptideCLM-2 MTR** | Multi-Target Regression | 99 RDKit Descriptors |
| **PeptideCLM-2 Hybrid** | Split-Head Architecture | Masked SMILES tokens & 99 RDKit Descriptors |


### Tokenizer
The project utilizes a custom tokenizer optimized for the peptide chemical space. This ensures robust handling of both standard and non-canonical amino acids, facilitating the mapping of SMILES strings to the model's latent space.

### Usage
PeptideMTR models are designed for ease of use. Regardless of the training objective (including the MTR variants), all models accept a **SMILES string** as the primary input for inference. 

If you would like to use the models, they are hosted on Huggingface. An example script is below:

```
from transformers import AutoTokenizer, AutoModel
import torch

# 1. Specify the model repository
model_name = "aaronfeller/peptideclm-2-hybrid-small" # Replace with model of interest

# 2. Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)

# 3. Define your SMILES string (Example: Glycyl-Glycine)
smiles_string = "NCC(=O)NCC(=O)O"

# 4. Tokenize the input
inputs = tokenizer(smiles_string, return_tensors="pt")

# 5. Move model and inputs to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 6. Run inference
with torch.no_grad():
    outputs = model(**inputs)

# 7. Extract the embeddings (last hidden state)
embeddings = outputs.last_hidden_state
print(f"Embedding shape: {embeddings.shape}")
```

### Datasets
The training and validation data used to develop these models—including the 99 pre-computed RDKit descriptors and their corresponding biochemical targets—are available at [PeptideMTR_pretraining_data](https://huggingface.co/datasets/aaronfeller/peptideclm-2-pretraining-data).


## Repository Installation

If you would like to set up an environment and use the code for recreating the models, installation instructions are below:

This repository uses `pyproject.toml` for dependency management. We recommend using [uv](https://github.com/astral-sh/uv) for an extremely fast and reproducible setup.

1. **Clone the repository:**
   ```
   git clone https://github.com/aaronfeller/PeptideMTR.git
   cd PeptideMTR
   ```

3. **Install dependencies and create a virtual environment:**
   Using `uv`, you can sync the entire environment in seconds:  
   ```
   uv sync
   ```

5. **Activate the environment:**
   ```
   source .venv/bin/activate  # On macOS/Linux
   .venv\Scripts\activate     # On Windows
   ```

Alternatively, you can install the packages using standard pip:
  ```
  pip install .
  ```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License
The author(s) are protected under the MIT License - see the LICENSE file for details.


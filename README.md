# HEP Minor Module Coursework (A3)

## Overview
This repository contains the code and report for the HEP Minor Module Coursework (A3).

- **A2**: Weight sharing layers for rotation-equivariant image processing
- **B1**: Higgs boson discovery analysis using ATLAS Open Data
- **C2**: Event classification for Z boson decay identification using neural networks

## Repository Structure

```
.
├── A2_weight_sharing_layers/   # Section A implementation
│   ├── A2_weight_sharing.ipynb # Main notebook
│   ├── figures/                # Generated figures
│   ├── src/                    # Source code
│   └── requirements.txt        # Dependencies
├── B1_Higgs/                   # Section B implementation
│   ├── datasets/               # Data storage
│   ├── notebooks/              # Analysis notebooks
│   ├── src/                    # Source code
│   └── requirements.txt        # Dependencies
├── C2_event_classification/    # Section C implementation
│   ├── datasets/               # Data storage
│   ├── notebooks/              # Analysis notebooks
│   ├── run/                    # Scripts to run experiments
│   ├── src/                    # Source code
│   └── requirements.txt        # Dependencies
├── report/                     # PDF report
└── README.md                   # This file
```


## Setup and Installation

### Requirements
- Python 3.8+
- Additional packages as listed in each section's requirements.txt file

### Installation
```bash
# Create a virtual environment (optional)
python -m venv hep_env
source hep_env/bin/activate     # hep_env\Scripts\activate on Windows

# Install dependencies for Section A2
pip install -r A2_weight_sharing_layers/requirements.txt

# Install dependencies for Section B1
pip install -r B1_Higgs/requirements.txt

# Install dependencies for Section C2
pip install -r C2_event_classification/requirements.txt
```

## Report
The final report is available in the `report/` directory.

### Literature Review
- Utilised “Deep research” (OpenAI) and “DeepSearch” (xAI) for high-level overviews and literature searches

### Development Assistance
- Initial code scaffolding and project structure design  
- Template generation for configuration files (`requirements`, `.gitignore`, etc.)  
- Assistance with network architecture design  
- Script prototyping for model training (`.py`, `.sh`)  
- Debugging support and performance optimisation

### Documentation
- README generation and documentation structuring  
- Code commenting and function-level documentation

### Code Quality
- Refactoring suggestions to enhance readability  
- Implementation optimisation

**AI tools used**: Claude (Anthropic), ChatGPT (OpenAI), Grok (xAI)
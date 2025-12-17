# ThoughtProbe

Official code implementation for the EMNLP 2025 main conference paper:

**[ThoughtProbe: Classifier-Guided LLM Thought Space Exploration via Probing Representations](https://aclanthology.org/2025.emnlp-main.307.pdf)**

*Zijian Wang and Chang Xu*

This project explores and analyzes thought processes in language models using classifier-guided tree search decoding and multi-layer probing techniques.

## Environment Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set OpenAI API Key (if using GPT features):**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

The main entry point is `main_leaves.py`, which performs tree search decoding with thought probing:

```bash
python main_leaves.py
```

### Configuration

Edit `config.py` to customize:
- Model selection (e.g., Mistral-7B, Phi-1.5, Gemma-2-2B)
- Dataset selection (GSM8K, MultiArith, SVAMP, etc.)
- Probing method (Logistic Regression, SVM)
- Search parameters and decoding strategies

## Project Structure

- `main_leaves.py` - Main code for tree search decoding with thought probing
- `config.py` - Configuration settings
- `methods.py` - Decoding methods (tree search, greedy, CoT)
- `data_utils.py` - Data loading utilities
- `Prober/` - Thought probing models (LR, SVM)
- `Utils/` - Helper utilities and tools
- `dataset/` - Dataset files

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wang-xu-2025-thoughtprobe,
    title = "{T}hought{P}robe: Classifier-Guided {LLM} Thought Space Exploration via Probing Representations",
    author = "Wang, Zijian  and
      Xu, Chang",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.307/",
    doi = "10.18653/v1/2025.emnlp-main.307",
    pages = "6029--6050",
    ISBN = "979-8-89176-332-6",
}
```

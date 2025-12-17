# Thoughtprobe

A research project for probing and analyzing thought processes in language models using tree search decoding and multi-layer probing techniques.

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

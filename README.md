# Sentiment Analysis of Financial News for Stock Price Prediction

Using pretrained and fine-tuned transformer models to test whether financial news sentiment predicts stock returns.

## Key Finding

Pretrained models (FinBERT, FLANG-RoBERTa) extract sentiment effectively, but that signal doesn't translate into usable trading edge on its own. LoRA fine-tuning FinBERT achieves the best directional accuracy, though all models hover near 50% — consistent with the efficient market hypothesis.

## Results

| Model | Accuracy | F1 | OOS R² |
|---|---|---|---|
| Pretrained FinBERT | .498 | .453 | — |
| Pretrained FLANG-RoBERTa | .492 | .457 | — |
| Frozen BERT (multimodal) | .485 | .529 | -.014 |
| **LoRA FinBERT** | **.506** | **.564** | — |

## Data

- **News**: ~20,550 financial articles (2017–2020)
- **Price**: 603,840 daily closing prices for 480 S&P 500 constituents (2017–2021)

## Tech Stack

Python, PyTorch, HuggingFace Transformers, PEFT (LoRA), statsmodels, scikit-learn

## Structure

- `stock_nlp_demo.ipynb` — full pipeline: data preprocessing, model training, evaluation, econometric analysis
- `run_lora.py` — LoRA fine-tuning script for FinBERT
- `haggett_dustin_mini-project_news_sentiment.pdf` — final paper

## Running

```bash
pip install torch transformers peft datasets scikit-learn statsmodels matplotlib
jupyter notebook stock_nlp_demo.ipynb
```

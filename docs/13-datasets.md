# Bundled and public datasets

## Bundled teaching data

| File | Used by | Meaning |
|---|---|---|
| `data/house_prices.csv` | Linear regression | Synthetic features and numeric labels |
| `data/customer_segments.csv` | K-means | Synthetic points without cluster labels |
| `data/lora_tiny_instructions.jsonl` | LoRA | Synthetic instruction-response pairs |
| No file | Q-learning | Experience generated through environment interaction |

Bundled datasets are small and synthetic. They make code easy to inspect and
tests fast; they are not suitable for real decisions or performance claims.

## Popular public datasets

### California Housing for regression

Scikit-learn can download a dataset with 20,640 samples and eight numeric
features:

```bash
uv run --with scikit-learn python - <<'PY'
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
frame = data.frame
frame.to_csv("data/california_housing.csv", index=False)
print(frame.shape)
print(frame.head())
PY
```

Official documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

### Iris for clustering or classification

Iris contains 150 flower measurements. It is small enough for quick K-means
experiments:

```bash
uv run --with scikit-learn python - <<'PY'
from sklearn.datasets import load_iris
data = load_iris(as_frame=True)
data.frame.to_csv("data/iris.csv", index=False)
print(data.frame.shape)
print(data.frame.head())
PY
```

Official documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html

The species labels are useful for evaluation, but do not give those labels to
K-means during fitting.

### Alpaca-style instructions for LoRA

The LoRA lab can stream a limited number of rows directly:

```bash
uv sync --extra lora
uv run ml-lab-lora \
  --hf-dataset yahma/alpaca-cleaned \
  --max-examples 200 \
  --steps 100
```

Dataset page: https://huggingface.co/datasets/yahma/alpaca-cleaned

## Data checklist

Before training on any public dataset, check:

- license and allowed uses;
- source and collection method;
- personal or sensitive information;
- duplicates and contamination;
- missing, malformed, or extreme values;
- class and demographic imbalance;
- whether the evaluation data accidentally overlaps training data;
- whether the dataset represents the environment where the model will run.

Record the exact dataset revision or checksum. A dataset with the same name can
change later.


# Linear regression

Supervised learning in its simplest useful form: predict a number from other
numbers. This lab predicts a house price from square footage, bedroom count,
and age, using gradient descent written in plain Python.

```bash
uv sync
uv run ml-lab-linear
```

```text
rows: 20 (train=16, test=4)
training MSE: 16205.03 -> 83.66
test RMSE: 21.33 thousand dollars
sample prediction for 1,800 sq ft, 3 bedrooms, age 12: $612k
```

Four lines, four things worth understanding. This guide walks through each.

## The model

The prediction is a weighted sum — that is all "linear" means:

```text
prediction = bias
           + weight_1 * square_feet
           + weight_2 * bedrooms
           + weight_3 * age_years
```

Four numbers (three weights and a bias) are the *entire* model. Training means
searching for the four values that make predictions closest to real prices.

The weights carry meaning. A positive weight on square footage says bigger
houses cost more; a negative weight on age says older houses cost less. The
bias is the baseline before any feature is considered.

## Measuring wrongness: MSE

For each house, subtract prediction from actual, square it, and average:

```text
house    actual   predicted   error   error²
A          485        470       15      225
B          535        560      -25      625
C          570        575       -5       25
                              mean squared error = 291.7
```

Squaring does two jobs. It makes every error positive, so overestimating by 10
and underestimating by 10 cannot cancel out. And it punishes large misses much
more than small ones — being wrong by 20 is four times worse than by 10, not
twice.

The cost is units. MSE is in *squared* thousand-dollars, which means nothing
intuitively. That is why the lab reports **RMSE** at the end — the square root
puts it back into thousands of dollars, so "21.33" is directly readable as
"typically off by about $21,000."

## Gradient descent

Picture the error as a landscape. Your position is the current set of weights;
your altitude is the error:

```text
error
  ^
  |  \                             /
  |   \                          /
  |    \                       /
  |     \_                  _/
  |       \__            __/
  |          \__      __/
  |             \____/  <- the weights we want
  +----------------------------------> weight value
        step size = learning rate
```

At any position you can compute which way is downhill — the **gradient** — take
a small step that way, and repeat. Each full pass over the data is one
**epoch**.

```text
1. Start the weights near zero.
2. Predict prices for the training rows.
3. Measure MSE.
4. Compute how much each parameter contributed to the error (the gradient).
5. Move each parameter a small distance downhill.
6. Repeat for many epochs.
```

The step size is the **learning rate**, and it is the setting most likely to
ruin your day.

## Watch the learning rate break things

This is the most instructive experiment in the lab. Same data, same epochs,
only the step size changes:

```bash
uv run ml-lab-linear --epochs 1500 --learning-rate 0.01
uv run ml-lab-linear --epochs 1500 --learning-rate 0.05    # the default
uv run ml-lab-linear --epochs 1500 --learning-rate 0.2
uv run ml-lab-linear --epochs 1500 --learning-rate 0.5
uv run ml-lab-linear --epochs 1500 --learning-rate 0.9
```

Real results from those runs:

| Learning rate | Final training MSE | Test RMSE | What happened |
|---|---|---|---|
| 0.01 | 105.19 | 34.52 | Too small — still descending when epochs ran out |
| 0.05 (default) | 83.66 | 21.33 | Converged |
| 0.1 | 83.65 | 21.29 | Converged |
| 0.2 | 83.65 | 21.29 | Converged; no further gain |
| 0.5 | a 222-digit number | ~10¹¹⁴ | **Diverged** |
| 0.9 | `nan` | `nan` | Diverged so hard the numbers stopped existing |

Look at the jump between 0.2 and 0.5. There is no gentle degradation — the
model goes from working to reporting a training error with **222 digits before
the decimal point**. At 0.9 it overflows into `nan` ("not a number").

```text
learning rate too small        just right           too large
     \                            \                    \      /
      \___                         \                    \    /   overshoots
          \___                      \                    \  /    the valley
              \___                   \_                   \/     and climbs
   still going when             lands in it            out the other side
   epochs ran out
```

That is what divergence looks like: each step overshoots the valley and lands
somewhere *worse*, so the next gradient is bigger, and the error explodes. If
you ever see `nan` in a training log, an overly large learning rate is the
first suspect.

## Why standardization matters

Look at the raw features:

```text
square_feet:  850 to 2700      <- range of ~1850
bedrooms:       1 to 5         <- range of 4
age_years:      3 to 35        <- range of 32
```

Square footage is hundreds of times larger than bedroom count. Without
rescaling, the same learning rate is simultaneously too large for one feature
and far too small for another, and the error landscape becomes a long narrow
valley that gradient descent zigzags down slowly.

The lab standardizes each feature first — subtract its mean, divide by its
spread — so all three occupy a comparable range and one learning rate suits
them all. This is why the working learning rates above are around 0.05 rather
than something like 0.0000001.

## Train and test

```text
20 rows total
├── 16 rows: training  -> the model sees these, learns from them
└──  4 rows: test      -> held back, used once at the end
```

Reporting error on data the model trained on is meaningless — it could simply
memorize. The 21.33 RMSE figure comes from the four rows it never saw, which is
the only number that says anything about *generalization*.

With just 16 training rows, expect the two to differ noticeably. That gap is
the lab showing you overfitting rather than hiding it.

## Epochs

```bash
uv run ml-lab-linear --epochs 50
uv run ml-lab-linear --epochs 1500
```

| Epochs | Final training MSE | Test RMSE |
|---|---|---|
| 50 | 315.20 | 64.86 |
| 1500 (default) | 83.66 | 21.33 |

At 50 epochs, training simply stopped early — the model was still descending.
This is *underfitting*, and it looks quite different from divergence: the error
is merely high, not exploding.

More epochs are not always better. Once the error stops falling, extra epochs
cost time and, on richer datasets, start fitting noise.

## Code to study

Open `training_labs/linear_regression.py` and find:

| What | Where |
|---|---|
| `load_csv` | Parses features and labels via `DictReader` (`:109`) |
| `_columns` | Transposes rows to columns for per-feature statistics |
| Standardization | Computes each feature's mean and spread |
| The prediction expression | The weighted sum — the forward pass |
| MSE calculation | Turns errors into one number |
| The gradient update | Where learning actually happens |
| Held-out evaluation | Scores the four unseen rows |

The whole file is standard-library Python — no NumPy, no PyTorch. Every step of
the algorithm is visible arithmetic.

Note `load_csv` reads columns **by name** (`square_feet`, `bedrooms`,
`age_years`, `price_thousands`), so pointing `--data` at a different CSV
requires either matching those names or editing the loader. See
[datasets](13-datasets.md#moving-to-real-data).

## About the data

`data/house_prices.csv` is synthetic and deliberately near-linear: price rises
with size and falls with age. It is small enough to read in full, and far too
small and too clean for any real property valuation.

```bash
head -5 data/house_prices.csv
```

## Experiments worth running

1. **Find your own divergence point.** Binary-search between 0.2 and 0.5. The
   boundary is sharp.
2. **Underfit deliberately.** Try `--epochs 5`, then `--epochs 20`.
3. **Predict with the mean as a baseline.** If the model barely beats "always
   guess the average price," it has learned little.
4. **Break standardization.** Comment it out and find a learning rate that
   still works. This is the fastest way to appreciate why it exists.

## Useful extensions

- Add location, lot size, or renovation features.
- Add an L2 penalty to discourage very large weights.
- Shuffle and use mini-batches instead of full-batch gradient descent.
- Plot training and validation loss by epoch — divergence becomes obvious.
- Compare against scikit-learn's `LinearRegression`, which solves this exactly
  rather than iteratively.
- Swap in California Housing, per [datasets](13-datasets.md).

## Next

- [K-means](10-kmeans.md) — learning with no labels at all.
- [Training overview](08-training-overview.md) — how the four strategies relate.

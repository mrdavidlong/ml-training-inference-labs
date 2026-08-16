# Linear regression

Linear regression is supervised learning for predicting a numeric value. The
sample predicts a house price from square footage, bedroom count, and age.

## Model

For three features, the prediction is:

```text
prediction = bias
           + weight_1 * square_feet
           + weight_2 * bedrooms
           + weight_3 * age_years
```

The lab standardizes each feature first. Without standardization, square
footage might be measured in thousands while bedroom count is only 1 to 5.
That scale difference makes gradient descent harder to tune.

## Training loop

1. Start the weights near zero.
2. Predict prices for the training rows.
3. Measure mean squared error (MSE).
4. Compute how each parameter contributed to the error.
5. Move each parameter a small distance in the direction that reduces loss.
6. Repeat for many epochs.

Run it:

```bash
uv sync
uv run ml-lab-linear
```

Try changing one variable:

```bash
uv run ml-lab-linear --epochs 50
uv run ml-lab-linear --epochs 1500 --learning-rate 0.01
uv run ml-lab-linear --epochs 1500 --learning-rate 0.2
```

Compare the initial and final training MSE and the test root mean squared error
(RMSE). A very large learning rate may bounce past a good answer or diverge. A
very small learning rate may require many epochs.

## Code to study

Open `training_labs/linear_regression.py` and locate:

- `load_csv`: parses features and labels;
- feature standardization: computes means and scales;
- the prediction expression: performs the forward pass;
- the MSE calculation: measures error;
- the gradient update: changes weights and bias;
- the held-out test calculation: checks unseen rows.

The bundled `data/house_prices.csv` is synthetic and educational. It is small
enough to inspect, but too small and simple for a real property valuation.

## Useful extensions

- Add location, lot size, or renovation features.
- Add an L2 penalty to discourage very large weights.
- Shuffle and use mini-batches instead of full-batch gradient descent.
- Plot training and validation loss by epoch.
- Compare against scikit-learn's `LinearRegression`.
- Replace the synthetic file with California Housing as described in
  [datasets](13-datasets.md).


"""Train linear regression with gradient descent and no ML framework.

THE IDEA

This is supervised learning in its simplest useful form. We have a table of
houses where each row lists some measurements (square feet, bedrooms, age) and
the price it sold for. We want a rule that turns measurements into a price
prediction for a house we have not seen.

"Linear" means the rule is just a weighted sum:

    price = bias + w1*square_feet + w2*bedrooms + w3*age

The three `w` values and the `bias` are the only things the program learns.
Training means searching for the four numbers that make predictions closest to
the real prices.

HOW THE SEARCH WORKS

We measure "closeness" with mean squared error (MSE): for every house, take
prediction minus actual, square it, and average over all houses. Squaring makes
every mistake positive (so overestimating by 10 and underestimating by 10 do not
cancel out) and punishes large misses much more than small ones.

Then we use gradient descent. Picture the error as a landscape where your
position is the current set of weights and altitude is the error:

     error
       ^
       |  \                    /
       |   \                  /
       |    \_             _/
       |      \__       __/
       |         \__ __/
       |            V           <- the weights we want
       +------------------------> weight value

At any position we can compute which way is downhill (the "gradient"), take a
small step that way, and repeat. Each full pass over the data is one "epoch".
The step size is the "learning rate": too small and it takes forever, too large
and you bounce over the valley and the error grows instead of shrinking.

Everything here is plain Python arithmetic -- no NumPy, no PyTorch -- so every
line of the algorithm is visible.

Further reading: https://developers.google.com/machine-learning/crash-course/linear-regression
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


# `parents[1]` climbs from training_labs/ to the repository root, so the data
# file is found no matter which directory the command runs from.
DEFAULT_DATA = Path(__file__).parents[1] / "data" / "house_prices.csv"


@dataclass
class LinearModel:
    """A trained linear regression model: the four learned numbers, plus the
    normalization constants needed to reuse it on new data.

    The `means` and `scales` are not part of the prediction rule itself. They
    are a record of how the training data was rescaled, and any new house must
    be rescaled exactly the same way or the weights will not apply to it.
    """

    # One weight per input feature, in the same column order as the CSV:
    # [square_feet, bedrooms, age_years]. A weight says how much the prediction
    # moves per one standard deviation of that feature. Because the features
    # were normalized, these are directly comparable to each other: a weight of
    # 40 matters twice as much as a weight of 20.
    weights: list[float]
    # The baseline prediction when every feature sits at its average value.
    # Measured in thousands of dollars, like the target column.
    bias: float
    # The average of each feature column in the training data, used to re-center
    # new inputs. Units match the raw CSV (square feet, count, years).
    means: list[float]
    # The standard deviation of each feature column, used to re-scale new
    # inputs. Never zero -- see the floor applied in `fit_linear_regression`.
    scales: list[float]

    def predict(self, features: list[float]) -> float:
        """Estimate the price of one house from its raw measurements.

        Args:
            features: Raw, un-normalized values in the training column order:
                [square_feet, bedrooms, age_years]. For example
                [1800.0, 3.0, 12.0].

        Returns:
            Predicted sale price in thousands of dollars.
        """
        # Apply the same shift-and-scale the training data received. Skipping
        # this would feed a raw 1800 into a weight tuned for values near 0.
        normalized = [
            (value - mean) / scale
            for value, mean, scale in zip(features, self.means, self.scales)
        ]
        # The weighted sum from the module docstring.
        return self.bias + sum(w * x for w, x in zip(self.weights, normalized))


def load_csv(path: Path) -> tuple[list[list[float]], list[float]]:
    """Read the house dataset and split it into inputs and answers.

    Args:
        path: A CSV file with the columns square_feet, bedrooms, age_years, and
            price_thousands.

    Returns:
        A pair `(features, targets)` where:
        - features is one list per house, holding its three measurements in a
          fixed order. In ML terms these are the model's inputs.
        - targets is the matching list of sale prices. These are the "labels":
          the known correct answers that make this *supervised* learning.

        The two lists are aligned by position: features[i] describes the house
        that sold for targets[i].
    """
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # Column order is fixed here and relied on everywhere else, including
    # LinearModel.predict and the sample prediction in main().
    features = [
        [float(row["square_feet"]), float(row["bedrooms"]), float(row["age_years"])]
        for row in rows
    ]
    targets = [float(row["price_thousands"]) for row in rows]
    return features, targets


def _columns(rows: list[list[float]]) -> list[list[float]]:
    """Flip a table from row-major to column-major layout.

    Statistics like "the average square footage" are computed down a column,
    but the data arrives as one list per house. `zip(*rows)` transposes it:

        rows:                    columns:
        [[1800, 3, 12],          [[1800, 2100],
         [2100, 4,  5]]    ->     [   3,    4],
                                  [  12,    5]]

    Args:
        rows: One inner list per house.

    Returns:
        One inner list per feature.
    """
    return [list(column) for column in zip(*rows)]


def fit_linear_regression(
    features: list[list[float]],
    targets: list[float],
    *,
    learning_rate: float = 0.05,
    epochs: int = 1_500,
) -> tuple[LinearModel, list[float]]:
    """Learn weights and a bias by repeatedly nudging them downhill.

    Args:
        features: One list of raw measurements per house.
        targets: The true price for each house, same order and length.
        learning_rate: How big a step to take each epoch. Values around 0.01 to
            0.1 work for this normalized data. If the reported loss grows
            instead of shrinking, this is too large.
        epochs: How many full passes to make over the data. More epochs cost
            more time and eventually stop helping, since the loss flattens out.

    Returns:
        A pair `(model, losses)`:
        - model: the trained LinearModel, carrying the normalization constants.
        - losses: the MSE after each epoch, oldest first. Plotting or just
          comparing losses[0] to losses[-1] shows whether learning happened.

    Raises:
        ValueError: If there are no rows, or if the number of feature rows does
            not match the number of targets.
    """
    if not features or len(features) != len(targets):
        raise ValueError("features and targets must contain the same nonzero number of rows")

    # ---- Normalization ---------------------------------------------------
    # Square footage runs in the thousands while bedrooms run 1-5. Left raw,
    # one shared learning rate cannot suit both: a step size sensible for
    # square feet would barely move the bedroom weight, and a step sensible for
    # bedrooms would make the square-feet weight explode. Rescaling every
    # column to roughly mean 0 and spread 1 puts them on equal footing.
    columns = _columns(features)
    means = [sum(column) / len(column) for column in columns]
    scales = [
        # Standard deviation: the typical distance from the average.
        # `max(..., 1e-12)` guards against a column where every value is
        # identical, which would give a spread of 0 and then divide by zero.
        max((sum((value - mean) ** 2 for value in column) / len(column)) ** 0.5, 1e-12)
        for column, mean in zip(columns, means)
    ]
    normalized = [
        [(value - mean) / scale for value, mean, scale in zip(row, means, scales)]
        for row in features
    ]

    # ---- Starting point --------------------------------------------------
    # All weights start at zero: no feature is assumed to matter yet. The bias
    # starts at the average price, so the very first guess is "every house
    # costs the average" -- a reasonable baseline that gradient descent then
    # improves on. Starting the bias at 0 instead would work but waste epochs
    # climbing from nothing up to the right ballpark.
    weights = [0.0] * len(features[0])
    bias = sum(targets) / len(targets)
    losses: list[float] = []
    count = len(targets)

    for _ in range(epochs):
        # 1. Predict every house with the current numbers.
        predictions = [bias + sum(w * x for w, x in zip(weights, row)) for row in normalized]
        # 2. Measure the miss on each. Positive means we guessed too high.
        errors = [prediction - target for prediction, target in zip(predictions, targets)]
        # 3. Record the mean squared error so the caller can see the trend.
        losses.append(sum(error * error for error in errors) / count)

        # 4. Step downhill. These two updates are the calculus derivative of
        #    the MSE, which works out to a simple form: the gradient for the
        #    bias is the average error, and the gradient for a weight is the
        #    average error weighted by that feature's value. The 2 comes from
        #    differentiating the square. Intuitively: if we tend to overshoot
        #    (positive errors) on houses where a feature is large, that
        #    feature's weight is too big and gets reduced.
        #
        #    Note every weight is updated from the SAME `errors` list computed
        #    at the top of the epoch, not recomputed in between. That is what
        #    makes this "batch" gradient descent -- one coordinated step using
        #    the whole dataset, rather than a scramble of separate adjustments.
        bias -= learning_rate * (2.0 / count) * sum(errors)
        for index in range(len(weights)):
            gradient = (2.0 / count) * sum(
                error * row[index] for error, row in zip(errors, normalized)
            )
            weights[index] -= learning_rate * gradient

    return LinearModel(weights, bias, means, scales), losses


def main() -> None:
    """Train on most of the data, score the rest, and print a sample prediction.

    Nothing is returned; results go to standard output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=1_500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    features, targets = load_csv(args.data)

    # Hold back the last 20% of rows. The model never sees them during
    # training, so scoring on them answers the question that actually matters:
    # does this work on houses it has not memorized? A model can score
    # perfectly on its training data and still be useless on new data --
    # that failure is called overfitting.
    #
    # A real project would shuffle before splitting, in case the CSV is sorted
    # by something meaningful. This lab keeps the split fixed so that repeated
    # runs give identical, comparable numbers.
    split = max(1, int(len(features) * 0.8))
    model, losses = fit_linear_regression(
        features[:split], targets[:split],
        learning_rate=args.learning_rate, epochs=args.epochs,
    )
    test_predictions = [model.predict(row) for row in features[split:]]
    test_targets = targets[split:]
    test_mse = sum((p - y) ** 2 for p, y in zip(test_predictions, test_targets)) / len(test_targets)

    print(f"rows: {len(features)} (train={split}, test={len(features) - split})")
    # First epoch versus last epoch. A large drop means the model learned; a
    # flat line means the learning rate or epoch count needs attention.
    print(f"training MSE: {losses[0]:.2f} -> {losses[-1]:.2f}")
    # RMSE is the square root of MSE, which undoes the squaring and puts the
    # error back into the target's own units. "Typically off by this much in
    # thousands of dollars" is far easier to judge than a squared quantity.
    print(f"test RMSE: {test_mse ** 0.5:.2f} thousand dollars")
    sample = [1_800.0, 3.0, 12.0]
    print(f"sample prediction for 1,800 sq ft, 3 bedrooms, age 12: ${model.predict(sample):.0f}k")


if __name__ == "__main__":
    main()

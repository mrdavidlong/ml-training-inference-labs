"""Cluster two-dimensional points with K-means and no ML framework.

THE IDEA

Linear regression had labels: every house came with its true price. Here there
are no labels at all. We have customers described by how much they spend per
year and how often they visit, and we want to know whether they naturally fall
into groups -- without anyone telling us what the groups are. That is
*unsupervised* learning.

K-means finds exactly `k` groups. You choose k; the algorithm finds where the
groups sit. Each group is represented by its "centroid", the average position
of its members.

THE ALGORITHM (Lloyd's algorithm)

Two steps, repeated until nothing changes:

    1. ASSIGN each point to whichever centroid is nearest.
    2. MOVE  each centroid to the average position of the points assigned to it.

    Start (centroids X placed):     After assign:        After move:

      .  .        .  .                a  a       b  b       a  a      b  b
        X       .   .                  a  X    b   b         aXa    b  b
      .    .   .    X                 a    a   b    X       a   a   b  X b
         .  .    .                       a  a    b            a  a   b

Moving the centroids can put different points nearer a different centroid, so
step 1 is run again, and so on. This always settles down, usually within a few
rounds.

A caveat worth knowing: K-means finds *a* good answer, not necessarily the best
one. Where the centroids start affects where they end up, which is why the
starting positions are chosen carefully (see `fit_kmeans`) and why the run is
seeded so results are reproducible.

Everything here is plain Python -- no NumPy -- so the algorithm is fully visible.

Further reading: https://developers.google.com/machine-learning/clustering/overview
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


# `parents[1]` climbs from training_labs/ to the repository root.
DEFAULT_DATA = Path(__file__).parents[1] / "data" / "customer_segments.csv"


def load_points(path: Path) -> list[tuple[float, float]]:
    """Read the customer dataset as a list of two-dimensional points.

    Args:
        path: A CSV file with the columns annual_spend and visits_per_month.

    Returns:
        One (annual_spend, visits_per_month) tuple per customer. Note there is
        no separate "answer" column: unsupervised learning has nothing to
        compare against, which is exactly what makes it a different problem.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            (float(row["annual_spend"]), float(row["visits_per_month"]))
            for row in csv.DictReader(handle)
        ]


def squared_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Measure how far apart two points are, without taking the square root.

    Ordinary straight-line distance is sqrt((x1-x2)^2 + (y1-y2)^2). This
    function omits the sqrt. That is safe wherever we only *compare* distances,
    because squaring preserves order: if A is nearer than B, its squared
    distance is also smaller. Skipping the square root avoids a lot of needless
    arithmetic in the innermost loop.

    Args:
        left: A point, as a tuple of coordinates.
        right: Another point with the same number of coordinates.

    Returns:
        The sum of squared differences per coordinate. Not in the same units as
        the input -- take `math.sqrt` if you need a real distance, as
        `silhouette_hint` does.
    """
    return sum((a - b) ** 2 for a, b in zip(left, right))


def fit_kmeans(
    points: list[tuple[float, ...]],
    k: int,
    *,
    max_iterations: int = 100,
    seed: int = 7,
) -> tuple[list[tuple[float, ...]], list[int], float]:
    """Group points into k clusters and report how tight the result is.

    Args:
        points: The data to cluster. Any consistent number of dimensions works,
            though this lab uses two so results can be reasoned about by hand.
        k: How many clusters to find. Chosen by you, not discovered. Try
            several values and compare the silhouette score.
        max_iterations: A safety cap on the assign/move cycle. The loop
            normally exits far earlier, as soon as assignments stop changing.
        seed: Fixes the random number generator so repeated runs give identical
            output. Change it to see how much the starting positions matter.

    Returns:
        A triple `(centroids, assignments, inertia)`:
        - centroids: the final center of each cluster, indexed 0..k-1.
        - assignments: one cluster index per input point, aligned by position,
          so assignments[i] says which cluster points[i] belongs to.
        - inertia: total squared distance from every point to its own centroid.
          Lower means tighter clusters. Useful only for comparing runs at the
          SAME k -- inertia always falls as k rises, hitting zero when every
          point is its own cluster, so it cannot be used to choose k.

    Raises:
        ValueError: If k is below 1 or exceeds the number of points.
    """
    if not 1 <= k <= len(points):
        raise ValueError("k must be between 1 and the number of points")

    rng = random.Random(seed)

    # ---- Choosing starting centroids (the "k-means++" method) ------------
    # Picking all k starting points purely at random often puts two of them in
    # the same dense blob, which leaves a real cluster unrepresented and gives
    # a poor final answer. k-means++ instead spreads them out: after the first
    # is chosen at random, each additional one is drawn with probability
    # proportional to its squared distance from the nearest centroid already
    # chosen. Far-away points are therefore much more likely to be picked.
    #
    # The selection below is a weighted random draw. Imagine laying every
    # point's distance end to end as a segment of a line, then dropping a pin
    # at a random spot -- larger segments catch the pin more often:
    #
    #    |--A--|-B-|--------C--------|--D--|
    #                    ^ pin lands here -> C is chosen
    centroids = [rng.choice(points)]
    while len(centroids) < k:
        distances = [min(squared_distance(point, center) for center in centroids) for point in points]
        total = sum(distances)
        threshold = rng.random() * total
        running = 0.0
        # Fallback for the edge case where floating-point rounding leaves the
        # running total just under the threshold and the loop never breaks.
        selected = points[-1]
        for point, distance in zip(points, distances):
            running += distance
            if running >= threshold:
                selected = point
                break
        centroids.append(selected)

    # ---- The assign / move cycle ----------------------------------------
    # -1 is a sentinel meaning "not yet assigned". No real cluster index is
    # negative, so the first comparison below is guaranteed not to match.
    assignments = [-1] * len(points)
    for _ in range(max_iterations):
        # ASSIGN: each point joins its nearest centroid. `min` over the range
        # of cluster indices returns the index whose centroid is closest.
        new_assignments = [
            min(range(k), key=lambda index: squared_distance(point, centroids[index]))
            for point in points
        ]
        # If nobody switched clusters, the centroids will not move either, so
        # further rounds would change nothing. This is the natural stopping
        # point, usually reached well before max_iterations.
        if new_assignments == assignments:
            break
        assignments = new_assignments

        # MOVE: each centroid becomes the average of its members. `zip(*members)`
        # transposes the member list so each coordinate can be averaged
        # separately -- all the x values, then all the y values.
        for cluster in range(k):
            members = [point for point, label in zip(points, assignments) if label == cluster]
            # A cluster can end up empty when no point is nearest to it. Its
            # centroid is then left where it was rather than averaging an empty
            # list, which would divide by zero.
            if members:
                centroids[cluster] = tuple(
                    sum(values) / len(values) for values in zip(*members)
                )

    inertia = sum(
        squared_distance(point, centroids[label])
        for point, label in zip(points, assignments)
    )
    return centroids, assignments, inertia


def silhouette_hint(points: list[tuple[float, ...]], assignments: list[int]) -> float:
    """Return a simple average silhouette score for teaching, not large datasets.

    Inertia cannot tell you whether k was a sensible choice, because it always
    improves as k grows. The silhouette score can, because it asks a question
    that has a natural best answer: is each point closer to its own group than
    to the nearest other group?

    For one point:
        a = its average distance to the other members of its own cluster
        b = its average distance to the members of the nearest other cluster
        score = (b - a) / max(a, b)

    Reading the result:
         1.0  the point sits snugly inside its cluster, far from any other
         0.0  the point is on the boundary; it could belong to either
        -1.0  the point is closer to another cluster than its own -- misplaced

    The function averages that score over every point. Run it for k=2, 3, 4 and
    prefer the k that scores highest.

    Args:
        points: The clustered data.
        assignments: One cluster index per point, as returned by `fit_kmeans`.

    Returns:
        The mean silhouette score, between -1 and 1. Higher is better.

    Note:
        This compares every point against every other point, so the work grows
        with the square of the dataset size. Fine for the few dozen rows in this
        lab; use scikit-learn's implementation for anything real.
        https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
    """
    scores: list[float] = []
    labels = sorted(set(assignments))
    for index, point in enumerate(points):
        # Members of this point's own cluster, excluding the point itself --
        # its distance to itself is zero and would drag the average down.
        same = [p for i, (p, label) in enumerate(zip(points, assignments)) if label == assignments[index] and i != index]
        # A lone point in its own cluster has no in-cluster neighbors; treat
        # its internal distance as 0 rather than dividing by an empty list.
        a = sum(math.sqrt(squared_distance(point, p)) for p in same) / len(same) if same else 0.0

        # Average distance to each *other* cluster, so the nearest can be found.
        other_means = []
        for label in labels:
            if label == assignments[index]:
                continue
            members = [p for p, member_label in zip(points, assignments) if member_label == label]
            other_means.append(sum(math.sqrt(squared_distance(point, p)) for p in members) / len(members))
        b = min(other_means) if other_means else 0.0

        # The max(a, b) divisor scales the result into -1..1. The final guard
        # covers k=1, where there is no other cluster and both a and b are 0.
        scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return sum(scores) / len(scores)


def main() -> None:
    """Cluster the bundled customer data and print the resulting groups.

    Nothing is returned; results go to standard output. Try `--clusters 2`,
    `3`, and `4` and compare the silhouette scores to see which k fits best.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    points = load_points(args.data)
    centroids, assignments, inertia = fit_kmeans(points, args.clusters, seed=args.seed)
    print(f"points: {len(points)}, clusters: {args.clusters}")
    for index, centroid in enumerate(centroids):
        # The centroid coordinates are in the original units, so a center of
        # (4200, 8.5) reads as "spends about $4,200 a year, visits ~8.5 times
        # a month". Naming the segments is a human judgment the algorithm does
        # not make for you.
        count = assignments.count(index)
        print(f"cluster {index}: center={tuple(round(v, 2) for v in centroid)}, members={count}")
    print(f"inertia (lower is tighter): {inertia:.2f}")
    print(f"silhouette score (-1 to 1; higher is better): {silhouette_hint(points, assignments):.3f}")


if __name__ == "__main__":
    main()

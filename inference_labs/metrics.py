"""Turn a pile of raw request timings into an honest performance summary.

WHY AVERAGES LIE

The obvious way to report speed is the average response time. It is also the
most misleading. Consider ten requests, nine taking 100ms and one taking 5
seconds. The average is 590ms -- a number that describes none of them, and that
hides the fact that one user in ten had an awful experience.

Percentiles describe the shape instead. "p95 = 800ms" means 95% of requests
finished within 800ms and the slowest 5% took longer. Sorting the timings and
reading off positions gives the whole picture:

    fastest                                             slowest
    |------------------------------|--------|-----|-------|
                                  p50      p95   p99
                                (median)

The tail is what users actually complain about, and it is exactly what an
average conceals. A change that improves the mean while worsening p99 has
usually made the service worse.

Percentile definitions vary slightly between tools; this module interpolates
between neighbors, which is the same convention NumPy uses by default. Small
differences at p99 on a few dozen samples are normal and not worth chasing.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean


def percentile(values: list[float], p: float) -> float:
    """Return a linearly interpolated percentile for p in [0, 100].

    Sorts the values and reads off the position p% of the way along. When that
    position falls between two samples, the result is blended between them
    rather than rounded to one.

    Worked example with four values [10, 20, 30, 40] and p=95:
        position = (4 - 1) * 95 / 100 = 2.85
        lower = 2 (value 30), upper = 3 (value 40), fraction = 0.85
        result = 30 * 0.15 + 40 * 0.85 = 38.5

    Args:
        values: The measurements. Order does not matter; the function sorts a
            copy and leaves the caller's list untouched.
        p: Which percentile to compute, 0 to 100. 50 gives the median.

    Returns:
        The value at that percentile, in the same units as the input.

    Raises:
        ValueError: If `values` is empty, or `p` is outside 0..100.
    """
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    ordered = sorted(values)
    # `len - 1` because positions are counted from 0, so p=100 must land on the
    # last index rather than one past the end.
    position = (len(ordered) - 1) * p / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    # The position landed exactly on a sample, so there is nothing to blend.
    # This also covers a single-element list, where both are 0.
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    # Weighted blend: the nearer neighbor contributes more.
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class Summary:
    """The complete result of one load test.

    Times are stored in milliseconds because that is the readable unit for a
    request, even though the inputs arrive in seconds.
    """

    # How many requests were attempted in total.
    requests: int
    # How many came back without an error.
    succeeded: int
    # How many failed: a connection problem, a timeout, or an HTTP error.
    failed: int
    # succeeded / requests, from 0.0 to 1.0. Read this before any timing --
    # a server that is fast because it rejected half the traffic is not fast.
    success_rate: float
    # Total elapsed time for the whole test, start to finish, in seconds.
    # Real clock time, so it includes waiting caused by the concurrency limit.
    wall_seconds: float
    # Completed requests per second. The throughput number: how much total work
    # the server got through, as opposed to how fast any one request felt.
    requests_per_second: float
    # Mean end-to-end time per successful request. Included for familiarity,
    # but prefer the percentiles below -- see this module's docstring.
    mean_latency_ms: float
    # Median: half of requests were faster than this. The "typical" experience.
    p50_latency_ms: float
    # 95th percentile: only 1 request in 20 was slower. The usual target for
    # a service-level objective.
    p95_latency_ms: float
    # 99th percentile: the near-worst case. Needs a few hundred requests before
    # it means much -- with 40 requests it is essentially the slowest one.
    p99_latency_ms: float
    # Mean time to first token: how long until the reply *starts* arriving.
    # None when the test did not use streaming, since there are no tokens to
    # time in a non-streamed response. This is what a user perceives as
    # responsiveness -- text appearing quickly matters more than the total.
    mean_ttft_ms: float | None
    # 95th percentile time to first token. None for the same reason.
    p95_ttft_ms: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        """Convert to a plain dictionary for JSON output.

        Returns:
            One key per field above, with the same values.
        """
        return asdict(self)


def summarize(
    latencies_seconds: list[float],
    failures: int,
    wall_seconds: float,
    ttft_seconds: list[float] | None = None,
) -> Summary:
    """Compute the full statistics for one load test.

    Args:
        latencies_seconds: End-to-end duration of each *successful* request.
            Failures are excluded, since a request that died after 50ms is not
            evidence of a fast server.
        failures: How many requests failed. Counted rather than timed.
        wall_seconds: Total elapsed time for the run, used for throughput.
        ttft_seconds: Time to first token for each streamed request, if the test
            streamed. Its length can be shorter than `latencies_seconds`, since
            a request can succeed without a first-token time being captured.

    Returns:
        A populated Summary. Times are converted from seconds to milliseconds.

    Raises:
        ValueError: If no requests were made at all.
    """
    succeeded = len(latencies_seconds)
    requests = succeeded + failures
    if requests == 0:
        raise ValueError("at least one request is required")
    if succeeded == 0:
        # Every request failed, so there are no timings to summarize, and
        # `percentile` would raise on an empty list. Substituting a single zero
        # lets the Summary be built and reported. The timing fields are then
        # meaningless -- success_rate will be 0.0, which is the field that
        # matters in this case.
        zeros = [0.0]
        latencies_seconds = zeros
    ttft = ttft_seconds or []
    return Summary(
        requests=requests,
        succeeded=succeeded,
        failed=failures,
        success_rate=succeeded / requests,
        wall_seconds=wall_seconds,
        # Guard against a zero-duration run, which would divide by zero.
        requests_per_second=succeeded / wall_seconds if wall_seconds else 0.0,
        # 1000 converts seconds to milliseconds throughout.
        mean_latency_ms=mean(latencies_seconds) * 1000,
        p50_latency_ms=percentile(latencies_seconds, 50) * 1000,
        p95_latency_ms=percentile(latencies_seconds, 95) * 1000,
        p99_latency_ms=percentile(latencies_seconds, 99) * 1000,
        # None rather than 0 when nothing was measured, so a reader can tell
        # "not streamed" apart from "instant".
        mean_ttft_ms=mean(ttft) * 1000 if ttft else None,
        p95_ttft_ms=percentile(ttft, 95) * 1000 if ttft else None,
    )

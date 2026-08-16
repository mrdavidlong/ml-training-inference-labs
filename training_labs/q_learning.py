"""Learn a GridWorld policy using tabular Q-learning.

THE IDEA

The two earlier labs both learned from a fixed dataset. Reinforcement learning
has no dataset. Instead an *agent* acts in an *environment*, receives a reward
or penalty, and has to work out for itself which actions were worth taking.

Nobody ever tells the agent the correct move. It only ever learns that some
sequence of moves eventually paid off, and has to figure out which of those
moves deserve the credit. That is the hard part, and it is what the update rule
below solves.

THE ENVIRONMENT

A 5x5 grid. The agent starts top-left and wants to reach the goal bottom-right.
Black squares are walls it cannot enter.

        col 0   1   2   3   4
    row 0  S   .   .   .   .        S = start (0, 0)
    row 1  .   #   #   .   .        G = goal  (4, 4)
    row 2  .   .   .   .   .        # = obstacle
    row 3  .   #   .   #   .
    row 4  .   .   .   .   G

Reaching the goal earns +10 and ends the episode. Every other move costs -0.1.
That small penalty is what makes the agent prefer short routes: dawdling is not
forbidden, just mildly expensive, so the shortest path collects the least
penalty on the way to the prize.

THE Q-TABLE

"Q" stands for quality. The agent keeps a table with one row per square and one
column per action, holding its current estimate of "if I take this action here
and play sensibly afterwards, how much total reward do I end up with?"

                  up     right   down    left
    (0, 0)      -0.35    4.21    4.98   -0.35
    (0, 1)      -0.30    4.87    2.10    3.90
    ...

Once the table is filled in, behaving well is trivial: in any square, take the
action with the highest number. The table IS the learned policy. "Tabular"
means literally storing every square, which only works because this world is
tiny -- real problems replace the table with a neural network.

Further reading: http://incompleteideas.net/book/the-book-2nd.html (Sutton &
Barto, chapter 6 covers Q-learning)
"""

from __future__ import annotations

import argparse
import random


# Action names, in a fixed order. The index into this tuple is also the index
# into each row of the Q-table, so "right" is always column 1.
ACTIONS = ("up", "right", "down", "left")
# How each action changes (row, column). Row 0 is the top, so moving "up"
# subtracts 1 from the row -- the same convention as screen coordinates.
MOVES = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}


class GridWorld:
    """The maze the agent moves through: its layout and its rules.

    This is the "environment" half of reinforcement learning. It knows nothing
    about learning -- it only answers the question "if I am here and do that,
    what happens?" Keeping it separate from the agent is the standard split,
    and it is what lets the same learning code work on a different world.
    """

    # Grid is `size` x `size`, with coordinates running 0..size-1.
    size = 5
    # Where every episode begins, as (row, column).
    start = (0, 0)
    # The square that ends an episode and pays the reward.
    goal = (4, 4)
    # Squares that cannot be entered. Stored as a set because the only question
    # ever asked is "is this square blocked?", which a set answers instantly
    # however many walls there are.
    obstacles = {(1, 1), (1, 2), (3, 1), (3, 3)}

    def step(self, state: tuple[int, int], action: str) -> tuple[tuple[int, int], float, bool]:
        """Apply one action and report what happened.

        Args:
            state: Where the agent is now, as (row, column).
            action: One of "up", "right", "down", "left".

        Returns:
            A triple `(next_state, reward, done)`:
            - next_state: where the agent ended up. Equal to `state` if the move
              was blocked.
            - reward: +10.0 for reaching the goal, otherwise -0.1.
            - done: True if the episode is over, i.e. the goal was reached.
        """
        row_delta, column_delta = MOVES[action]
        candidate = (state[0] + row_delta, state[1] + column_delta)
        # An illegal move -- off the edge or into a wall -- is not an error. The
        # agent simply stays put and still pays the -0.1. It has to *learn* that
        # walking into walls is a waste; the environment does not forbid it.
        if (
            candidate[0] not in range(self.size)
            or candidate[1] not in range(self.size)
            or candidate in self.obstacles
        ):
            candidate = state
        if candidate == self.goal:
            return candidate, 10.0, True
        # The small per-move cost. Without it, a route that wanders for 50 steps
        # would score the same as the direct one, and the agent would have no
        # reason to prefer either.
        return candidate, -0.1, False


def train_q_learning(
    *,
    episodes: int = 2_000,
    learning_rate: float = 0.2,
    discount: float = 0.95,
    epsilon: float = 0.8,
    seed: int = 7,
) -> dict[tuple[int, int], list[float]]:
    """Fill in the Q-table by repeatedly attempting the maze.

    Args:
        episodes: How many attempts to make. Each starts fresh at the start
            square and runs until the goal is reached or 100 moves elapse.
        learning_rate: How much each new experience overwrites the old estimate,
            from 0 (learn nothing) to 1 (believe only the latest result). 0.2
            blends the new observation gently into what is already known.
        discount: How much a future reward is worth compared to an immediate
            one, between 0 and 1. At 0.95, a reward one step away counts for
            95% of its value, two steps away 90%, and so on. Near 0 the agent
            becomes short-sighted and cannot learn multi-step routes; at
            exactly 1 there is nothing pushing it to arrive sooner.
        epsilon: The starting probability of ignoring the table and moving at
            random. See the exploration note in the loop below.
        seed: Fixes the random number generator for reproducible runs.

    Returns:
        The Q-table: a dictionary whose key is a (row, column) square and whose
        value is a list of four estimated returns, ordered to match `ACTIONS`.
        Obstacle squares are absent from the table entirely, since the agent
        can never stand on one.
    """
    environment = GridWorld()
    rng = random.Random(seed)

    # Every reachable square starts with all four actions valued at 0.0. The
    # agent begins with no opinion whatsoever about what is worth doing.
    q_values = {
        (row, column): [0.0] * len(ACTIONS)
        for row in range(environment.size)
        for column in range(environment.size)
        if (row, column) not in environment.obstacles
    }

    for episode in range(episodes):
        state = environment.start

        # ---- The explore / exploit balance -------------------------------
        # An agent that always follows its current best guess will keep
        # repeating the first route that happened to work and never discover a
        # shorter one. An agent that always moves randomly never puts what it
        # learned to use. So: act randomly with probability epsilon, otherwise
        # follow the table.
        #
        # Epsilon decays from 0.8 toward 0.02 across the run -- explore widely
        # early on when the table is worthless, then increasingly trust it:
        #
        #   epsilon
        #     0.8 |*
        #         | ****
        #         |     *****
        #         |          ********
        #    0.02 |__________________********
        #         0                    episodes ->
        #
        # The 0.02 floor keeps a sliver of exploration alive to the end.
        current_epsilon = max(0.02, epsilon * (1.0 - episode / episodes))

        # Cap each attempt at 100 moves so an episode that never finds the goal
        # (very likely early on, while moves are mostly random) still ends.
        for _ in range(100):
            if rng.random() < current_epsilon:
                action_index = rng.randrange(len(ACTIONS))
            else:
                # Take the highest-valued action for this square. `__getitem__`
                # is used as the key function, so `max` compares the four
                # stored values and returns the index of the best.
                action_index = max(range(len(ACTIONS)), key=q_values[state].__getitem__)

            next_state, reward, done = environment.step(state, ACTIONS[action_index])

            # ---- The Q-learning update -----------------------------------
            # The best achievable value from wherever we landed. At the goal
            # there is no "afterwards", so the future is worth 0.
            future = 0.0 if done else max(q_values[next_state])
            old = q_values[state][action_index]
            #
            #   new = old + learning_rate * (reward + discount * future - old)
            #                                \_______________________/   \_/
            #                                   a better estimate      current
            #                                \___________________________/
            #                                  how wrong we were
            #
            # Read it as: "here is what I now think this move was worth; move my
            # old estimate part of the way toward it." The bracketed term is the
            # error, and learning_rate decides what fraction of it to act on.
            #
            # This is how credit reaches moves far from the goal. Only the final
            # step ever sees the +10 directly. But once the square before the
            # goal has a high value, the square before THAT picks it up through
            # `future` on a later episode, and the value seeps backward through
            # the grid one square per pass:
            #
            #   episode ~1:    . . . . .        after many episodes:
            #                  . . . . .           values increase as
            #                  . . . . .           you approach the goal
            #                  . . . .[10]         . . . 6 8
            #                                      . . 6 8 [10]
            q_values[state][action_index] = old + learning_rate * (
                reward + discount * future - old
            )
            state = next_state
            if done:
                break
    return q_values


def greedy_path(q_values: dict[tuple[int, int], list[float]]) -> list[tuple[int, int]]:
    """Walk the maze following the learned table, with no randomness at all.

    This is the exam. Training used exploration; here the agent always takes the
    action it rates highest, which shows what it actually learned.

    Args:
        q_values: A trained Q-table from `train_q_learning`.

    Returns:
        The squares visited, starting at the start and ending at the goal.
        `len(path) - 1` is the number of moves taken.

    Raises:
        RuntimeError: If the goal is not reached within 100 moves. That means
            training did not converge -- typically too few episodes, or a
            discount so low the agent cannot see far enough ahead. Following a
            bad table tends to produce a loop between two squares that each
            point at the other.
    """
    environment = GridWorld()
    state = environment.start
    path = [state]
    for _ in range(100):
        action_index = max(range(len(ACTIONS)), key=q_values[state].__getitem__)
        state, _, done = environment.step(state, ACTIONS[action_index])
        path.append(state)
        if done:
            return path
    raise RuntimeError("the learned policy did not reach the goal")


def render_policy(q_values: dict[tuple[int, int], list[float]]) -> str:
    """Draw the learned policy as a grid of arrows.

    Each cell shows the single action the agent would choose there, which makes
    the whole strategy readable at a glance -- arrows should generally flow
    toward the bottom-right and route around the walls.

    Args:
        q_values: A trained Q-table from `train_q_learning`.

    Returns:
        A multi-line string, one text row per grid row. The default 2,000-episode
        run produces::

            → → → → ↓
            ↓ ■ ■ → ↓
            → → → → ↓
            ↑ ■ ↓ ■ ↓
            ↑ → → → G

        where ■ marks an obstacle and G the goal. Following the arrows from the
        top-left traces the route along the top edge and down the right side.

        Arrows on squares the agent rarely visits may point in odd directions --
        the ↑ in the bottom-left corner here would walk into a dead end. Those
        cells sit off the discovered route, so they received few updates and
        their estimates stayed near their initial zeros. A policy is only
        trustworthy on the states it actually practiced.
    """
    arrows = {"up": "↑", "right": "→", "down": "↓", "left": "←"}
    environment = GridWorld()
    rows = []
    for row in range(environment.size):
        cells = []
        for column in range(environment.size):
            state = (row, column)
            if state == environment.goal:
                cells.append("G")
            elif state in environment.obstacles:
                cells.append("■")
            else:
                best = max(range(len(ACTIONS)), key=q_values[state].__getitem__)
                cells.append(arrows[ACTIONS[best]])
        rows.append(" ".join(cells))
    return "\n".join(rows)


def main() -> None:
    """Train an agent, print its learned policy, and show the route it takes.

    Nothing is returned; results go to standard output. Try lowering
    `--episodes` to see a policy that has not finished learning.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    q_values = train_q_learning(episodes=args.episodes, seed=args.seed)
    path = greedy_path(q_values)
    print(render_policy(q_values))
    # The shortest legal route through this grid is 8 moves, so a path much
    # longer than that means training has not fully converged.
    print(f"greedy path ({len(path) - 1} steps): {path}")


if __name__ == "__main__":
    main()

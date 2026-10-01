"""The game API: `Game`, its `State`, and the `TimeStep` players observe."""

import abc
from collections.abc import Callable
import dataclasses
import functools
from typing import dataclass_transform

import jax
import jax.numpy as jnp
import numpy as np

from nashbench import sequence_form as sequence_form_lib


@dataclass_transform(frozen_default=True)
def pytree_dataclass[T](cls: type[T]) -> type[T]:
    """Makes `cls` a frozen dataclass that JAX transformations accept."""
    return jax.tree_util.register_dataclass(
        dataclasses.dataclass(frozen=True)(cls)
    )


@pytree_dataclass
class TimeStep:
    """What the players see after `Game.reset` or `Game.step`.

    `observation` and `legal_action_mask` belong to the player to act and are
    defined only while `done` is false.

    Attributes:
        observation: `[*observation_shape]` float32 information state.
        legal_action_mask: `[num_actions]` bool.
        reward: `[2]` float32 reward of each player for the last step.
        done: `[]` bool, true once the episode has ended.
        current_player: `[]` int32 player to act: 0 or 1.
    """

    observation: jax.Array
    legal_action_mask: jax.Array
    reward: jax.Array
    done: jax.Array
    current_player: jax.Array


@pytree_dataclass
class GameState:
    """Base class for the state of a game's rules, indexed by seat.

    Attributes:
        current_seat: `[]` int32 seat to act: 0 or 1.
        done: `[]` bool, true once the game has ended.
    """

    current_seat: jax.Array
    done: jax.Array


@pytree_dataclass
class State[S: GameState]:
    """The full state of an episode, including what players cannot see.

    Attributes:
        key: PRNG key of the next episode, used by `auto_reset`.
        seat: `[2]` int32 permutation; `seat[p]` is the seat of player `p`.
        game_state: The state of the game's rules.
    """

    key: jax.Array
    seat: jax.Array
    game_state: S


class Game[S: GameState](abc.ABC):
    """A two-player zero-sum game, where players take turns.

    Players interact with a game through `reset` and `step`. At every reset,
    a fair coin assigns the two players to the game's two seats, where seat
    `i` is OpenSpiel's player `i`.

    In games where both seats move at once, seat 0 moves first, and seat 1
    moves without seeing seat 0's move, as in OpenSpiel's
    `turn_based_simultaneous_game`. The two games have the same information
    states, so they have the same equilibria.

    To add a game, subclass `Game`, set `num_actions` and `observation_shape`,
    and implement the rules: `initial_states`, `apply_action`, `observe`,
    `legal_action_mask`, and `returns`. The rules deal with seats only;
    `reset` and `step` translate between seats and players.

    Attributes:
        num_actions: Number of actions.
        observation_shape: Shape of one player's observation.
    """

    num_actions: int
    observation_shape: tuple[int, ...]

    def reset(self, key: jax.Array) -> tuple[State[S], TimeStep]:
        """Starts an episode, assigning players to seats at random."""
        state = self._initial_state(key)
        return state, self._timestep(state, reward=jnp.zeros(2))

    def step(
        self, state: State[S], action: jax.Array
    ) -> tuple[State[S], TimeStep]:
        """Advances the episode by one step.

        Args:
            state: The current state.
            action: `[]` int32 action of the player to act.

        Returns:
            The next state and timestep. Once the episode is done, `step`
            leaves the state unchanged and returns zero rewards.
        """
        state, reward = self._advance(state, action)
        return state, self._timestep(state, reward)

    def _initial_state(self, key: jax.Array) -> State[S]:
        """Returns the state of a new episode, without its timestep."""
        key, seat_key, chance_key = jax.random.split(key, 3)
        # Compute the chance tables once, when tracing, not at every call.
        with jax.ensure_compile_time_eval():
            states, probs = self.initial_states()
            cumulative = jnp.cumsum(probs)
        u = jax.random.uniform(chance_key, maxval=cumulative[-1])
        # Unrolled, the binary search needs no loop on accelerators.
        i = jnp.searchsorted(
            cumulative, u, side="right", method="scan_unrolled"
        )
        return State(
            key=key,
            # A fair coin swaps the seats of both players or of neither.
            seat=jnp.arange(2) ^ jax.random.bernoulli(seat_key),
            game_state=jax.tree.map(lambda x: x[i], states),
        )

    def _advance(
        self, state: State[S], action: jax.Array
    ) -> tuple[State[S], jax.Array]:
        """Returns the next state and the `[2]` reward of each player."""
        # A permutation of two elements is its own inverse, so indexing with
        # `seat` maps seat-indexed arrays to player-indexed ones.
        old = state.game_state
        new = self.apply_action(old, action)
        new = jax.tree.map(lambda o, n: jnp.where(old.done, o, n), old, new)
        reward = (self.returns(new) - self.returns(old))[state.seat]
        return dataclasses.replace(state, game_state=new), reward

    def _timestep(self, state: State[S], reward: jax.Array) -> TimeStep:
        game_state = state.game_state
        seat = game_state.current_seat
        return TimeStep(
            observation=self.observe(game_state, seat),
            legal_action_mask=self.legal_action_mask(game_state, seat),
            reward=reward.astype(jnp.float32),
            done=game_state.done,
            current_player=state.seat[seat],
        )

    @abc.abstractmethod
    def initial_states(self) -> tuple[S, np.ndarray]:
        """Returns every possible initial state, stacked, and its probability.

        All chance events, such as card deals, happen at the start, so initial
        states differ only in their outcomes. Probabilities are float64.
        """

    @abc.abstractmethod
    def apply_action(self, state: S, action: jax.Array) -> S:
        """Returns the state after the seat to act takes `action`.

        Args:
            state: The current state. If it is done, the result is discarded.
            action: `[]` int32 action of `state.current_seat`.
        """

    @abc.abstractmethod
    def observe(self, state: S, seat: jax.Array) -> jax.Array:
        """Returns the information state of `seat` as a float32 tensor.

        The tensor is a one-hot encoding of `seat` followed by OpenSpiel's
        information-state tensor for that seat. If OpenSpiel's tensor already
        starts with the seat, it is not repeated.
        """

    @abc.abstractmethod
    def legal_action_mask(self, state: S, seat: jax.Array) -> jax.Array:
        """Returns the `[num_actions]` bool mask of legal actions of `seat`."""

    @abc.abstractmethod
    def returns(self, state: S) -> jax.Array:
        """Returns the `[2]` total reward of each seat so far."""

    @functools.cached_property
    def sequence_form(self) -> sequence_form_lib.SequenceForm:
        """The sequence form of the game, for exact evaluation.

        Built on first access and cached with the game. By default, enumerates
        the game tree; games whose tree doesn't fit in memory override this.
        """
        return sequence_form_lib.enumerate_tree(self)


def auto_reset(
    game: Game,
) -> Callable[[State, jax.Array], tuple[State, TimeStep]]:
    """Returns `game.step`, modified to start a new episode when one ends.

    On the step that ends an episode, the timestep keeps that step's `reward`
    and `done`. Its other fields, and the returned state, belong to the first
    step of the next episode.
    """

    def step(state: State, action: jax.Array) -> tuple[State, TimeStep]:
        state, reward = game._advance(state, action)
        done = state.game_state.done
        state = jax.tree.map(
            lambda n, o: jnp.where(done, n, o),
            game._initial_state(state.key),
            state,
        )
        timestep = game._timestep(state, reward)
        return state, dataclasses.replace(timestep, done=done)

    return step

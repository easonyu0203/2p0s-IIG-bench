"""The game API: `Game`, its `State`, and the `TimeStep` players observe."""

import abc
from collections.abc import Callable
import dataclasses
import functools
from typing import dataclass_transform

import jax
import jax.numpy as jnp

from nashbench import sequence_form as sequence_form_lib

#: `TimeStep.current_player` when both players act at once. Equals OpenSpiel's
#: kSimultaneousPlayerId.
BOTH = -2


@dataclass_transform(frozen_default=True)
def pytree_dataclass[T](cls: type[T]) -> type[T]:
    """Makes `cls` a frozen dataclass that JAX transformations accept."""
    return jax.tree_util.register_dataclass(
        dataclasses.dataclass(frozen=True)(cls)
    )


@pytree_dataclass
class TimeStep:
    """What the players see after `Game.reset` or `Game.step`.

    Arrays with a leading axis of size 2 are indexed by player.
    `observation` and `legal_action_mask` are defined only for the player to
    act and only while `done` is false; other entries are unspecified.
    `reward` is always defined for both players.

    Attributes:
        observation: `[2, *observation_shape]` float32 information states.
        legal_action_mask: `[2, num_actions]` bool.
        reward: `[2]` float32 reward of the last step.
        done: `[]` bool, true once the episode has ended.
        current_player: `[]` int32 player to act: 0, 1, or `BOTH`.
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
        current_seat: `[]` int32 seat to act: 0, 1, or `BOTH`.
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
    """A two-player zero-sum game.

    Players interact with a game through `reset` and `step`. At every reset,
    a fair coin assigns the two players to the game's two seats, where seat
    `i` is OpenSpiel's player `i`.

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
        key, seat_key, chance_key = jax.random.split(key, 3)
        states, probs = self.initial_states()
        i = jax.random.choice(chance_key, probs.size, p=probs)
        state = State(
            key=key,
            seat=jax.random.permutation(seat_key, 2),
            game_state=jax.tree.map(lambda x: x[i], states),
        )
        return state, self._timestep(state, reward=jnp.zeros(2))

    def step(
        self, state: State[S], action: jax.Array
    ) -> tuple[State[S], TimeStep]:
        """Advances the episode by one step.

        Args:
            state: The current state.
            action: `[2]` int32 action of each player. Actions of players who
                are not acting are ignored.

        Returns:
            The next state and timestep. Once the episode is done, `step`
            leaves the state unchanged and returns zero rewards.
        """
        # A permutation of two elements is its own inverse, so indexing with
        # `seat` maps player-indexed arrays to seat-indexed ones and back.
        old = state.game_state
        new = self.apply_action(old, action[state.seat])
        new = jax.tree.map(lambda o, n: jnp.where(old.done, o, n), old, new)
        reward = (self.returns(new) - self.returns(old))[state.seat]
        state = dataclasses.replace(state, game_state=new)
        return state, self._timestep(state, reward)

    def _timestep(self, state: State[S], reward: jax.Array) -> TimeStep:
        game_state = state.game_state
        observe = jax.vmap(self.observe, in_axes=(None, 0))
        legal_action_mask = jax.vmap(self.legal_action_mask, in_axes=(None, 0))
        current_seat = game_state.current_seat
        return TimeStep(
            observation=observe(game_state, state.seat),
            legal_action_mask=legal_action_mask(game_state, state.seat),
            reward=reward.astype(jnp.float32),
            done=game_state.done,
            current_player=jnp.where(
                current_seat == BOTH, BOTH, state.seat[current_seat]
            ),
        )

    @abc.abstractmethod
    def initial_states(self) -> tuple[S, jax.Array]:
        """Returns every possible initial state, stacked, and its probability.

        All chance events, such as card deals, happen at the start, so initial
        states differ only in their outcomes.
        """

    @abc.abstractmethod
    def apply_action(self, state: S, action: jax.Array) -> S:
        """Returns the state after the seats take `action`.

        Args:
            state: The current state. If it is done, the result is discarded.
            action: `[2]` int32 action of each seat. Turn-based games read
                `action[state.current_seat]`.
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
        state, timestep = game.step(state, action)
        next_state, next_timestep = game.reset(state.key)
        next_timestep = dataclasses.replace(
            next_timestep, reward=timestep.reward, done=timestep.done
        )
        return jax.tree.map(
            lambda n, o: jnp.where(timestep.done, n, o),
            (next_state, next_timestep),
            (state, timestep),
        )

    return step

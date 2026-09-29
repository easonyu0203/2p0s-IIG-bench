"""The sequence form of Oshi-Zumo, by enumerating its histories.

Both bids are public, so each history where the game goes on is an
information set of both seats. Histories are numbered level by level, and
each level is sorted by the coins left and the wrestler's position, which
decide what follows. The children of a group of equal histories after one
pair of bids are then a block: consecutive histories with consecutive
parents. The host enumerates blocks, and the accelerator expands them.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from nashbench import sequence_form as sequence_form_lib


def sequence_form(game) -> sequence_form_lib.SequenceForm:
    """Returns the sequence form of an Oshi-Zumo game."""
    blocks, ends, offset = _enumerate(game)
    num_nodes = int(offset[-1])
    parent, legal = _tables(game, num_nodes, *blocks)
    levels = tuple(
        (s * num_nodes + offset[t], s * num_nodes + offset[t + 1])
        for t in range(len(offset) - 1)
        for s in range(2)
    )
    observe = jax.jit(functools.partial(_observe, game))
    num_ends = int(ends[1].sum())
    return sequence_form_lib.SequenceForm(
        levels=levels,
        parent=parent,
        legal_action_mask=legal,
        observe=functools.partial(observe, jnp.asarray(offset), parent),
        gradient=functools.partial(
            _gradient, *_ends(game, num_nodes, num_ends, *ends)
        ),
    )


def _enumerate(game):
    """Enumerates the blocks of histories, level by level.

    Returns:
        The blocks of histories where the game goes on, from history 1 on:
        their first parent, size, and `[2]` bids and coins left; the blocks
        of terminal histories that a seat wins, with the return of seat 0 in
        place of coins; and the first history of each level, then their
        number.
    """
    middle = game.size + 1
    # The groups of the current level: first history, size, coins, position.
    first, size = np.zeros(1, int), np.ones(1, int)
    coins, position = np.full((1, 2), game.coins), np.full(1, middle)
    blocks, ends, offset = [], [], [0, 1]
    while len(size):
        # Each group after each pair of legal bids: 1 to the coins left, or 0.
        choices = np.maximum(coins, 1)
        count = choices.prod(1)
        group = np.repeat(np.arange(len(count)), count)
        first_child = np.cumsum(count) - count
        sibling = np.arange(len(group)) - first_child[group]
        bids = np.stack(np.divmod(sibling, choices[group, 1]), 1)
        bids += coins[group] > 0
        first, size = first[group], size[group]
        coins = coins[group] - bids
        position = position[group] + np.sign(bids[:, 0] - bids[:, 1])
        margin = np.sign(position - middle)
        done = (position == 0) | (position == 2 * middle) | (coins == 0).all(1)
        won = done & (margin != 0)
        ends.append((first[won], size[won], bids[won], margin[won]))
        # The next level's blocks, with equal coins and positions together.
        key = (coins[:, 0] * (game.coins + 1) + coins[:, 1]) * 2 * middle
        key += position
        keep = np.nonzero(~done)[0]
        keep = keep[np.argsort(key[keep])]
        first, size, bids, coins, position, key = (
            x[keep] for x in (first, size, bids, coins, position, key)
        )
        blocks.append((first, size, bids, coins))
        # Its groups merge blocks with equal keys.
        start = np.unique(key, return_index=True)[1]
        first = offset[-1] + (np.cumsum(size) - size)[start]
        size = np.add.reduceat(size, start) if len(start) else start
        coins, position = coins[start], position[start]
        offset.append(offset[-1] + int(size.sum()))
    blocks, ends = (
        tuple(np.concatenate(x) for x in zip(*b, strict=True))
        for b in (blocks, ends)
    )
    return blocks, ends, np.array(offset[:-1])


def _sequences(game, num_nodes, total, first, size, bids):
    """Returns the last sequences of each seat in `total` histories of blocks.

    Also returns the block of each history.
    """
    block = jnp.repeat(jnp.arange(size.size), size, total_repeat_length=total)
    parent = first[block] + jnp.arange(total) - (jnp.cumsum(size) - size)[block]
    seat = jnp.arange(2)[:, None]
    sequence = game.num_actions * (1 + seat * num_nodes + parent)
    return sequence + bids[block].T, block


@functools.partial(jax.jit, static_argnums=(0, 1))
def _tables(game, num_nodes, first, size, bids, coins):
    """Returns the parent sequence and legal actions of information sets."""
    sequence, block = _sequences(
        game, num_nodes, num_nodes - 1, first, size, bids
    )
    # The root follows the empty sequences, with all coins left.
    parent = jnp.concatenate([jnp.arange(2)[:, None], sequence], 1)
    coins = jnp.concatenate([jnp.full((1, 2), game.coins), coins[block]])
    # A seat bids at least 1 coin, or 0 once it has none.
    left = coins.T.reshape(-1, 1)
    bid = jnp.arange(game.num_actions)
    legal = jnp.where(left > 0, (bid >= 1) & (bid <= left), bid == 0)
    return parent.ravel().astype(jnp.int32), legal


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _ends(game, num_nodes, total, first, size, bids, margin):
    """Returns the last sequences and return of seat 0 of terminal histories."""
    sequence, block = _sequences(game, num_nodes, total, first, size, bids)
    return *sequence.astype(jnp.int32), margin[block].astype(jnp.float32)


@jax.jit
def _gradient(seq0, seq1, weight, plan):
    """Returns the gradient of `plan`, summing over terminal histories."""
    gradient = jnp.zeros_like(plan).at[seq0].add(weight * plan[seq1])
    return gradient.at[seq1].add(-weight * plan[seq0])


def _observe(game, offset, parent, ids):
    """Returns observations of information sets, rebuilding their bids."""
    num_nodes = parent.size // 2
    seat, node = jnp.divmod(ids, num_nodes)
    # The turn of the bids that lead to each node, walking up to the root.
    turn = jnp.searchsorted(offset, node, side="right") - 2
    history = jnp.full((ids.size, 2, game.coins), -1)
    for _ in range(game.coins):
        # A node's parent sequences hold its parent and the bids.
        sequence = parent[jnp.stack([node, num_nodes + node], 1)]
        at_turn = jnp.arange(game.coins) == turn[:, None, None]
        bids = sequence[:, :, None] % game.num_actions
        history = jnp.where(at_turn, bids, history)
        # The root is its own parent.
        node = jnp.maximum(sequence[:, 0] // game.num_actions - 1, 0)
        turn -= 1
    states = jax.tree.map(
        lambda x: jnp.repeat(x, ids.size, 0), game.initial_states()[0]
    )
    states = dataclasses.replace(states, bids=history)
    return jax.vmap(game.observe)(states, seat)

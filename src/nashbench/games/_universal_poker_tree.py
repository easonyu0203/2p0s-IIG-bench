"""The sequence form of limit poker, from its public betting tree.

Bets are public and don't depend on the cards. At a public node where a seat
acts in round `j`, the seat's information sets are its private card followed
by the public cards of rounds 1 to `j`: partial permutations of the cards.
They are numbered `start + rank`, where `rank` orders the permutations
lexicographically, so that dropping the last cards divides it.

Both seats act last in the round where a terminal node ends, so they see the
same public cards there. Given those cards, a seat's return is a matrix over
the two private cards: constant for a fold, and the sign of the difference in
strength at a showdown.
"""

import dataclasses
import functools
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

from nashbench import sequence_form as sequence_form_lib


def sequence_form(game) -> sequence_form_lib.SequenceForm:
    """Returns the sequence form of a limit poker game."""
    nodes, terminals = _public_tree(game)
    d, num_actions = game.num_cards, game.num_actions
    # Number public nodes by seat, then by level.
    order = np.lexsort((nodes["level"], nodes["seat"]))
    number = np.argsort(order)
    nodes = jax.tree.map(lambda x: x[order], nodes)
    terminals["last"][..., 0] = number[terminals["last"][..., 0]]
    sizes = np.array([math.perm(d, r + 1) for r in nodes["round"]])
    start = np.cumsum(sizes) - sizes

    # At a node, the information set of rank `k` continues the sequence
    # `first + scale * (k // divisor)`: dividing drops the public cards
    # revealed since the seat's previous node. At a seat's first node, it
    # continues the seat's empty sequence.
    has_parent = nodes["parent"] >= 0
    parent = number[np.maximum(nodes["parent"], 0)]
    first = np.where(
        has_parent,
        num_actions * (start[parent] + 1) + nodes["action"],
        nodes["seat"],
    )
    scale = np.where(has_parent, num_actions, 0)
    divisor = sizes // np.where(has_parent, sizes[parent], sizes)
    parent, legal = _tables(
        int(sizes.sum()),
        *map(jnp.asarray, (sizes, first, scale, divisor, nodes["legal"])),
    )
    levels = []
    for level, seat in itertools.product(
        range(nodes["level"].max() + 1), range(2)
    ):
        ids = np.flatnonzero(
            (nodes["level"] == level) & (nodes["seat"] == seat)
        )
        if ids.size:
            levels.append(
                (int(start[ids[0]]), int(start[ids[-1]] + sizes[ids[-1]]))
            )
    return sequence_form_lib.SequenceForm(
        levels=tuple(levels),
        parent=parent,
        legal_action_mask=legal,
        observe=jax.jit(functools.partial(_observe, game, nodes, start)),
        gradient=jax.jit(
            functools.partial(_gradient, _groups(game, terminals, start))
        ),
    )


def _public_tree(game):
    """Plays every betting sequence with one deal.

    Returns:
        The decision nodes: the seat to act, the round, the legal actions,
        the seat's previous node and its action there (or -1), the seat's
        level (number of actions so far), and the state. The terminal nodes:
        the round, the chips each seat spent, whether each folded, and each
        seat's last node and action.
    """
    apply_action = sequence_form_lib._batched(game.apply_action)
    legal_action_mask = sequence_form_lib._batched(game.legal_action_mask)
    take = sequence_form_lib._take
    states = jax.tree.map(lambda x: np.asarray(x[:1]), game.initial_states()[0])
    # Per history: each seat's last node and action, and its level.
    last = np.full((1, 2, 2), -1)
    level = np.zeros((1, 2), int)
    nodes, terminals = [], []
    while len(level):
        done = states.done
        terminals.append(
            {
                "round": states.round[done],
                "spent": states.spent[done],
                "folded": states.folded[done],
                "last": last[done],
            }
        )
        states, last, level = take((states, last, level), ~done)
        seat = states.current_seat
        rows = np.arange(len(seat))
        ids = sum(len(n["seat"]) for n in nodes) + rows
        mask = legal_action_mask(states, seat)
        nodes.append(
            {
                "seat": seat,
                "round": states.round,
                "legal": mask,
                "parent": last[rows, seat, 0],
                "action": last[rows, seat, 1],
                "level": level[rows, seat],
                "state": states,
            }
        )
        history, action = np.nonzero(mask)
        states = apply_action(take(states, history), action)
        seat, last, level = seat[history], last[history], level[history]
        rows = np.arange(len(history))
        last[rows, seat] = np.stack([ids[history], action], 1)
        level[rows, seat] += 1
    return (
        jax.tree.map(lambda *x: np.concatenate(x), *nodes),
        jax.tree.map(lambda *x: np.concatenate(x), *terminals),
    )


@functools.partial(jax.jit, static_argnums=0)
def _tables(num_infosets, sizes, first, scale, divisor, legal):
    """Returns the parent sequence and legal actions of information sets."""
    node = jnp.repeat(
        jnp.arange(sizes.size), sizes, total_repeat_length=num_infosets
    )
    rank = jnp.arange(num_infosets) - (jnp.cumsum(sizes) - sizes)[node]
    parent = first[node] + scale[node] * (rank // divisor[node])
    return parent.astype(jnp.int32), legal[node]


def _groups(game, terminals, start):
    """Groups terminal nodes by round, and by fold or showdown.

    Returns:
        Per group, as arrays: each terminal's last sequence of each seat with
        the lowest cards, `[Z, 2]`; seat 0's return per probability of a
        deal, which multiplies the sign of the difference in strength at a
        showdown, `[Z]`; the offset of the sequences of each seat's private
        card with each public cards, `[B, C]`; and at a showdown, the signs,
        `[B, C, C]`, or None.
    """
    d, num_actions = game.num_cards, game.num_actions
    node, action = terminals["last"][..., 0], terminals["last"][..., 1]
    first = num_actions * (start[node] + 1) + action
    spent, folded = terminals["spent"], terminals["folded"]
    value = np.where(folded[:, 0], -spent[:, 0], spent[:, 1])
    showdown = ~folded.any(1)
    groups = []
    for j, show in itertools.product(range(game.num_rounds), (False, True)):
        z = np.flatnonzero((terminals["round"] == j) & (showdown == show))
        if not z.size:
            continue
        # The rank of each private card with each public cards, `[B, C]`:
        # deals sorted by the public cards, then by the private card
        # (`lexsort` sorts by its last key first).
        cards = np.array(list(itertools.permutations(range(d), j + 1)))
        rank = np.lexsort([cards[:, 0], *cards[:, :0:-1].T])
        rank = rank.reshape(math.perm(d, j), d - j)
        cards = cards[rank]
        sign = None
        if show:
            ranks = cards // (d // game.num_ranks)
            strength = game._strength[tuple(np.moveaxis(ranks, -1, 0))]
            sign = np.sign(strength[..., :, None] - strength[..., None, :])
        # A deal of both private cards and the public cards so far.
        probability = 1 / math.perm(d, j + 2)
        groups.append(
            (first[z], value[z] * probability, num_actions * rank, sign)
        )
    return groups


def _gradient(groups, plan):
    """Returns the gradient of `plan`, summing over terminal nodes."""
    gradient = jnp.zeros_like(plan)
    for first, value, offset, sign in groups:
        seq = jnp.asarray(first)[:, :, None, None] + jnp.asarray(offset)
        seq0, seq1 = seq[:, 0], seq[:, 1]  # [Z, B, C]
        x0, x1 = plan[seq0], plan[seq1]
        v = jnp.asarray(value)[:, None, None]
        if sign is None:
            # The other seat holds any private card but the seat's own.
            y0 = v * (x1.sum(-1, keepdims=True) - x1)
            y1 = -v * (x0.sum(-1, keepdims=True) - x0)
        else:
            y0 = v * jnp.einsum("bij,zbj->zbi", sign, x1)
            y1 = -v * jnp.einsum("bij,zbi->zbj", sign, x0)
        gradient = gradient.at[seq0].add(y0).at[seq1].add(y1)
    return gradient


def _observe(game, nodes, start, ids):
    """Returns observations of information sets, from representative states.

    The other seat's private card and the public cards not yet revealed are
    the lowest cards left.
    """
    d, r = game.num_cards, game.num_rounds
    start = jnp.asarray(start)
    node = (
        jnp.searchsorted(start, ids, side="right", method="scan_unrolled") - 1
    )
    seat = jnp.asarray(nodes["seat"])[node]
    # Extend each rank to that of a whole deal: the seat's private card, the
    # public cards, then the other seat's private card.
    scale = [math.perm(d, r + 1) // math.perm(d, j + 1) for j in range(r)]
    round_ = jnp.asarray(nodes["round"])[node]
    rank = (ids - start[node]) * jnp.asarray(scale)[round_]
    deal = jnp.asarray(list(itertools.permutations(range(d), r + 1)))[rank]
    private = jnp.where(
        jnp.arange(2) == seat[:, None], deal[:, :1], deal[:, -1:]
    )
    states = jax.tree.map(lambda x: jnp.asarray(x)[node], nodes["state"])
    states = dataclasses.replace(
        states, cards=jnp.concatenate([private, deal[:, 1:r]], 1)
    )
    return jax.vmap(game.observe)(states, seat)

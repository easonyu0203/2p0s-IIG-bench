"""The sequence form of Goofspiel, from the regular structure of its tree.

At turn `t`, a seat's information set is the first `t + 1` point cards, its
first `t` bids, and who won each of the first `t` turns. Point cards are
independent of bids, so the information sets of turn `t` are numbered

    offset[t] + rank(point cards) * num_pairs[t] + pair,

where `pair` numbers the (bids, winners) that some opponent bids make
possible. A terminal history is a triple of permutations: the point cards and
each seat's bids.
"""

import dataclasses
import functools
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

from nashbench import sequence_form as sequence_form_lib

# Information sets per chunk when building tables.
_CHUNK = 2**22


def sequence_form(game) -> sequence_form_lib.SequenceForm:
    """Returns the sequence form of a Goofspiel game."""
    k = game.num_cards
    tables = _Tables(k)
    ids = jnp.arange(2 * tables.num_infosets)
    parent, legal = (
        jnp.concatenate(x)
        for x in zip(
            *(
                _parent_and_legal(tables, ids[i : i + _CHUNK])
                for i in range(0, ids.size, _CHUNK)
            ),
            strict=True,
        )
    )
    levels = tuple(
        (
            s * tables.num_infosets + int(tables.offset[t]),
            s * tables.num_infosets + int(tables.offset[t + 1]),
        )
        for t in range(k - 1)
        for s in range(2)
    )
    observe = jax.jit(functools.partial(_observe, game, tables))
    return sequence_form_lib.SequenceForm(
        levels=levels,
        parent=parent,
        legal_action_mask=legal,
        observe=observe,
        gradient=jax.jit(functools.partial(_gradient, tables)),
    )


class _Tables:
    """Small tables that number Goofspiel's information sets and leaves.

    Attributes:
        k: Number of cards.
        perms: `[k!, k]` permutations of the cards, in lexicographic order.
            The first `m` cards of permutation `i` have rank `i // (k - m)!`
            among partial permutations of length `m`.
        pair_of, pair_of_base: Maps of every turn `t`, concatenated, and where
            each starts. A map takes `rank(bids) * 3**t + winners` to the pair,
            or -1 if no opponent bids produce those winners. Winners are base-3
            digits, one per turn: 0 lost, 1 tied, 2 won.
        last_pair_of: The map of the last decision turn, as a numpy array.
        pair_bids, pair_opponent, pair_base: Per pair of every turn,
            concatenated, the seat's bids and opponent bids that produce its
            winners, padded with -1, and where each turn's pairs start.
        num_pairs, offset: Per turn, the number of pairs and the number of
            information sets before the turn's.
        num_infosets: Number of information sets of a seat.
    """

    def __init__(self, k):
        self.k = k
        perms = np.array(list(itertools.permutations(range(k))))
        pair_of, pair_bids, pair_opponent, num_pairs = [], [], [], []
        for t in range(k - 1):  # The last turn plays itself.
            mine = perms[:: math.factorial(k - t), :t]
            witness = {}
            for rank, bids in enumerate(mine):
                for theirs in mine:
                    key = rank * 3**t + _code(np.sign(bids - theirs) + 1)
                    witness.setdefault(int(key), theirs)
            keys = sorted(witness)
            table = np.full(len(mine) * 3**t, -1)
            table[keys] = np.arange(len(keys))
            pair_of.append(table)
            for key in keys:
                pair_bids.append(
                    np.pad(mine[key // 3**t], (0, k - t), constant_values=-1)
                )
                pair_opponent.append(
                    np.pad(witness[key], (0, k - t), constant_values=-1)
                )
            num_pairs.append(len(keys))
        self.offset = np.cumsum(
            [0] + [math.perm(k, t + 1) * n for t, n in enumerate(num_pairs)]
        )
        self.num_infosets = int(self.offset[-1])
        self.num_pairs = num_pairs
        self.perms = jnp.asarray(perms)
        self.last_pair_of = pair_of[-1]
        self.pair_of = jnp.asarray(np.concatenate(pair_of))
        self.pair_of_base = jnp.asarray(
            np.cumsum([0] + [p.size for p in pair_of])
        )
        self.pair_base = jnp.asarray(np.cumsum([0, *num_pairs]))
        self.pair_bids = jnp.asarray(np.array(pair_bids))
        self.pair_opponent = jnp.asarray(np.array(pair_opponent))

    def describe(self, ids):
        """Returns what identifies information sets, and a way to reach them.

        Returns each information set's seat, turn, point-card rank, point
        cards, bids, and opponent bids that produce its winners.
        """
        k, offset = self.k, jnp.asarray(self.offset)
        seat, local = jnp.divmod(ids, self.num_infosets)
        turn = jnp.searchsorted(offset, local, side="right") - 1
        num_pairs = jnp.asarray(self.num_pairs)[turn]
        rank, pair = jnp.divmod(local - offset[turn], num_pairs)
        pair = self.pair_base[turn] + pair
        # Completing the point cards in increasing order gives a permutation.
        factorial = jnp.asarray([math.factorial(n) for n in range(k + 1)])
        points = self.perms[rank * factorial[k - turn - 1]]
        return (
            seat,
            turn,
            rank,
            points,
            self.pair_bids[pair],
            self.pair_opponent[pair],
        )


def _code(digits):
    """Returns base-3 numbers of rows of digits, first digit most significant."""
    return (digits * 3 ** np.arange(digits.shape[-1])[::-1]).sum(-1)


@functools.partial(jax.jit, static_argnums=0)
def _parent_and_legal(tables, ids):
    """Returns the parent sequence and legal actions of information sets."""
    k = tables.k
    seat, turn, rank, _, bids, opponent = tables.describe(ids)
    # The parent drops the last turn: its point card, bid, and winner.
    previous = jnp.maximum(turn - 1, 0)
    active = jnp.arange(k) < previous[:, None]
    own_rank = jnp.zeros_like(ids)
    for i in range(k):
        smaller_unused = (jnp.arange(k) < bids[:, i : i + 1]).sum(1) - (
            (bids[:, :i] < bids[:, i : i + 1]).sum(1)
        )
        own_rank = jnp.where(
            active[:, i], own_rank * (k - i) + smaller_unused, own_rank
        )
    digits = jnp.where(active, jnp.sign(bids - opponent) + 1, 0)
    power = jnp.maximum(previous[:, None] - 1 - jnp.arange(k), 0)
    winners = (digits * 3**power).sum(1)
    pair = tables.pair_of[
        tables.pair_of_base[previous] + own_rank * 3**previous + winners
    ]
    parent_rank = rank // (k - turn)
    offset, num_pairs = (
        jnp.asarray(tables.offset),
        jnp.asarray(tables.num_pairs),
    )
    parent = (
        seat * tables.num_infosets
        + offset[previous]
        + parent_rank * num_pairs[previous]
        + pair
    )
    last_bid = jnp.take_along_axis(bids, previous[:, None], 1)[:, 0]
    parent = jnp.where(turn == 0, seat, k * (parent + 1) + last_bid)
    legal = ~(bids[:, :, None] == jnp.arange(k)).any(1)
    return parent.astype(jnp.int32), legal


def _gradient(tables, plan):
    """Returns the gradient of `plan`, summing over all terminal histories."""
    k, perms = tables.k, tables.perms
    t = k - 2  # Last decision turn.
    # For every pair of bid permutations: each seat's pair and last decision,
    # and who won each turn (+1 seat 0, -1 seat 1).
    i0, i1 = jnp.divmod(jnp.arange(perms.shape[0] ** 2), perms.shape[0])
    b0, b1 = perms[i0], perms[i1]
    code = 3 ** jnp.arange(t)[::-1]
    last_pair_of = jnp.asarray(tables.last_pair_of)
    pair0 = last_pair_of[
        (i0 // 2) * 3**t + ((jnp.sign(b0 - b1) + 1)[:, :t] * code).sum(1)
    ]
    pair1 = last_pair_of[
        (i1 // 2) * 3**t + ((jnp.sign(b1 - b0) + 1)[:, :t] * code).sum(1)
    ]
    won = jnp.sign(b0 - b1)
    offset, num_pairs = int(tables.offset[t]), tables.num_pairs[t]
    chunk = math.gcd(8, perms.shape[0])  # Point-card permutations per step.

    def body(i, gradient):
        points = jax.lax.dynamic_slice_in_dim(perms, i * chunk, chunk) + 1
        weight = jnp.sign(won @ points.T) / perms.shape[0]  # [pairs, chunk]
        infoset = offset + (i * chunk + jnp.arange(chunk)) * num_pairs
        seq0 = k * (infoset + pair0[:, None] + 1) + b0[:, t, None]
        seq1 = (
            k * (tables.num_infosets + infoset + pair1[:, None] + 1)
            + b1[:, t, None]
        )
        gradient = gradient.at[seq0].add(weight * plan[seq1])
        return gradient.at[seq1].add(-weight * plan[seq0])

    return jax.lax.fori_loop(
        0, perms.shape[0] // chunk, body, jnp.zeros_like(plan)
    )


def _observe(game, tables, ids):
    """Returns observations of information sets, from representative states."""
    seat, turn, _, points, bids, opponent = tables.describe(ids)
    states = jax.tree.map(
        lambda x: jnp.repeat(x[:1], ids.shape[0], 0), game.initial_states()[0]
    )
    own = jnp.arange(2)[None, :, None] == seat[:, None, None]
    states = dataclasses.replace(
        states,
        point_cards=points,
        bids=jnp.where(own, bids[:, None], opponent[:, None]),
        turn=turn,
    )
    return jax.vmap(game.observe)(states, seat)

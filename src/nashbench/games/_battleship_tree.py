"""The sequence form of Battleship, from the regular structure of its tree.

A seat first places its ship, at a single information set. At its `i`-th
shot, its information set has two independent parts: its shots and which of
them hit, numbered `a` among those that an afloat ship explains, `A[i]`; and
its placement and the other seat's `j = i + seat` shots, numbered `b` among
those that leave its ship afloat, `B[j]`. These information sets are
numbered `start + a * len(B[j]) + b`.

Returns are 0 unless a ship sinks. A sinking shot and the sinker's `a` fix
the sunk ship, and the sinker's `b` fixes the victim's last shot, so each
sinking (`a`, shot) and each `b` make one terminal history.
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
    """Returns the sequence form of a Battleship game."""
    t = _Tables(game)
    num_actions = game.num_actions
    legal_shots = np.arange(num_actions) < game.height * game.width
    parent, legal = [], []
    for seat, level in itertools.product(range(2), range(game.num_shots + 1)):
        start = t.start[seat, level]
        if not level:
            parent.append(np.full(1, seat))
            legal.append(game._ship.any(1)[None])
            continue
        i = level - 1
        nb = len(t.b_place[i + seat])
        # The information set of the seat's previous action, per `a` and `b`.
        if i:
            previous = (
                t.start[seat, level - 1]
                + t.a_parent[i - 1][:, None] * len(t.b_place[i + seat - 1])
                + t.b_parent[i + seat - 1]
            )
            action = t.a_shots[i][:, -1:]
        else:
            previous = np.full((1, nb), start - 1)  # The placement.
            action = t.place[t.b_place[seat]]
        parent.append((num_actions * (previous + 1) + action).ravel())
        free = np.ones((len(t.a_shots[i]), num_actions), bool) & legal_shots
        np.put_along_axis(free, t.a_shots[i], False, 1)
        legal.append(np.repeat(free, nb, 0))
    levels = tuple(
        (int(t.start[seat, level]), int(t.end[seat, level]))
        for level in range(game.num_shots + 1)
        for seat in range(2)
        if t.end[seat, level] > t.start[seat, level]
    )
    return sequence_form_lib.SequenceForm(
        levels=levels,
        parent=jnp.asarray(np.concatenate(parent), jnp.int32),
        legal_action_mask=jnp.asarray(np.concatenate(legal)),
        observe=jax.jit(functools.partial(_observe, game, t)),
        gradient=jax.jit(functools.partial(_gradient, num_actions, t)),
    )


class _Tables:
    """Tables that number Battleship's information sets and terminals.

    Lists are indexed by the number of shots. Shots index partial
    permutations of the cells in lexicographic order, and hits are bits, the
    first shot's most significant.

    Attributes:
        place: `[P]` placement actions.
        a_shots, a_hits, a_witness: Per `A[i]`: the shots, hit bits, and a
            placement that explains them.
        a_parent: Per `A[i + 1]`, the parent in `A[i]`.
        b_place, b_shots: Per `B[j]`: the placement (an index into `place`),
            and the other seat's shots.
        b_parent: Per `B[j + 1]`, the parent in `B[j]`.
        start, end: `[2, num_shots + 1]` information sets of each seat's
            placement (level 0) and shots, as ranges.
        sinks: Per `A[i]`, the `a` one hit short of sinking, the sinking
            shot, the sunk placement, and the victim's `b` in `B[i]`.
        last: Per `B[j + 1]`, the other seat's last shot, and its `a` before
            it.
    """

    def __init__(self, game):
        n, s, size = game.height * game.width, game.num_shots, game.ship_size
        self.place = np.flatnonzero(game._ship.any(1))
        cells = game._ship[self.place]  # [P, n]
        perms = [
            np.array(list(itertools.permutations(range(n), k)), int).reshape(
                math.perm(n, k), k
            )
            for k in range(s + 1)
        ]
        afloat = [cells[:, p].sum(-1) < size for p in perms]

        self.a_shots, self.a_hits, self.a_witness, a_index = [], [], [], []
        for i in range(s):
            # Shots and hits as one key: the shots' index, then the hits.
            key = np.arange(len(perms[i])) << i | _bits(cells[:, perms[i]])
            witness, _ = np.nonzero(afloat[i])
            key, first = np.unique(key[afloat[i]], return_index=True)
            self.a_shots.append(perms[i][key >> i])
            self.a_hits.append(key & ((1 << i) - 1))
            self.a_witness.append(witness[first])
            a_index.append(np.full(len(perms[i]) << i, -1))
            a_index[i][key] = np.arange(len(key))
        self.a_parent = [
            a_index[i - 1][
                _index(self.a_shots[i][:, :-1], n) << (i - 1)
                | self.a_hits[i] >> 1
            ]
            for i in range(1, s)
        ]

        self.b_place, self.b_shots, b_index = [], [], []
        for j in range(s + 1):
            ship, perm = np.nonzero(afloat[j])
            self.b_place.append(ship)
            self.b_shots.append(perms[j][perm])
            b_index.append(np.full(afloat[j].shape, -1))
            b_index[j][ship, perm] = np.arange(len(ship))
        self.b_parent = [
            b_index[j - 1][self.b_place[j], _index(self.b_shots[j][:, :-1], n)]
            for j in range(1, s + 1)
        ]

        sizes = np.array(
            [
                [1]
                + [
                    len(a) * len(self.b_place[i + seat])
                    for i, a in enumerate(self.a_shots)
                ]
                for seat in range(2)
            ]
        )
        self.end = np.cumsum(sizes).reshape(2, -1)
        self.start = self.end - sizes

        self.last = []
        for j in range(1, s + 1):
            shots, ship = self.b_shots[j], self.b_place[j]
            key = _index(shots[:, :-1], n) << (j - 1) | _bits(
                cells[ship[:, None], shots[:, :-1]]
            )
            self.last.append((shots[:, -1], a_index[j - 1][key]))

        self.sinks = []
        for i in range(s):
            shots, hits = self.a_shots[i], self.a_hits[i]
            explains = _bits(cells[:, shots]).T == hits[:, None]  # [A, P]
            near = np.bitwise_count(hits) == size - 1
            a, ship = np.nonzero(explains & near[:, None])
            shot = np.zeros((len(shots), n), bool)
            np.put_along_axis(shot, shots, True, 1)
            target = np.argmax(cells[ship] & ~shot[a], 1)
            victim = b_index[i][ship, _index(shots[a], n)]
            self.sinks.append((a, target, ship, victim))


def _bits(hits):
    """Returns the bits of the last axis, the first most significant."""
    k = hits.shape[-1]
    return (hits.astype(int) << np.arange(k)[::-1]).sum(-1)


def _index(perms, n):
    """Returns the lexicographic indices of partial permutations of `n`."""
    index = np.zeros(len(perms), int)
    for i in range(perms.shape[1]):
        unused = np.arange(n) < perms[:, i : i + 1]
        unused = unused.sum(1) - (perms[:, :i] < perms[:, i : i + 1]).sum(1)
        index = index * (n - i) + unused
    return index


def _gradient(num_actions, t, plan):
    """Returns the gradient of `plan`, summing over histories with a sinking.

    The sinker wins 1 and the victim loses 1.
    """
    gradient = jnp.zeros_like(plan)
    for (i, sinks), x in itertools.product(enumerate(t.sinks), range(2)):
        # Seat `x` sinks with its shot `i`, after `j` shots of the victim.
        a, target, ship, victim = map(jnp.asarray, sinks)
        j = i + x
        b = jnp.arange(len(t.b_place[j]))
        infoset = t.start[x, i + 1] + a[:, None] * b.size + b
        seq_x = num_actions * (infoset + 1) + target[:, None]
        if j:
            shot, a_y = map(jnp.asarray, t.last[j - 1])
            infoset = (
                t.start[1 - x, j] + a_y * len(t.b_place[i]) + victim[:, None]
            )
            seq_y = num_actions * (infoset + 1) + shot
        else:
            # The victim hasn't shot; its last sequence is its placement.
            place = jnp.asarray(t.place)[ship]
            seq_y = num_actions * (t.start[1, 0] + 1) + place[:, None]
            seq_y = jnp.broadcast_to(seq_y, seq_x.shape)
        seq_x, seq_y = seq_x.ravel(), seq_y.ravel()
        gradient = gradient.at[seq_x].add(plan[seq_y])
        gradient = gradient.at[seq_y].add(-plan[seq_x])
    return gradient


def _observe(game, t, ids):
    """Returns observations of information sets, from representative states.

    At a shot, the state holds the seat's placement and the other seat's
    shots from `b`, and the seat's shots from `a`, against a ship that
    explains their hits.
    """
    s = game.num_shots

    def pad(shots):
        return np.pad(
            shots, ((0, 0), (0, s - shots.shape[1])), constant_values=-1
        )

    # `A` and `B` of all numbers of shots, concatenated.
    a_shots = jnp.asarray(np.concatenate([pad(x) for x in t.a_shots]))
    a_ship = jnp.asarray(t.place[np.concatenate(t.a_witness)])
    b_shots = jnp.asarray(np.concatenate([pad(x) for x in t.b_shots]))
    b_ship = jnp.asarray(t.place[np.concatenate(t.b_place)])
    a_first = np.cumsum([0] + [len(x) for x in t.a_shots])
    b_first = np.cumsum([0] + [len(x) for x in t.b_place])
    # Per seat and level: the size of `B`, and the first rows of `A` and `B`.
    num_b, first = np.ones((2, s + 1), int), np.zeros((2, 2, s + 1), int)
    for seat, i in itertools.product(range(2), range(s)):
        num_b[seat, i + 1] = len(t.b_place[i + seat])
        first[:, seat, i + 1] = a_first[i], b_first[i + seat]

    start = jnp.asarray(t.start.ravel())
    k = jnp.searchsorted(start, ids, side="right") - 1
    seat, level = jnp.divmod(k, s + 1)
    a, b = jnp.divmod(ids - start[k], jnp.asarray(num_b.ravel())[k])
    a, b = jnp.asarray(first.reshape(2, -1))[:, k] + jnp.stack([a, b])
    own = (jnp.arange(2) == seat[:, None])[..., None]
    shooting = (level > 0)[:, None, None]
    placements = jnp.where(own, b_ship[b, None, None], a_ship[a, None, None])
    shots = jnp.where(own, a_shots[a, None], b_shots[b, None])
    states = jax.tree.map(
        lambda x: jnp.repeat(x, ids.size, 0), game.initial_states()[0]
    )
    states = dataclasses.replace(
        states,
        current_seat=seat,
        placements=jnp.where(shooting, placements, -1)[..., 0],
        shots=jnp.where(shooting, shots, -1),
    )
    return jax.vmap(game.observe)(states, seat)

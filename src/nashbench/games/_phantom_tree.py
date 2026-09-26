"""The sequence form of 3x3 phantom games, by traversal on the accelerator.

These trees have about 10^10 histories, too many to store. Each evaluation
therefore traverses the whole tree depth-first inside one `lax.while_loop`,
expanding a chunk of nodes per iteration. A node is three int32s: the board
and one number per seat. Devices split the tree by the subtrees at depth
`SPLIT_DEPTH`, and traverse their parts in parallel.
"""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from nashbench import sequence_form as sequence_form_lib

NUM_CELLS = 9
SPLIT_DEPTH = 4
# Board bits: stones of seat 0, stones of seat 1, stones discovered by the
# opponent (9 bits each), then the seat to move.
SEAT_BIT = 3 * NUM_CELLS
# Out-of-bounds index for writes to drop.
_DROP = 2**30
# Information sets per chunk when building tables.
_TABLE_CHUNK = 2**20

# A history of `k` tries is numbered `OFFSET[k] + rank * 2**k + placed`: `rank`
# numbers the sequence of tried cells in mixed radix (9, 8, ...), and the bits
# of `placed` say which tries placed a stone. Decisions happen after at most 8
# tries.
OFFSET = np.cumsum([0] + [math.perm(9, k) * 2**k for k in range(9)])
NUM_HISTORIES = int(OFFSET[-1])
_OFFSET = jnp.asarray(OFFSET, jnp.int32)


def sequence_form(game) -> sequence_form_lib.SequenceForm:
    """Returns the sequence form of a phantom game."""
    devices = jax.devices()
    win = _win_table(game)
    seen = _check_stack(_build(game, win, devices))
    histories = tuple(
        jnp.nonzero(seen[s])[0].astype(jnp.int32) for s in range(2)
    )
    del seen
    history = jnp.concatenate(histories)
    seat = (jnp.arange(history.size) >= histories[0].size).astype(jnp.int32)
    # Tables are built in chunks of information sets to bound memory.
    tables = [
        _tables(
            histories, history[i : i + _TABLE_CHUNK], seat[i : i + _TABLE_CHUNK]
        )
        for i in range(0, history.size, _TABLE_CHUNK)
    ]
    parent, legal, next_infoset, k = (
        jnp.concatenate(t) for t in zip(*tables, strict=True)
    )
    roots = (
        jnp.full((1, NUM_CELLS, 2), -1)
        .at[0, :2]
        .set(jnp.asarray([0, histories[0].size])[:, None])
    )
    next_infoset = jnp.concatenate([roots, next_infoset])

    levels = []
    k, seats = np.asarray(k), np.asarray(seat)
    for level in range(NUM_CELLS):
        for s in range(2):
            ids = np.nonzero((k == level) & (seats == s))[0]
            if ids.size:
                levels.append((int(ids[0]), int(ids[-1]) + 1))

    observe = jax.jit(functools.partial(_observe, game))
    gradient = _gradient(game, win, devices)
    return sequence_form_lib.SequenceForm(
        levels=tuple(levels),
        parent=parent,
        legal_action_mask=legal,
        observe=lambda ids: observe(history[ids], seat[ids]),
        gradient=lambda plan: _check_stack(
            gradient(plan.reshape(-1, NUM_CELLS), next_infoset)
        ).ravel(),
    )


@jax.jit
def _tables(histories, history, seat):
    """Returns tables of information sets, given their histories.

    The tables are the parent sequence, legal actions, the information set
    reached after each action and outcome, and the level.
    """

    def infoset_of(history):
        """Returns the information set of the seat with `history`, or -1."""
        found = []
        for s, sorted_histories in enumerate(histories):
            i = jnp.searchsorted(sorted_histories, history)
            i = jnp.minimum(i, sorted_histories.size - 1)
            match = sorted_histories[i] == history
            found.append(jnp.where(match, s * histories[0].size + i, -1))
        seats = seat.reshape(-1, *[1] * (history.ndim - 1))
        return jnp.where(seats == 0, found[0], found[1])

    cells, placed, k = _decode(history)
    # The parent sequence is the last try; the parent history drops it.
    last = jnp.take_along_axis(cells, jnp.maximum(k - 1, 0)[:, None], 1)[:, 0]
    parent = infoset_of(_encode(cells, placed, k - 1))
    parent = jnp.where(k == 0, seat, NUM_CELLS * (parent + 1) + last)
    tried = _tried(cells)
    legal = ((tried[:, None] >> jnp.arange(NUM_CELLS)) & 1) == 0
    # The information set the seat reaches after each action and outcome.
    a = jnp.arange(NUM_CELLS)[None, :, None]
    outcome = jnp.arange(2)[None, None, :]
    child = _extend(history[:, None, None], tried[:, None, None], a, outcome)
    return parent, legal, infoset_of(child), k


def _win_table(game):
    """Returns `[2 * 512]`: whether a seat's stones, as a bitmask, win."""
    stones = (jnp.arange(512)[:, None] >> jnp.arange(NUM_CELLS)) & 1
    return jnp.concatenate(
        [
            jax.vmap(game._has_won, (0, None))(
                jnp.where(stones == 1, seat, -1), seat
            )
            for seat in range(2)
        ]
    )


def _chunk_size():
    """Returns how many nodes to expand per iteration."""
    return 2**12 if jax.default_backend() == "cpu" else 2**18


def _traverse(expand, root, carry):
    """Traverses the tree depth-first from `root`.

    `expand(nodes, valid, carry)` returns the children of a chunk of nodes,
    the position of each child among the chunk's children (`_DROP` to drop
    it), their number, and the updated carry.

    Returns:
        The final carry, and whether the stack overflowed.
    """
    chunk = _chunk_size()
    # The stack holds at most one chunk's children per level of the tree.
    capacity = (2 * NUM_CELLS + 2) * NUM_CELLS * chunk
    stack = tuple(
        jnp.zeros(capacity + NUM_CELLS * chunk, jnp.int32).at[0].set(r)
        for r in root
    )

    def body(loop):
        stack, size, carry, max_size = loop
        start = jnp.maximum(size - chunk, 0)
        nodes = tuple(
            jax.lax.dynamic_slice(s, (start,), (chunk,)) for s in stack
        )
        valid = jnp.arange(chunk) < size - start
        children, position, num_children, carry = expand(nodes, valid, carry)
        stack = tuple(
            s.at[start + position].set(c, mode="drop", unique_indices=True)
            for s, c in zip(stack, children, strict=True)
        )
        size = start + num_children
        return stack, size, carry, jnp.maximum(max_size, size)

    loop = (stack, jnp.int32(1), carry, jnp.int32(1))
    _, _, carry, max_size = jax.lax.while_loop(lambda x: x[1] > 0, body, loop)
    return carry, max_size > capacity


def _check_stack(result):
    carry, overflowed = result
    if overflowed.any():
        raise RuntimeError("The traversal stack overflowed.")
    return carry


def _moves(game, win, board, valid, device, num_devices):
    """Returns the effect of every action at a chunk of nodes.

    Past depth `SPLIT_DEPTH`, a device only keeps its own subtrees, and only
    device 0 reports leaves above it.
    """
    seat = (board >> SEAT_BIT) & 1
    mine = (board >> (NUM_CELLS * seat)) & 511
    theirs = (board >> (NUM_CELLS * (1 - seat))) & 511
    discovered = (board >> (2 * NUM_CELLS)) & 511
    tried = mine | (theirs & discovered)
    depth = jax.lax.population_count(board & ((1 << SEAT_BIT) - 1))
    cell = 1 << jnp.arange(NUM_CELLS)
    legal = ((tried[:, None] & cell) == 0) & valid[:, None]
    placed = ((mine | theirs)[:, None] & cell) == 0
    won = placed & win[seat[:, None] * 512 + (mine[:, None] | cell)]
    full = placed & (jax.lax.population_count(mine | theirs)[:, None] == 8)
    s = seat[:, None]
    # A blocked try ends the turn only in the abrupt version.
    next_seat = jnp.where(placed | game.abrupt, 1 - s, s)
    child = (
        (board[:, None] & ~(1 << SEAT_BIT))
        | jnp.where(placed, cell << (NUM_CELLS * s), cell << (2 * NUM_CELLS))
        | (next_seat << SEAT_BIT)
    )
    owner = (
        (child.astype(jnp.uint32) * jnp.uint32(2654435761) >> 16) % num_devices
    ).astype(jnp.int32)
    mine_to_expand = (depth[:, None] + 1 != SPLIT_DEPTH) | (owner == device)
    keep = legal & ~won & ~full & mine_to_expand
    count = keep.sum(1)
    position = (jnp.cumsum(count) - count)[:, None] + jnp.cumsum(keep, 1) - keep
    position = jnp.where(keep, position, _DROP)
    shallow = depth[:, None] + 1 <= SPLIT_DEPTH
    won &= legal & (~shallow | (device == 0))
    return seat, mine, tried, placed, won, child, position, count.sum()


def _build(game, win, devices):
    """Returns which histories of each seat are decision points."""

    def expand(nodes, valid, seen, device):
        board, history0, history1 = nodes
        seat, _, tried, placed, _, child, position, num = _moves(
            game, win, board, valid, device, len(devices)
        )
        history = jnp.where(seat == 0, history0, history1)
        seen = seen.at[seat, jnp.where(valid, history, _DROP)].set(
            True, mode="drop"
        )
        new = _extend(
            history[:, None],
            tried[:, None],
            jnp.arange(NUM_CELLS),
            placed.astype(jnp.int32),
        )
        s = seat[:, None]
        children = (
            child,
            jnp.where(s == 0, new, history0[:, None]),
            jnp.where(s == 1, new, history1[:, None]),
        )
        return children, position, num, seen

    def build(device):
        root = (jnp.int32(0),) * 3
        seen = jnp.zeros((2, NUM_HISTORIES), bool)
        return _traverse(functools.partial(expand, device=device), root, seen)

    seen, overflowed = _on_devices(build, devices)()
    return seen.any(0), overflowed


def _gradient(game, win, devices):
    """Returns a function from plans to gradients, as `[infosets + 1, 9]`."""

    def expand(nodes, valid, gradient, device, plan, next_infoset):
        board, last0, last1 = nodes
        seat, mine, _, _, won, child, position, num = _moves(
            game, win, board, valid, device, len(devices)
        )
        mine_last = jnp.where(seat == 0, last0, last1)
        their_last = jnp.where(seat == 0, last1, last0)
        cell = mine_last % NUM_CELLS
        outcome = jnp.where(mine_last >= NUM_CELLS, (mine >> cell) & 1, 0)
        row = next_infoset.reshape(-1, 2)[mine_last, outcome] + 1
        # The mover wins 1 and the opponent loses 1; draws are worth 0.
        any_won = won.any(1)
        their_plan = plan.ravel()[their_last]
        gradient = gradient.at[jnp.where(any_won, row, _DROP)].add(
            jnp.where(won, their_plan[:, None], 0.0), mode="drop"
        )
        loss = jnp.where(won, plan[row], 0.0).sum(1)
        gradient = gradient.ravel().at[jnp.where(any_won, their_last, _DROP)]
        gradient = gradient.add(-loss, mode="drop").reshape(plan.shape)
        sequence = NUM_CELLS * row[:, None] + jnp.arange(NUM_CELLS)
        s = seat[:, None]
        children = (
            child,
            jnp.where(s == 0, sequence, last0[:, None]),
            jnp.where(s == 1, sequence, last1[:, None]),
        )
        return children, position, num, gradient

    def gradient(device, plan, next_infoset):
        root = (jnp.int32(0), jnp.int32(0), jnp.int32(1))
        return _traverse(
            functools.partial(
                expand, device=device, plan=plan, next_infoset=next_infoset
            ),
            root,
            jnp.zeros_like(plan),
        )

    run = _on_devices(gradient, devices)

    def total(plan, next_infoset):
        g, overflowed = run(plan, next_infoset)
        return g.sum(0), overflowed

    return total


def _on_devices(fn, devices):
    """Returns a function that runs `fn(device_index, *args)` on each device.

    Outputs are stacked across devices. A single device avoids `pmap`, which
    would copy the arguments.
    """
    if len(devices) == 1:
        single = jax.jit(functools.partial(fn, 0))
        return lambda *args: jax.tree.map(lambda x: x[None], single(*args))
    parallel = jax.pmap(
        lambda i, args: fn(i, *args), in_axes=(0, None), devices=devices
    )
    return lambda *args: parallel(jnp.arange(len(devices)), args)


def _extend(history, tried, cell, placed):
    """Returns the number of `history` followed by a try of `cell`."""
    k = jax.lax.population_count(tried)
    x = history - _OFFSET[k]
    rank, bits = x >> k, x & ((1 << k) - 1)
    digit = cell - jax.lax.population_count(tried & ((1 << cell) - 1))
    rank = rank * (NUM_CELLS - k) + digit
    return _OFFSET[k + 1] + (rank << (k + 1)) + (bits << 1) + placed


def _decode(history):
    """Returns the cells tried, in order, and whether each try placed a stone.

    Also returns the number of tries. Cells after the last try are -1.
    """
    k = jnp.searchsorted(_OFFSET, history, side="right").astype(jnp.int32) - 1
    x = history - _OFFSET[k]
    rank, bits = x >> k, x & ((1 << k) - 1)
    digits = []
    for i in reversed(range(NUM_CELLS)):
        digits.append(rank % (NUM_CELLS - i))
        rank = jnp.where(i < k, rank // (NUM_CELLS - i), rank)
    digits.reverse()
    cells, tried = [], jnp.zeros_like(history)
    for i in range(NUM_CELLS):
        # The try picks the digits[i]-th cell not tried yet.
        cell, count = jnp.full_like(history, -1), jnp.zeros_like(history)
        for c in range(NUM_CELLS):
            free = ((tried >> c) & 1) == 0
            cell = jnp.where(free & (count == digits[i]) & (cell < 0), c, cell)
            count += free
        cell = jnp.where(i < k, cell, -1)
        tried |= jnp.where(cell >= 0, 1 << cell, 0)
        cells.append(cell)
    shift = jnp.maximum(k[:, None] - 1 - jnp.arange(NUM_CELLS), 0)
    placed = (bits[:, None] >> shift) & 1
    return jnp.stack(cells, 1), placed, k


def _encode(cells, placed, k):
    """Returns the number of the history of the first `k` tries."""
    history, tried = jnp.zeros_like(k), jnp.zeros_like(k)
    for i in range(NUM_CELLS):
        active = i < k
        cell = jnp.maximum(cells[:, i], 0)
        history = jnp.where(
            active, _extend(history, tried, cell, placed[:, i]), history
        )
        tried = jnp.where(active, tried | (1 << cell), tried)
    return history


def _tried(cells):
    """Returns the tried cells as a bitmask."""
    return jnp.where(cells >= 0, 1 << jnp.maximum(cells, 0), 0).sum(1)


def _observe(game, history, seat):
    """Returns the observations of information sets, given their histories."""
    cells, placed, _ = _decode(history)
    # A try shows the seat's own stone if it placed one, else the opponent's.
    stone = jnp.where(placed == 1, seat[:, None], 1 - seat[:, None])
    rows = jnp.arange(cells.shape[0])[:, None]
    view = jnp.full(cells.shape, -1).at[
        rows, jnp.where(cells >= 0, cells, _DROP)
    ]
    view = view.set(stone, mode="drop")
    own = jnp.arange(2)[None, :, None] == seat[:, None, None]
    states = jax.tree.map(
        lambda x: jnp.repeat(x, cells.shape[0], 0), game.initial_states()[0]
    )
    states = dataclasses.replace(
        states,
        current_seat=seat,
        view=jnp.where(own, view[:, None], -1),
        history=jnp.where(own, cells[:, None], -1),
    )
    return jax.vmap(game.observe)(states, seat)

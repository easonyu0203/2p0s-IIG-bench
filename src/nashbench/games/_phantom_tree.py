"""The sequence form of 3x3 phantom games, by traversal on the accelerator.

These trees have about 10^10 histories, too many to store. Each evaluation
therefore traverses the whole tree depth-first inside one `lax.while_loop`,
visiting a chunk of nodes per iteration. A node is three int32s: the board
and one number per seat.
"""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from nashbench import sequence_form as sequence_form_lib

NUM_CELLS = 9
# Board bits: stones of seat 0, stones of seat 1, stones discovered by the
# opponent (9 bits each), then the seat to move.
SEAT_BIT = 3 * NUM_CELLS
# Out-of-bounds index for writes to drop.
_DROP = 2**30
# Information sets per chunk when building tables.
_TABLE_CHUNK = 2**22

# A history of `k` tries is numbered `OFFSET[k] + rank * 2**k + placed`: `rank`
# numbers the sequence of tried cells in mixed radix (9, 8, ...), and the bits
# of `placed` say which tries placed a stone. Decisions happen after at most 8
# tries.
OFFSET = np.cumsum([0] + [math.perm(9, k) * 2**k for k in range(9)])
NUM_HISTORIES = int(OFFSET[-1])
_OFFSET = jnp.asarray(OFFSET, jnp.int32)


def sequence_form(game) -> sequence_form_lib.SequenceForm:
    """Returns the sequence form of a phantom game."""
    # Whether a seat wins, at `512 * seat + stones` for stones as a bitmask.
    win = jnp.asarray(game._win_table.ravel())
    seen = _check_stack(_build(game, win))
    # Information sets are numbered by seat, then by history. A dense table
    # maps each seat's histories to their numbers, or -1.
    number = jnp.cumsum(seen.ravel(), dtype=jnp.int32).reshape(seen.shape) - 1
    number = jnp.where(seen, number, -1)
    seat, history = (x.astype(jnp.int32) for x in jnp.nonzero(seen))
    del seen
    # Tables are built in chunks of information sets to bound memory, padded
    # to compile once.
    chunks = (
        jnp.pad(x, (0, -x.size % _TABLE_CHUNK)).reshape(-1, _TABLE_CHUNK)
        for x in (history, seat)
    )
    tables = [_tables(number, h, s) for h, s in zip(*chunks, strict=True)]
    del number
    parent, legal, next_infoset, k = (
        jnp.concatenate(t)[: history.size] for t in zip(*tables, strict=True)
    )
    roots = (
        jnp.full((1, NUM_CELLS, 2), -1)
        .at[0, :2]
        .set(jnp.asarray([0, (seat == 0).sum()])[:, None])
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
    gradient = _gradient(game, win)
    return sequence_form_lib.SequenceForm(
        levels=tuple(levels),
        parent=parent,
        legal_action_mask=legal,
        observe=functools.partial(observe, history, seat),
        gradient=lambda plan: _check_stack(
            gradient(plan.reshape(-1, NUM_CELLS), next_infoset)
        ).ravel(),
    )


@jax.jit
def _tables(number, history, seat):
    """Returns tables of information sets, given their histories.

    The tables are the parent sequence, legal actions, the information set
    reached after each action and outcome, and the level.
    """

    def infoset_of(history):
        """Returns the information set of the seat with `history`, or -1."""
        seats = seat.reshape(-1, *[1] * (history.ndim - 1))
        return number.at[seats, history].get(mode="fill", fill_value=-1)

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


def _chunk_size():
    """Returns how many nodes to expand per iteration."""
    return 2**12 if jax.default_backend() == "cpu" else 2**19


def _traverse(visit, root, carry):
    """Traverses the tree depth-first from `root`.

    The stack holds nodes whose children are left to visit, with a bitmask of
    their actions. Each iteration visits the first child left of a chunk of
    nodes at the top, then pushes back each node with children left, and
    each child with children to visit. Pushing two entries per node, rather
    than one per action, keeps the pushes cheap.

    `visit(nodes, action, valid, carry)` returns the child of each node after
    `action`, the bitmask of the child's children to visit, and the updated
    carry. `root` is the root's fields and bitmask.

    Returns:
        The final carry, and whether the stack overflowed.
    """
    chunk = _chunk_size()
    # About a chunk per level of the tree, which has at most 17 tries. The
    # stack holds up to 8 chunks in these games.
    capacity = 2 * NUM_CELLS * chunk
    # The barrier keeps XLA from spending seconds at compile time folding the
    # stack into a constant.
    zeros = jax.lax.optimization_barrier(
        jnp.zeros(capacity + 2 * chunk, jnp.int32)
    )
    stack = tuple(zeros.at[0].set(r) for r in root)

    def body(loop):
        stack, size, carry, max_size = loop
        start = jnp.maximum(size - chunk, 0)
        *nodes, mask = (
            jax.lax.dynamic_slice(s, (start,), (chunk,)) for s in stack
        )
        valid = jnp.arange(chunk) < size - start
        action = jax.lax.population_count((mask & -mask) - 1)
        children, child_mask, carry = visit(tuple(nodes), action, valid, carry)
        # Each node pushes itself if it has children left, then its child.
        left = mask & (mask - 1)
        keep = (
            (jnp.stack([left, child_mask], 1) != 0) & valid[:, None]
        ).ravel()
        position = jnp.where(keep, start + jnp.cumsum(keep) - 1, _DROP)
        entries = zip((*nodes, left), (*children, child_mask), strict=True)
        # With jax_enable_x64, sums and some children are int64; the stack
        # stays int32.
        stack = tuple(
            s.at[position].set(
                jnp.stack(e, 1).ravel().astype(s.dtype),
                mode="drop",
                unique_indices=True,
            )
            for s, e in zip(stack, entries, strict=True)
        )
        size = start + keep.sum(dtype=jnp.int32)
        return stack, size, carry, jnp.maximum(max_size, size)

    loop = (stack, jnp.int32(1), carry, jnp.int32(1))
    _, _, carry, max_size = jax.lax.while_loop(lambda x: x[1] > 0, body, loop)
    return carry, max_size > capacity


def _check_stack(result):
    carry, overflowed = result
    if overflowed:
        raise RuntimeError("The traversal stack overflowed.")
    return carry


def _view(board):
    """Returns the seat to move, its stones, the opponent's, and its tries."""
    seat = (board >> SEAT_BIT) & 1
    mine = (board >> (NUM_CELLS * seat)) & 511
    theirs = (board >> (NUM_CELLS * (1 - seat))) & 511
    discovered = (board >> (2 * NUM_CELLS)) & 511
    return seat, mine, theirs, mine | (theirs & discovered)


def _play(game, board, action):
    """Returns the board after the seat to move tries `action`.

    Also returns whether the try placed a stone, and whether the turn passed.
    """
    seat, mine, theirs, _ = _view(board)
    cell = 1 << action
    placed = ((mine | theirs) & cell) == 0
    # A blocked try ends the turn only in the abrupt version.
    passes = placed | game.abrupt
    child = (
        (board & ~(1 << SEAT_BIT))
        | jnp.where(placed, cell << (NUM_CELLS * seat), cell << (2 * NUM_CELLS))
        | (jnp.where(passes, 1 - seat, seat) << SEAT_BIT)
    )
    return child, placed, passes


def _moves(game, win, board, valid):
    """Returns the effect of every action at a chunk of nodes.

    Returns the seat to move, its stones, the actions that win, and the
    bitmask of actions whose children to visit.
    """
    seat, mine, theirs, tried = _view(board)
    cell = 1 << jnp.arange(NUM_CELLS)
    legal = ((tried[:, None] & cell) == 0) & valid[:, None]
    placed = ((mine | theirs)[:, None] & cell) == 0
    won = legal & placed & win[seat[:, None] * 512 + (mine[:, None] | cell)]
    full = placed & (jax.lax.population_count(mine | theirs)[:, None] == 8)
    to_visit = legal & ~won & ~full
    return seat, mine, won, (to_visit * cell).sum(1)


def _build(game, win):
    """Returns which histories of each seat are decision points."""

    def visit(nodes, action, valid, seen):
        # A node holds the histories of the seat to move and of the other.
        board, history, other = nodes
        _, _, _, tried = _view(board)
        child, placed, passes = _play(game, board, action)
        new = _extend(history, tried, action, placed.astype(jnp.int32))
        history, other = (
            jnp.where(passes, other, new),
            jnp.where(passes, new, other),
        )
        seat, _, _, to_visit = _moves(game, win, child, valid)
        seen = seen.at[seat, jnp.where(valid, history, _DROP)].set(
            True, mode="drop"
        )
        return (child, history, other), to_visit, seen

    @jax.jit
    def build():
        root = (jnp.int32(0),) * 3 + (jnp.int32(511),)
        seen = jnp.zeros((2, NUM_HISTORIES), bool)
        return _traverse(visit, root, seen)

    seen, overflowed = build()
    # The root, no node's child, is seat 0's first decision point.
    return seen.at[0, 0].set(True), overflowed


def _gradient(game, win):
    """Returns a function from plans to gradients, as `[infosets + 1, 9]`."""

    def visit(nodes, action, valid, gradient, plan, next_infoset):
        # A node holds the row of the seat to move and the last sequence of
        # the other.
        board, row, their_last = nodes
        child, _, passes = _play(game, board, action)
        sequence = NUM_CELLS * row + action
        mine_last = jnp.where(passes, their_last, sequence)
        their_last = jnp.where(passes, sequence, their_last)
        _, mine, won, to_visit = _moves(game, win, child, valid)
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
        # Nearby nodes often share `their_last`, and atomic adds to one address
        # serialize within a GPU warp. Transposing spreads them across warps.
        index = jnp.where(any_won, their_last, _DROP).reshape(-1, 32).T.ravel()
        gradient = gradient.ravel().at[index]
        gradient = gradient.add(-loss.reshape(-1, 32).T.ravel(), mode="drop")
        gradient = gradient.reshape(plan.shape)
        return (child, row, their_last), to_visit, gradient

    @jax.jit
    def gradient(plan, next_infoset):
        # Seat 0 moves first, at its first information set (row 1); sequence 1
        # is the empty sequence of seat 1.
        root = (jnp.int32(0), jnp.int32(1), jnp.int32(1), jnp.int32(511))
        return _traverse(
            functools.partial(visit, plan=plan, next_infoset=next_infoset),
            root,
            jnp.zeros_like(plan),
        )

    return gradient


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
    # An elementwise sum, not a reduction, which XLA compiles slowly.
    k = sum((history >= o for o in _OFFSET[1:]), jnp.zeros_like(history))
    x = history - _OFFSET[k]
    rank, bits = x >> k, x & ((1 << k) - 1)
    digits = []
    for i in reversed(range(NUM_CELLS)):
        digits.append(rank % (NUM_CELLS - i))
        rank = jnp.where(i < k, rank // (NUM_CELLS - i), rank)
    digits.reverse()
    cells, tried = [], jnp.zeros_like(history)
    c = jnp.arange(NUM_CELLS)
    for i in range(NUM_CELLS):
        # The try picks the digits[i]-th cell not tried yet.
        free = ((tried[:, None] >> c) & 1) == 0
        before = jax.lax.population_count(~tried[:, None] & ((1 << c) - 1))
        cell = jnp.argmax(free & (before == digits[i][:, None]), 1)
        cell = jnp.where(i < k, cell, -1)
        tried |= jnp.where(cell >= 0, 1 << cell, 0)
        cells.append(cell)
    shift = jnp.maximum(k[:, None] - 1 - jnp.arange(NUM_CELLS), 0)
    placed = (bits[:, None] >> shift) & 1
    return jnp.stack(cells, 1), placed, k


def _encode(cells, placed, k):
    """Returns the number of the history of the first `k` tries."""
    # The barrier keeps XLA from folding the first tries into large constants
    # at compile time.
    history = tried = jax.lax.optimization_barrier(jnp.zeros_like(k))
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


def _observe(game, history, seat, ids):
    """Returns the observations of information sets `ids`.

    `history` and `seat` are those of every information set.
    """
    history, seat = history[ids], seat[ids]
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

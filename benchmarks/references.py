"""Independent evaluators that validate.py compares with nashbench.

Each lists, per seat, the observation that nashbench gives at each of its
information sets, packed into bits, and their legal actions. `match` pairs
them with nashbench's, so that both evaluators play the same probabilities.
"""

import copy
import importlib

import common
import jax.numpy as jnp
import numpy as np
from open_spiel.python import policy as openspiel_policy
from open_spiel.python.algorithms import policy_aggregator
import pyspiel

from nashbench.games import goofspiel
from nashbench.games import oshi_zumo


class OpenSpiel:
    """OpenSpiel's best responses and expected returns, in C++."""

    def __init__(self, name, game):
        """Lists the OpenSpiel game's information states."""
        self.game = pyspiel.load_game(_openspiel_string(name, game))
        if (
            self.game.get_type().dynamics
            == pyspiel.GameType.Dynamics.SIMULTANEOUS
        ):
            self.game = pyspiel.convert_to_turn_based(self.game)
        self.policy = openspiel_policy.TabularPolicy(self.game)
        states = self.policy.states
        self.rows = [
            np.flatnonzero([s.current_player() == p for s in states])
            for p in range(2)
        ]
        self.keys = [
            _pack([_observation(game, states[i], p) for i in rows])
            for p, rows in enumerate(self.rows)
        ]
        actions = np.arange(game.num_actions)
        self.legal = [
            np.array(
                [np.isin(actions, states[i].legal_actions()) for i in rows]
            )
            for rows in self.rows
        ]

    def evaluate(self, members):
        """Returns each seat's best-response and profile values.

        Args:
            members: `[(tables, weights)]`: each seat's probabilities, in
                this evaluator's order, and the member's weight per seat.
        """
        seats = [[], []]
        for tables, weights in members:
            member = copy.copy(self.policy)
            member.action_probability_array = np.zeros_like(
                self.policy.action_probability_array
            )
            for seat in range(2):
                member.action_probability_array[self.rows[seat]] = tables[seat]
                if weights[seat] > 0:
                    seats[seat].append((member, weights[seat]))
        if len(members) == 1:
            policy = seats[0][0][0]
        else:
            policy = policy_aggregator.PolicyAggregator(self.game).aggregate(
                [0, 1],
                [[p for p, _ in seat] for seat in seats],
                [[w for _, w in seat] for seat in seats],
            )
        table = {}
        for state in self.policy.states:
            probs = policy.action_probabilities(state)
            table[state.information_state_string()] = [
                (a, probs.get(a, 0.0)) for a in state.legal_actions()
            ]
        root = self.game.new_initial_state()
        best = [
            pyspiel.TabularBestResponse(self.game, p, table).value_from_state(
                root
            )
            for p in range(2)
        ]
        profile = pyspiel.expected_returns(
            root, pyspiel.TabularPolicy(table), -1, True
        )
        return tuple(best), tuple(profile)


def _openspiel_string(name, game):
    if name == "goofspiel":
        return f"goofspiel(players=2,num_cards={game.num_cards},imp_info=true)"
    if name == "liars_dice":
        return (
            f"liars_dice(numdice={game.numdice},dice_sides={game.dice_sides})"
        )
    if name == "oshi_zumo":
        return f"oshi_zumo(coins={game.coins},size={game.size},min_bid=1)"
    return name


def _observation(game, state, player):
    """Returns nashbench's observation of an OpenSpiel state."""
    if isinstance(game, oshi_zumo.OshiZumo):
        # nashbench appends the bids of finished turns to the observation.
        history = state.history()
        bids = np.full((game.coins, 2), -1)
        bids[: len(history) // 2] = np.reshape(
            history[: len(history) // 2 * 2], (-1, 2)
        )
        one_hot = bids.T[..., None] == np.arange(game.coins + 1)
        tensor = np.concatenate(
            [state.observation_tensor(player)[2:], one_hot.ravel()]
        )
    else:
        tensor = np.asarray(state.information_state_tensor(player))
        if isinstance(game, goofspiel.Goofspiel):
            tensor = tensor[2:]  # The turn-based game's who plays and observes.
    if tensor.size < game.observation_shape[0]:
        tensor = np.concatenate([np.eye(2)[player], tensor])
    return tensor


# exp-a-spiel's traverser of each game.
TRAVERSERS = {
    "phantom_ttt": "PtttTraverser",
    "phantom_ttt_abrupt": "AbruptPtttTraverser",
    "dark_hex3": "DhTraverser",
    "dark_hex3_abrupt": "AbruptDhTraverser",
}


class ExpASpiel:
    """exp-a-spiel's traversers of the phantom games, in C++."""

    def __init__(self, name, game):
        """Builds the traverser and lists its information sets."""
        del game  # Unused.
        self.eas = importlib.import_module("eas")
        self.traverser = getattr(self.eas, TRAVERSERS[name])()
        self.keys = []
        for p in range(2):
            x = self.traverser.compute_openspiel_infostates(p)
            seat = np.zeros((len(x), 2), bool)
            seat[:, p] = True
            self.keys.append(_pack(np.concatenate([seat, x], 1)))
        self.legal = [
            s > 0 for s in self.traverser.construct_uniform_strategies()
        ]

    def evaluate(self, members):
        """Returns each seat's best-response and profile values.

        Args:
            members: As in `OpenSpiel.evaluate`.
        """
        strategies = []
        for seat in range(2):
            active = [(t[seat], w[seat]) for t, w in members if w[seat] > 0]
            if len(active) == 1:
                strategies.append(active[0][0].astype(np.float64))
                continue
            averager = self.traverser.new_averager(
                seat, self.eas.AveragingStrategy.CUSTOM
            )
            for table, weight in active:
                averager.push(table.astype(np.float64), weight)
            strategies.append(np.ascontiguousarray(averager.running_avg()))
        ev = self.traverser.ev_and_exploitability(*strategies)
        # expl[i] is how much seat 1 - i gains by a best response.
        return (ev.expl[1] + ev.ev0, ev.expl[0] - ev.ev0), (ev.ev0, -ev.ev0)


def match(game, reference):
    """Returns, per seat, nashbench's information set at each reference one.

    Raises:
        ValueError: If the observations don't pair one to one, or legal
            actions differ.
    """
    form = game.sequence_form
    n = form.parent.size
    ours = np.concatenate(
        [
            _pack(form.observe(jnp.arange(i, min(i + common.CHUNK_SIZE, n))))
            for i in range(0, n, common.CHUNK_SIZE)
        ]
    )
    our_order, our_words = _sort(ours)
    their_order, their_words = _sort(np.concatenate(reference.keys))
    a, b = our_words[our_order], their_words[their_order]
    if a.shape != b.shape or (a != b).any() or (a[1:] == a[:-1]).all(1).any():
        raise ValueError("The observations don't pair one to one.")
    rows = np.empty(n, int)
    rows[their_order] = our_order
    rows = np.split(rows, [len(reference.keys[0])])
    legal = np.asarray(form.legal_action_mask)
    for seat in range(2):
        if (legal[rows[seat]] != reference.legal[seat]).any():
            raise ValueError(f"Seat {seat}'s legal actions differ.")
    return rows


def _pack(observations):
    """Packs binary observations into bits."""
    x = np.asarray(observations)
    if ((x != 0) & (x != 1)).any():
        raise ValueError("Observations aren't binary.")
    return np.packbits(x.astype(bool), axis=1)


def _sort(keys):
    """Returns the order of rows of bytes, and the rows as words."""
    words = np.pad(keys, ((0, 0), (0, -keys.shape[1] % 8))).view(">u8")
    return np.lexsort(words.T[::-1]), words

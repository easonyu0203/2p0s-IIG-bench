# API reference

The top-level `nashbench` package exports the public API, for example
`nashbench.Game`. The sections below document each name in the module that
defines it.

## Games

```{eval-rst}
.. autofunction:: nashbench.make

.. autodata:: nashbench.games.REGISTRY
   :no-value:

.. automodule:: nashbench.core
   :members:
```

## Policies

```{eval-rst}
.. automodule:: nashbench.policy
   :members:
   :special-members: __call__
```

## Evaluation

```{eval-rst}
.. automodule:: nashbench.exploitability
   :members:
```

## Sequence form

To make `exploitability` fast on a large game, override `Game.sequence_form`.

```{eval-rst}
.. automodule:: nashbench.sequence_form
   :members:
```

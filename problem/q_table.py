from __future__ import annotations

from dataclasses import dataclass, field

from problem.tabular_state import TabularState


@dataclass
class QTable:
    """
    Sparse tabular representation of an action-value function.

    The table maps pairs of discrete states and actions to scalar
    Q-values. Internally, values are stored in a dictionary keyed by
    ``(state, action)`` tuples.

    State-action pairs that have not yet been stored return
    ``default_value`` when queried. This allows the table to represent
    only the portion of the state-action space that has been encountered
    during an experiment.

    Parameters
    ----------
    default_value:
        Initial Q-value returned for state-action pairs that are not
        explicitly stored. Defaults to ``0.0``.
    """

    default_value: float = 0.0

    _values: dict[tuple[TabularState, int], float] = field(
        default_factory=dict
    )

    def get(
        self,
        state: TabularState,
        action: int,
    ) -> float:
        """
        Return the Q-value associated with a state-action pair.

        If the pair has not previously been stored, ``default_value``
        is returned.

        Parameters
        ----------
        state:
            Discrete state used as part of the table key.

        action:
            Action used as part of the table key.

        Returns
        -------
        float
            Stored Q-value, or ``default_value`` when the state-action
            pair has not yet been encountered.
        """

        return self._values.get(
            (state, action),
            self.default_value,
        )

    def set(
        self,
        state: TabularState,
        action: int,
        value: float,
    ) -> None:
        """
        Store a Q-value for a state-action pair.

        Parameters
        ----------
        state:
            Discrete state used as part of the table key.

        action:
            Action used as part of the table key.

        value:
            New Q-value. The value is converted to ``float`` before
            storage.
        """

        self._values[(state, action)] = float(value)

    def values(
        self,
        state: TabularState,
        actions: tuple[int, ...],
    ) -> tuple[float, ...]:
        """
        Return Q-values for a state in the specified action order.

        Parameters
        ----------
        state:
            Discrete state for which Q-values are requested.

        actions:
            Actions whose Q-values should be returned. The order of
            this tuple determines the order of the returned values.

        Returns
        -------
        tuple[float, ...]
            Q-values corresponding to ``actions`` in the same order.
        """

        return tuple(
            self.get(state, action)
            for action in actions
        )

    def snapshot(
        self,
    ) -> dict[tuple[TabularState, int], float]:
        """
        Return a copy of the currently stored Q-values.

        The returned dictionary can be serialized as part of a
        checkpoint without exposing the table's internal dictionary
        to modification.

        Returns
        -------
        dict[tuple[TabularState, int], float]
            Copy of all explicitly stored state-action Q-values.
        """

        return dict(self._values)

    def restore(
        self,
        values: dict[tuple[TabularState, int], float],
    ) -> None:
        """
        Replace the current table with a restored set of Q-values.

        Parameters
        ----------
        values:
            Mapping of ``(state, action)`` pairs to their stored
            Q-values. Values are converted to ``float`` during
            restoration.
        """

        self._values = {
            (state, action): float(value)
            for (state, action), value in values.items()
        }

    def contains(
        self,
        state: TabularState,
        action: int,
    ) -> bool:
        return (state, action) in self._values

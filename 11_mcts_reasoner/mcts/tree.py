"""
MCTS Tree Data Structures for Game of 24
"""

from __future__ import annotations
import math
from fractions import Fraction
from typing import Any
from mcts_reasoner.solver import get_possible_moves, Step


class Node:
    """
    Represents a state node in the Monte Carlo Tree.
    State is the remaining pool of numbers in the Game of 24.
    """

    def __init__(
        self,
        state: tuple[Fraction, ...],
        history: list[str] | None = None,
        parent: Node | None = None,
        action: str | None = None,
        prior_prob: float = 1.0,
    ):
        self.state = tuple(sorted(state))
        self.history = list(history or [])
        self.parent = parent
        self.action = action  # The step string that produced this node
        self.prior_prob = prior_prob

        self.children: list[Node] = []
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prm_score: float | None = None
        self.critique: str | None = None
        self.is_pruned: bool = False

        # Track untried legal moves from this state
        self.untried_moves: list[Step] = get_possible_moves(self.state)

    @property
    def q_value(self) -> float:
        """Mean action-value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_terminal(self) -> bool:
        """Terminal if only 1 number remains."""
        return len(self.state) == 1

    @property
    def is_target_reached(self) -> bool:
        """True if terminal and remaining number is 24."""
        return self.is_terminal and self.state[0] == Fraction(24, 1)

    def is_fully_expanded(self) -> bool:
        """True if all candidate moves have been converted to child nodes."""
        return len(self.untried_moves) == 0 and len(self.children) > 0

    def has_untried_moves(self) -> bool:
        return len(self.untried_moves) > 0

    def uct_score(self, c_puct: float = 1.414) -> float:
        """
        PUCT formula:
        UCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        """
        if self.is_pruned:
            return -999.0

        parent_n = self.parent.visit_count if self.parent else 1
        exploration = c_puct * self.prior_prob * (math.sqrt(parent_n) / (1 + self.visit_count))
        return self.q_value + exploration

    def select_best_child(self, c_puct: float = 1.414) -> Node | None:
        """Select child with highest UCT score among unpruned children."""
        active_children = [c for c in self.children if not c.is_pruned]
        if not active_children:
            return None
        return max(active_children, key=lambda c: c.uct_score(c_puct))

    def update_pruned_status(self):
        """If all children are pruned and no untried moves remain, this node is dead."""
        if not self.untried_moves and self.children and all(c.is_pruned for c in self.children):
            self.is_pruned = True
            if self.parent:
                self.parent.update_pruned_status()

    def to_dict(self) -> dict[str, Any]:
        """Serializes node and descendants for UI visualization."""
        rem_str = [str(x) for x in self.state]
        return {
            "action": self.action or "Root",
            "state": rem_str,
            "history": self.history,
            "visits": self.visit_count,
            "q_value": round(self.q_value, 3),
            "prm_score": round(self.prm_score, 3) if self.prm_score is not None else None,
            "critique": self.critique,
            "is_pruned": self.is_pruned,
            "is_target": self.is_target_reached,
            "is_terminal": self.is_terminal,
            "children": [child.to_dict() for child in self.children]
        }

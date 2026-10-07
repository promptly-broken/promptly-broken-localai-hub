"""
Monte Carlo Tree Search (MCTS) Engine with PRM-Guided Backtracking
"""

from __future__ import annotations
import time
from fractions import Fraction
from typing import NamedTuple

from mcts.tree import Node
from mcts.generator import StepGenerator
from mcts.verifier import PRMVerifier
from solver import verify_solution


class SearchResult(NamedTuple):
    success: bool
    solution_steps: list[str]
    total_nodes: int
    pruned_nodes: int
    simulations_run: int
    root: Node
    execution_time_sec: float
    verification_message: str
    verifier_calls: int = 0
    cache_hits: int = 0
    model_ms: float = 0.0


class MCTSSearchEngine:
    """
    Orchestrates MCTS guided by the Process Reward Model.
    Supports hard pruning (for exact symbolic oracle) and soft pruning with
    branch resurrection (for learned PRMs to prevent premature search collapse).
    """

    def __init__(
        self,
        generator: StepGenerator | None = None,
        verifier: PRMVerifier | None = None,
        c_puct: float = 1.414,
        max_simulations: int = 100,
        k_branching: int = 4,
        prune_mode: str | None = None,  # "hard" | "soft"
    ):
        self.generator = generator or StepGenerator()
        self.verifier = verifier or PRMVerifier()
        self.c_puct = c_puct
        self.max_simulations = max_simulations
        self.k_branching = k_branching

        # Default prune_mode: soft for learned models, hard for symbolic oracle
        if prune_mode is not None:
            self.prune_mode = prune_mode
        else:
            self.prune_mode = "soft" if self.verifier.mode in ["mlx", "mlx_gen", "ollama", "hybrid"] else "hard"

    def solve(
        self,
        initial_numbers: list[int],
        target: int = 24
    ) -> SearchResult:
        """
        Executes MCTS to solve the puzzle.
        Returns detailed SearchResult with root tree for visual inspection.
        """
        start_time = time.time()
        init_fractions = tuple(Fraction(x, 1) for x in initial_numbers)
        root = Node(state=init_fractions, history=[], action="Initial State")

        winning_node: Node | None = None
        is_soft = (self.prune_mode == "soft")

        for sim in range(1, self.max_simulations + 1):
            if is_soft and root.is_pruned:
                root.update_pruned_status(soft=True)
            elif not is_soft and root.is_pruned:
                break

            # 1. SELECTION: Traverse tree until reaching a node with untried moves or an unexpanded leaf
            node = root
            while not node.is_terminal and not node.is_pruned:
                if node.has_untried_moves():
                    # Stop here to expand untried moves
                    break

                best_child = node.select_best_child(self.c_puct)
                if best_child is None:
                    # All children pruned or no children
                    node.update_pruned_status(soft=is_soft)
                    best_child = node.select_best_child(self.c_puct)
                    if best_child is None:
                        break
                node = best_child

            # 2. EXPANSION & EVALUATION
            if not node.is_terminal and not node.is_pruned and node.has_untried_moves():
                candidates = self.generator.pop_untried_candidates(node, k=self.k_branching)

                for cand in candidates:
                    child_history = node.history + [cand.step_str]
                    child = Node(
                        state=cand.remaining,
                        history=child_history,
                        parent=node,
                        action=cand.step_str,
                        prior_prob=cand.prior_prob
                    )

                    # Evaluate candidate step with PRM
                    verdict = self.verifier.evaluate_step(
                        initial_numbers=initial_numbers,
                        history=node.history,
                        candidate_step=cand.step_str,
                        remaining_state=child.state
                    )

                    child.prm_score = verdict.score
                    child.critique = verdict.critique
                    child.is_pruned = verdict.is_pruned

                    node.children.append(child)

                    if child.is_target_reached:
                        winning_node = child
                        break

                node.update_pruned_status(soft=is_soft)

            if winning_node:
                break

            # 3. BACKPROPAGATION: Update visit counts and Q-values up to root
            if node.children:
                active_scores = [c.prm_score for c in node.children if not c.is_pruned and c.prm_score is not None]
                if active_scores:
                    leaf_value = max(active_scores)
                else:
                    leaf_value = max((c.prm_score for c in node.children if c.prm_score is not None), default=0.0)
            else:
                leaf_value = node.prm_score if node.prm_score is not None else 0.0

            curr: Node | None = node
            while curr is not None:
                curr.visit_count += 1
                curr.total_value += leaf_value
                curr = curr.parent

        elapsed = time.time() - start_time
        total_nodes = self._count_nodes(root)
        pruned_nodes = self._count_pruned(root)

        v_calls = getattr(self.verifier, "calls", 0)
        v_hits = getattr(self.verifier, "cache_hits", 0)
        v_ms = getattr(self.verifier, "model_ms", 0.0)

        if winning_node:
            valid, msg = verify_solution(initial_numbers, winning_node.history)
            return SearchResult(
                success=valid,
                solution_steps=winning_node.history,
                total_nodes=total_nodes,
                pruned_nodes=pruned_nodes,
                simulations_run=sim,
                root=root,
                execution_time_sec=elapsed,
                verification_message=msg,
                verifier_calls=v_calls,
                cache_hits=v_hits,
                model_ms=v_ms,
            )
        else:
            return SearchResult(
                success=False,
                solution_steps=[],
                total_nodes=total_nodes,
                pruned_nodes=pruned_nodes,
                simulations_run=min(sim, self.max_simulations),
                root=root,
                execution_time_sec=elapsed,
                verification_message="Exhausted search budget without confirmed solution",
                verifier_calls=v_calls,
                cache_hits=v_hits,
                model_ms=v_ms,
            )

    def _count_nodes(self, node: Node) -> int:
        return 1 + sum(self._count_nodes(c) for c in node.children)

    def _count_pruned(self, node: Node) -> int:
        count = 1 if node.is_pruned else 0
        return count + sum(self._count_pruned(c) for c in node.children)

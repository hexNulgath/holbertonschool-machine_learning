#!/usr/bin/env python3
"""
Isolation Forest Implementation

This module implements an Isolation Forest, an unsupervised anomaly detection
algorithm that works by isolating anomalies instead of profiling normal points.
It uses a collection of Isolation Random Trees to identify anomalies in data.
"""

from __future__ import annotations
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest:
    """Isolation Random Forest for anomaly detection.

    This class implements an isolation forest algorithm which detects anomalies
    by isolating them in the feature space using
    an ensemble of isolation trees.

    Attributes:
        numpy_predicts (list): List of numpy prediction functions
        target: Target variable (unused in current implementation)
        numpy_preds (list): List of numpy prediction functions from trees
        n_trees (int): Number of trees in the forest
        max_depth (int): Maximum depth for each tree
        seed (int): Random seed for reproducibility
    """

    def __init__(self, n_trees: int = 100, max_depth: int = 10,
                 min_pop: int = 1, seed: int = 0) -> None:
        """Initialize the Isolation Random Forest.

        Args:
            n_trees (int): Number of trees in the forest. Defaults to 100.
            max_depth (int): Maximum depth of each tree. Defaults to 10.
            min_pop (int): Minimum population for node splitting.
            Defaults to 1.
            seed (int): Random seed for reproducibility. Defaults to 0.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory: np.ndarray) -> np.ndarray:
        """Predict anomaly scores for the input data.

        Args:
            explanatory (np.ndarray): Input data to predict scores for

        Returns:
            np.ndarray: Anomaly scores for
            each input sample (mean across trees)
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory: np.ndarray, n_trees: int = 100,
            verbose: int = 0) -> None:
        """Fit the isolation forest to the training data.

        Args:
            explanatory (np.ndarray): Training data
            n_trees (int): Number of trees to grow. Defaults to 100.
            verbose (int): Verbosity level (0 or 1). Defaults to 0.
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []

        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))

        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory: np.ndarray, n_suspects: int) -> tuple:
        """Identify the most anomalous samples.

        Args:
            explanatory (np.ndarray): Input data to evaluate
            n_suspects (int): Number of anomalies to return

        Returns:
            tuple: (suspect_samples, suspect_scores) where:
                - suspect_samples: The n_suspects most anomalous samples
                - suspect_scores: Their corresponding anomaly scores
        """
        depths = self.predict(explanatory)
        indices = np.argsort(depths)[:n_suspects]
        return explanatory[indices], depths[indices]

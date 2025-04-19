#!/usr/bin/env python3
"""
Isolation Random Tree Implementation

This module implements the Isolation
Random Tree algorithm for anomaly detection.
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """Isolation Random Tree for anomaly detection.

    Attributes:
        rng (np.random.Generator): Random number generator
        root (Node): Root node of the tree
        explanatory (np.ndarray): Training data features
        max_depth (int): Maximum tree depth
        predict (function): Prediction function
        min_pop (int): Minimum population for splitting

    Examples:
        >>> np.random.seed(0)
        >>> X = np.random.rand(100, 2)
        >>> tree = Isolation_Random_Tree(max_depth=5, seed=0)
        >>> tree.fit(X)
        >>> tree.depth() <= 5
        True
        >>> tree.count_nodes() > 1
        True
        >>> scores = tree.predict(X)
        >>> len(scores) == 100
        True
    """

    def __init__(self, max_depth=10, seed=0, root=None):
        """Initialize Isolation Random Tree.

        Args:
            max_depth (int): Maximum tree depth
            seed (int): Random seed
            root (Node): Optional root node
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Return string representation of the tree.

        Returns:
            str: String representation of the tree
        """
        return self.root.__str__()

    def depth(self):
        """Calculate maximum depth of the tree.

        Returns:
            int: Maximum depth
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes in the tree.

        Args:
            only_leaves (bool): Whether to count only leaves

        Returns:
            int: Number of nodes
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Update bounds for all nodes."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Get all leaves in the tree.

        Returns:
            list: List of Leaf objects
        """
        return self.root.get_leaves_below()

    def update_predict(self):
        """Update prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        
        # Fix: Return exactly one prediction per sample
        def predict_func(A):
            results = np.zeros(len(A))
            for i in range(len(A)):
                for leaf in leaves:
                    if leaf.indicator(A[i]):
                        results[i] = leaf.pred(A[i])
                        break  # Only take first matching leaf
            return results
        
        self.predict = predict_func

    def np_extrema(self, arr):
        """Calculate min and max of array.

        Args:
            arr (np.ndarray): Input array

        Returns:
            tuple: (min, max) values
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Generate random split criterion.

        Args:
            node (Node): Current node

        Returns:
            tuple: (feature index, threshold)
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
                )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Create leaf child node.

        Args:
            node (Node): Parent node
            sub_population (np.ndarray): Boolean mask for samples

        Returns:
            Leaf: New leaf node
        """

        leaf_child = Leaf(node.depth)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create internal child node.

        Args:
            node (Node): Parent node
            sub_population (np.ndarray): Boolean mask for samples

        Returns:
            Node: New internal node
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively fit tree starting from node.

        Args:
            node (Node): Current node
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        feature_values = self.explanatory[node.sub_population, node.feature]
        left_population = node.sub_population.copy()
        left_population[node.sub_population] = feature_values > node.threshold
        right_population = node.sub_population.copy()
        right_population[
            node.sub_population] = feature_values <= node.threshold

        is_left_leaf = (node.depth + 1 >= self.max_depth or
                        np.sum(left_population) <= self.min_pop)
        is_right_leaf = (node.depth + 1 >= self.max_depth or
                         np.sum(right_population) <= self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Fit the tree to data.

        Args:
            explanatory (np.ndarray): Training data
            verbose (int): Verbosity level
        """
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype=bool)

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")

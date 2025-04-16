#!/usr/bin/env python3
"""
Decision Tree Classifier Module

This module implements a decision tree classifier for classification tasks.
It includes classes for tree nodes, leaf nodes, and the decision tree itself.
"""

import numpy as np


class Node:
    """
    A node in the decision tree.

    Attributes:
        feature (int): Index of the feature used for splitting at this node.
        threshold (float): Threshold value used for splitting.
        left_child (Node): Left child node (values <= threshold).
        right_child (Node): Right child node (values > threshold).
        is_leaf (bool): Whether this node is a leaf node.
        is_root (bool): Whether this node is the root of the tree.
        sub_population (numpy.ndarray): Subset of data at this node.
        depth (int): Depth of this node in the tree (0 for root).

    Methods:
        max_depth_below(): Calculates the maximum depth
        of the tree below this node.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """
        Initialize a Node instance.

        Args:
            feature (int, optional): Feature index for splitting
            Defaults to None.
            threshold (float, optional): Threshold value for splitting.
            Defaults to None.
            left_child (Node, optional): Left child node. Defaults to None.
            right_child (Node, optional): Right child node. Defaults to None.
            is_root (bool, optional): Whether this is the root node.
            Defaults to False.
            depth (int, optional): Depth of this node. Defaults to 0.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Calculate the maximum depth of the tree below this node.

        Returns:
            int: Maximum depth below this node.
        """
        if self.is_leaf:
            return self.depth
        else:
            if self.left_child:
                left_depth = self.left_child.max_depth_below()
            else:
                left_depth = 0
            if self.right_child:
                right_depth = self.right_child.max_depth_below()
            else:
                right_depth = 0
            # Return the maximum depth of the left and right children
            return (left_depth if left_depth > right_depth else right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes below this node.

        Args:
            only_leaves (bool, optional): If True, count only leaf nodes.
            Defaults to False.

        Returns:
            int: Number of nodes below this node.
        """
        if self.is_leaf:
            return 1
        else:
            left_count = self.left_child.count_nodes_below(only_leaves)
            right_count = self.right_child.count_nodes_below(only_leaves)
            if only_leaves:
                return left_count + right_count
            return 1 + left_count + right_count
        # Count this node itself

    def left_child_add_prefix(self, text):
        """
        Formats the left child subtree with
        exact indentation matching desired output.
        """
        if not text:
            return ""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x.strip():
                new_text += ("    |  "+x)+"\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Formats the right child subtree with
        exact indentation matching desired output.
        """
        if not text:
            return ""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x.strip():
                new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """
        Generates string representation that
        exactly matches desired output format.
        """
        if self.is_leaf:
            return f"leaf [value={self.value}]"

        # Format node information
        if self.is_root:
            node_info = f"root [feature={self.feature}, " + \
                        f"threshold={self.threshold}]\n"
        else:
            node_info = f"-> node [feature={self.feature}, " + \
                        f"threshold={self.threshold}]\n"

        # Handle children
        left_str = str(self.left_child) if self.left_child else ""
        right_str = str(self.right_child) if self.right_child else ""

        if left_str:
            node_info += self.left_child_add_prefix(left_str)
        if right_str:
            node_info += self.right_child_add_prefix(right_str)

        return node_info


class Leaf(Node):
    """
    A leaf node in the decision tree (terminal node).

    Attributes:
        value: The predicted value/class for this leaf.
        is_leaf (bool): Always True for leaf nodes.
        depth (int): Depth of this leaf in the tree.

    Methods:
        max_depth_below(): Returns the depth of this leaf node.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf instance.

        Args:
            value: The predicted value/class for this leaf.
            depth (int, optional): Depth of this leaf. Defaults to None.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of this leaf node.

        Returns:
            int: Depth of this leaf node.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes below this leaf node.
        Args:
            only_leaves (bool, optional): If True, count only leaf nodes.
            Defaults to False.
        Returns:
            int: Number of nodes below this leaf node.
        """
        return 1

    def __str__(self):
        """
        String representation of the leaf node.
        Returns:
            str: String representation of the leaf node.
        """
        return (f"-> leaf [value={self.value}]")


class Decision_Tree():
    """
    Decision Tree Classifier.

    Attributes:
        root (Node): Root node of the decision tree.
        explanatory (numpy.ndarray): Feature matrix for training.
        target (numpy.ndarray): Target values for training.
        max_depth (int): Maximum depth of the tree.
        min_pop (int): Minimum population required to split a node.
        split_criterion (str): Criterion used for splitting nodes.
        predict: Placeholder for prediction method (to be implemented).
        rng (numpy.random.Generator): Random number generator.

    Methods:
        depth(): Returns the maximum depth of the tree.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialize a Decision_Tree instance.

        Args:
            max_depth (int, optional): Maximum tree depth. Defaults to 10.
            min_pop (int, optional): Minimum samples to split. Defaults to 1.
            seed (int, optional): Random seed. Defaults to 0.
            split_criterion (str, optional): Splitting criterion.
            Defaults to "random".
            root (Node, optional): Root node if provided. Defaults to None.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Calculate the maximum depth of the decision tree.

        Returns:
            int: Maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count the number of nodes in the tree.
        Args:
            only_leaves (bool, optional): If True, count only leaf nodes.
            Defaults to False.
        Returns:
            int: Number of nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        String representation of the decision tree.
        Returns:
            str: String representation of the tree.
        """
        return self.root.__str__()

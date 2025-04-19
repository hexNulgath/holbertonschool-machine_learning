#!/usr/bin/env python3
"""Random Forest Classifier
This module implements a Random Forest classifier using decision trees.
It includes methods for fitting the model to training data,
predicting on new data, and calculating accuracy.
"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """ Random Forest Classifier.
    This class implements a Random Forest classifier using decision trees.
    Attributes:
        n_trees (int): Number of trees in the forest.
        max_depth (int): Maximum depth of each tree.
        min_pop (int): Minimum number of samples required to split a node.
        seed (int): Random seed for reproducibility.
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """ Initialize the Random Forest with the given parameters."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """ Predict the target variable for the given explanatory variables."""
        preds = np.zeros((self.n_trees, explanatory.shape[0]))
        for i in range(self.n_trees):
            preds[i] = self.numpy_preds[i](explanatory)
        # Return the most common prediction for each sample
        return np.array([
            np.bincount(
                preds[:, i].astype(int)
                ).argmax() for i in range(explanatory.shape[0])])

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """ Fit the random forest to the training data."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {
                        self.accuracy(self.explanatory, self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """ Calculate the accuracy of the model on the test data."""
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target)) / test_target.size

# END class Random_Forest

from TreeNode import TreeNode
import numpy as np
from typing import List


class ID3Classifier():
    """
    Custom implementation of the ID3 Decision Tree algorithm using Information Gain.
    """

    def __init__(self):
        """
        Initializes the ID3 classifier.
        """
        self._root = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ID3Classifier':
        """
        Builds the decision tree from the training data.
        """
        self._root = self._build_tree(X, y, list(range(X.shape[1])))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the provided data samples.
        """
        if self._root is None:
            raise Exception("The model has not been trained yet. Please call 'fit' before 'predict'.")

        predictions = np.array([self._root.predict(sample) for sample in X])
        return predictions

    def _entropy(self, y: np.ndarray) -> float:
        """
        Calculates the entropy of the target label distribution.
        """
        entropy = 0.0
        for label in np.unique(y):
            label_count = np.sum(y == label)
            probability = label_count / y.size
            entropy -= probability * np.log2(probability + 1e-9)  # Add small value to avoid log(0)
        return entropy

    def _information_gain(self, X: np.ndarray, y: np.ndarray, feature_index: int) -> float:
        """
        Calculates the Information Gain obtained by splitting the data on a specific feature.
        """
        initial_entropy = self._entropy(y)
        feature_values = np.unique(X[:, feature_index])
        weighted_entropy = 0.0

        for value in feature_values:
            subset_indices = np.where(X[:, feature_index] == value)[0]
            subset_y = y[subset_indices]
            weighted_entropy += (subset_y.size / y.size) * self._entropy(subset_y)

        return initial_entropy - weighted_entropy

    def _build_tree(self, X: np.ndarray, y: np.ndarray, available_features: List[int]) -> TreeNode:
        """
        Recursively builds the tree nodes by selecting the best attribute for splitting.
        """

        # If all labels are the same, create a leaf node
        if len(np.unique(y)) == 1:
            return TreeNode(value=y[0], is_leaf=True)

        # If no features are left, create a leaf node with the most common label
        if not available_features:
            most_common_label = np.bincount(y).argmax()
            return TreeNode(value=most_common_label, is_leaf=True)

        # Find the best feature to split on
        gains = [self._information_gain(X, y, feature) for feature in available_features]
        best_feature = available_features[np.argmax(gains)]

        # Create a root node for the current subtree
        root = TreeNode(feature_index=best_feature)

        # Split the data by the best feature
        for value in np.unique(X[:, best_feature]):
            subset_indices = np.where(X[:, best_feature] == value)[0]
            subset_X = X[subset_indices]
            subset_y = y[subset_indices]

            # Recursively build the subtree for the split data
            child_node = self._build_tree(subset_X, subset_y, [f for f in available_features if f != best_feature])
            root.children[value] = child_node

        return root

    """ Getters and Setters """

    @property
    def root(self) -> TreeNode:
        return self._root

    @root.setter
    def root(self, value: TreeNode):
        self._root = value

    @root.deleter
    def root(self):
        self._root = None

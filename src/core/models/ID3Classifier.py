from .TreeNode import TreeNode
import numpy as np
from typing import List
from collections import Counter


def entropy(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    label_counts = Counter(labels)
    total_count = len(labels)
    entropy = 0.0

    for count in label_counts.values():
        probability = count / total_count
        entropy -= probability * np.log2(probability + 1e-9)

    return entropy


def information_gain(features: np.ndarray, labels: np.ndarray, feature_index: int) -> float:
    base_entropy = entropy(labels)

    values = np.unique(features[:, feature_index])
    total = len(labels)

    split_entropy = 0.0

    for v in values:
        subset = labels[features[:, feature_index] == v]
        if len(subset) == 0:
            continue
        weight = len(subset) / total
        split_entropy += weight * entropy(subset)

    return base_entropy - split_entropy


class ID3Classifier():
    def __init__(self):
        self._root = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> 'ID3Classifier':
        self._root = self._build_tree(features, labels, list(range(features.shape[1])))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._root is None:
            raise Exception("Call fit before predict.")

        predictions = np.array([self._root.predict(sample) for sample in features])
        return predictions

    def _build_tree(self, features: np.ndarray, labels: np.ndarray, available_features: List[int]) -> TreeNode:
        if len(labels) == 0:
            return TreeNode(value=None, is_leaf=True, default_value=None)

        majority_label = Counter(labels).most_common(1)[0][0]

        if len(np.unique(labels)) == 1:
            return TreeNode(value=labels[0], is_leaf=True, default_value=labels[0])

        if not available_features:
            return TreeNode(value=majority_label, is_leaf=True, default_value=majority_label)

        best_feature = None
        best_gain = -1
        
        for feature_index in available_features:
            gain = information_gain(features, labels, feature_index)
            if best_feature is None or gain > best_gain:
                best_feature = feature_index
                best_gain = gain

        root = TreeNode(feature_index=best_feature, default_value=majority_label)

        for value in np.unique(features[:, best_feature]):
            subset_indices = np.where(features[:, best_feature] == value)[0]
            subset_features = features[subset_indices]
            subset_labels = labels[subset_indices]

            child_node = self._build_tree(subset_features, subset_labels, [f for f in available_features if f != best_feature])
            root.children[value] = child_node

        return root

    @property
    def root(self) -> TreeNode:
        return self._root

    @root.setter
    def root(self, value: TreeNode):
        self._root = value

    @root.deleter
    def root(self):
        self._root = None

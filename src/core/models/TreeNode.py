from typing import Any, Dict, Optional
import numpy as np


class TreeNode:
    """
    Represents a single node in the ID3 decision tree structure.
    """

    def __init__(
            self, 
            feature_index: Optional[int] = None,
            children: Optional[Dict[Any, 'TreeNode']] = None,
            value: Optional[Any] = None, 
            is_leaf: bool = False,
            default_value: Optional[Any] = None
        ):
        self._feature_index = feature_index
        self._children = children if children is not None else {}
        self._value = value
        self._is_leaf = is_leaf
        self._default_value = default_value

    def predict(self, sample: np.ndarray) -> Any:

        # Base case: if leaf node, return the stored class label
        if self._is_leaf:
            return self._value

        # Recursive case: traverse to the appropriate child node
        feature_value = sample[self._feature_index]
        child_node = self._children.get(feature_value)

        # If no child node exists for the feature value, return None
        if child_node is None:
            return self._default_value

        return child_node.predict(sample)

    @property
    def feature_index(self) -> Optional[int]:
        return self._feature_index

    @feature_index.setter
    def feature_index(self, value: int):
        self._feature_index = value

    @feature_index.deleter
    def feature_index(self):
        self._feature_index = None

    @property
    def children(self) -> Dict[Any, 'TreeNode']:
        return self._children

    @children.setter
    def children(self, value: Dict[Any, 'TreeNode']):
        self._children = value

    @children.deleter
    def children(self):
        self._children = None

    @property
    def value(self) -> Optional[Any]:
        return self._value

    @value.setter
    def value(self, value: Any):
        self._value = value

    @value.deleter
    def value(self):
        self._value = None

    @property
    def default_value(self) -> Optional[Any]:
        return self._default_value
    
    @default_value.setter
    def default_value(self, value: Any):
        self._default_value = value

    def is_leaf(self) -> bool:
        return self._is_leaf

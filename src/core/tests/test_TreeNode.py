from src.core.models.TreeNode import TreeNode
import numpy as np


class TestTreeNode:

    def test_initialization(self):
        node = TreeNode()
        assert node.feature_index is None
        assert node.children == {}
        assert node.value is None
        assert node.is_leaf() is False

        child = TreeNode(is_leaf=True)
        node_custom = TreeNode(feature_index=1, children={'a': child}, value=10, is_leaf=True)
        assert node_custom.feature_index == 1
        assert node_custom.children['a'] == child
        assert node_custom.value == 10
        assert node_custom.is_leaf() is True

    def test_predict_leaf_node(self):
        expected_class = "Class A"
        leaf = TreeNode(value=expected_class, is_leaf=True)

        sample = np.array([1, 2, 3])

        prediction = leaf.predict(sample)
        assert prediction == expected_class

    def test_predict_decision_node_traversal(self):
        left_leaf = TreeNode(value="Left", is_leaf=True)
        right_leaf = TreeNode(value="Right", is_leaf=True)

        root = TreeNode(
            feature_index=1,
            children={0: left_leaf, 1: right_leaf},
            is_leaf=False
        )

        sample_left = np.array([99, 0, 99])
        assert root.predict(sample_left) == "Left"

        sample_right = np.array([99, 1, 99])
        assert root.predict(sample_right) == "Right"

    def test_predict_deep_structure(self):
        leaf = TreeNode(value="DeepClass", is_leaf=True)

        middle = TreeNode(
            feature_index=2,
            children={5: leaf},
            is_leaf=False
        )

        root = TreeNode(
            feature_index=0,
            children={10: middle},
            is_leaf=False
        )

        sample = np.array([10, 0, 5])

        assert root.predict(sample) == "DeepClass"

    def test_predict_unknown_feature_value(self):
        child = TreeNode(value="A", is_leaf=True)
        root = TreeNode(
            feature_index=0,
            children={1: child},
            is_leaf=False
        )

        sample = np.array([99])

        result = root.predict(sample)
        assert result is None

    def test_getters_setters_deleters(self):
        node = TreeNode()

        node.feature_index = 5
        assert node.feature_index == 5
        del node.feature_index
        assert node.feature_index is None

        node.value = "Test"
        assert node.value == "Test"
        del node.value
        assert node.value is None

        child = TreeNode()
        node.children = {1: child}
        assert node.children[1] == child
        del node.children
        assert node.children is None

    def test_setter_logic_isolation(self):
        node = TreeNode()
        node.feature_index = 10
        assert node._feature_index == 10

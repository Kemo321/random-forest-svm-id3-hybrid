import pytest
import numpy as np
from src.core.models.ID3Classifier import ID3Classifier
from src.core.models.TreeNode import TreeNode


class TestID3Classifier:

    @pytest.fixture
    def classifier(self):
        return ID3Classifier()

    def test_initialization(self, classifier):
        assert classifier.root is None

    def test_entropy_homogeneous(self, classifier):
        y = np.array([1, 1, 1, 1])
        assert abs(classifier._entropy(y)) < 1e-4

    def test_entropy_split(self, classifier):
        y = np.array([1, 1, 0, 0])
        expected = 1.0
        assert np.isclose(classifier._entropy(y), expected)

    def test_entropy_multiclass(self, classifier):
        y = np.array([0, 0, 1, 1, 2, 2])
        # - (1/3 log2 1/3) * 3 = - log2(1/3) = log2(3) approx 1.58496
        expected = 1.58496
        assert np.isclose(classifier._entropy(y), expected, atol=1e-4)

    def test_information_gain_perfect_split(self, classifier):
        X = np.array([[0], [0], [1], [1]])
        y = np.array([0, 0, 1, 1])
        # H(S) = 1. Split by feat 0 gives two pure subsets. Weighted entropy = 0. IG = 1.
        ig = classifier._information_gain(X, y, 0)
        assert np.isclose(ig, 1.0)

    def test_information_gain_no_gain(self, classifier):
        X = np.array([[0], [1], [0], [1]])
        y = np.array([0, 0, 1, 1])
        # Feature is uncorrelated with Y.
        ig = classifier._information_gain(X, y, 0)
        assert np.isclose(ig, 0.0)

    def test_predict_without_fit_raises_exception(self, classifier):
        X = np.array([[1, 2]])
        with pytest.raises(Exception) as excinfo:
            classifier.predict(X)
        assert "not been trained" in str(excinfo.value)

    def test_simple_binary_classification(self, classifier):
        X = np.array([
            [0, 0],
            [1, 1],
        ])
        y = np.array([0, 1])

        classifier.fit(X, y)
        predictions = classifier.predict(X)

        np.testing.assert_array_equal(predictions, y)
        assert classifier.root is not None
        assert isinstance(classifier.root, TreeNode)

    def test_xor_problem(self, classifier):
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        y = np.array([0, 1, 1, 0])

        classifier.fit(X, y)
        predictions = classifier.predict(X)

        np.testing.assert_array_equal(predictions, y)

    def test_multiclass_classification(self, classifier):
        X = np.array([
            [0, 0],  # Class 0
            [1, 1],  # Class 1
            [2, 2]   # Class 2
        ])
        y = np.array([0, 1, 2])

        classifier.fit(X, y)
        predictions = classifier.predict(X)

        np.testing.assert_array_equal(predictions, y)

    def test_pure_leaf_base_case(self, classifier):
        X = np.array([[1], [2], [3]])
        y = np.array([1, 1, 1])

        classifier.fit(X, y)
        assert classifier.root.is_leaf()
        assert classifier.root.value == 1

    def test_no_features_left_base_case(self, classifier):
        X = np.array([
            [1],
            [1],
            [1]
        ])
        y = np.array([0, 1, 1])

        classifier.fit(X, y)
        prediction = classifier.predict(np.array([[1]]))

        assert prediction[0] == 1

    def test_getters_setters_deleters(self, classifier):
        mock_node = TreeNode(value=5, is_leaf=True)

        classifier.root = mock_node
        assert classifier.root == mock_node

        del classifier.root
        assert classifier.root is None

    def test_chaining_fit(self, classifier):
        X = np.array([[0], [1]])
        y = np.array([0, 1])
        result = classifier.fit(X, y)
        assert result is classifier

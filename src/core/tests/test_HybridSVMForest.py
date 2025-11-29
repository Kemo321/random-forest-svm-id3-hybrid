import pytest
import numpy as np
from unittest.mock import MagicMock
from sklearn.svm import SVC
from src.core.models.HybridSVMForest import HybridSVMForest
from src.core.models.ID3Classifier import ID3Classifier


class TestHybridSVMForest:

    @pytest.fixture
    def forest(self):
        return HybridSVMForest(n_estimators=5, p_svm=0.5, C=1.0, random_state=42)

    def test_initialization(self, forest):
        assert forest.n_estimators == 5
        assert forest.p_svm == 0.5
        assert forest.C == 1.0
        assert forest.random_state == 42
        assert forest.models == []
        assert forest._classes is None

    def test_fit_populates_models(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        forest = HybridSVMForest(n_estimators=3, p_svm=0.5, random_state=42)
        forest.fit(X, y)

        assert len(forest.models) == 3
        assert forest.classes is not None
        np.testing.assert_array_equal(forest.classes, np.unique(y))

    def test_fit_only_svm(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        forest = HybridSVMForest(n_estimators=2, p_svm=1.0, random_state=42)
        forest.fit(X, y)

        for model in forest.models:
            assert isinstance(model, SVC)
            assert model.C == 1.0

    def test_fit_only_id3(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        forest = HybridSVMForest(n_estimators=2, p_svm=0.0, random_state=42)
        forest.fit(X, y)

        for model in forest.models:
            assert isinstance(model, ID3Classifier)

    def test_predict_without_fit_raises_exception(self, forest):
        X = np.array([[1, 2]])
        with pytest.raises(Exception) as excinfo:
            forest.predict(X)
        assert "not been trained" in str(excinfo.value)

    def test_predict_majority_voting_logic(self):
        forest = HybridSVMForest(n_estimators=3)
        forest.classes = np.array([0, 1])

        model1 = MagicMock()
        model1.predict.return_value = np.array([0, 1, 0])

        model2 = MagicMock()
        model2.predict.return_value = np.array([0, 1, 1])

        model3 = MagicMock()
        model3.predict.return_value = np.array([1, 0, 0])

        forest.models = [model1, model2, model3]

        X_dummy = np.zeros((3, 2))
        predictions = forest.predict(X_dummy)

        expected_predictions = np.array([0, 1, 0])
        np.testing.assert_array_equal(predictions, expected_predictions)

    def test_getters_setters_deleters(self, forest):
        forest.n_estimators = 100
        assert forest.n_estimators == 100
        del forest.n_estimators
        assert forest.n_estimators is None

        forest.p_svm = 0.9
        assert forest.p_svm == 0.9
        del forest.p_svm
        assert forest.p_svm is None

        dummy_models = [1, 2, 3]
        forest.models = dummy_models
        assert forest.models == dummy_models
        del forest.models
        assert forest.models is None

    def test_integration_binary_classification(self):
        X = np.array([[0], [0], [1], [1]] * 5)
        y = np.array([0, 0, 1, 1] * 5)

        forest = HybridSVMForest(n_estimators=4, p_svm=0.5, random_state=42)
        forest.fit(X, y)
        preds = forest.predict(X)

        assert preds.shape == (20,)
        assert np.all(np.isin(preds, [0, 1]))

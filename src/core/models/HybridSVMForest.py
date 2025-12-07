from .ID3Classifier import ID3Classifier
from sklearn.svm import LinearSVC
import numpy as np
from typing import Optional


class HybridSVMForest():
    def __init__(self, n_estimators: int = 10, p_svm: float = 0.5, C: float = 1.0, random_state: Optional[int] = None):
        self._estimator_count = n_estimators
        self._p_svm = p_svm
        self._C = C
        self._random_state = random_state
        self._models = []
        self._feature_indices = []
        self._classes_array = None
        self._majority_class = None

    def fit(self, features: tuple, labels: np.ndarray) -> 'HybridSVMForest':
        self._classes_array = np.unique(labels)
        self._models.clear()
        self._feature_indices.clear()

        self._majority_class = np.bincount(labels.astype(int)).argmax()

        rng = np.random.default_rng(self._random_state)
        n_samples = features[0].shape[0]

        for _ in range(self._estimator_count):

            indices = rng.choice(n_samples, size=n_samples, replace=True)
            y_sample = labels[indices]

            if rng.random() < self._p_svm:
                X_sample = features[1][indices]
                model = LinearSVC(
                    C=self._C,
                    random_state=self._random_state,
                    dual=False,
                    max_iter=2000
                )
            else:
                X_sample = features[0][indices]
                model = ID3Classifier()

            n_features = X_sample.shape[1]
            k = max(1, int(np.sqrt(n_features)))
            f_idx = rng.choice(n_features, size=k, replace=False)
            X_sample = X_sample[:, f_idx]

            model.fit(X_sample, y_sample)
            self._models.append(model)
            self._feature_indices.append(f_idx)

        return self

    def predict(self, features: tuple) -> np.ndarray:
        if not self._models:
            raise Exception("Call fit before predict.")

        predictions = []

        for i, model in enumerate(self._models):
            f_idx = self._feature_indices[i]

            if isinstance(model, LinearSVC):
                X_input = features[1][:, f_idx]
            else:
                X_input = features[0][:, f_idx]

            preds = model.predict(X_input)
            predictions.append(preds)

        predictions = np.array(predictions)
        final_predictions = []

        most_common_class = self._majority_class

        for i in range(features[0].shape[0]):
            votes = predictions[:, i]
            if any(v is None for v in votes):
                print("Warning: None value found in predictions for sample index", i)

            votes_clean = np.array([v if v is not None else most_common_class for v in votes])

            try:
                votes_int = votes_clean.astype(np.int64)
            except ValueError as e:
                raise TypeError("Predicted class labels must be convertible to integers for majority voting.") from e

            counts = np.bincount(votes_int, minlength=len(self._classes_array))
            final_predictions.append(np.argmax(counts))

        return np.array(final_predictions)

    @property
    def estimator_count(self) -> int: return self._estimator_count

    @estimator_count.setter
    def estimator_count(self, value: int): self._estimator_count = value

    @estimator_count.deleter
    def estimator_count(self):
        self._estimator_count = None

    @property
    def p_svm(self) -> float: return self._p_svm

    @p_svm.setter
    def p_svm(self, value: float): self._p_svm = value

    @p_svm.deleter
    def p_svm(self):
        self._p_svm = None

    @property
    def C(self) -> float: return self._C

    @C.setter
    def C(self, value: float): self._C = value

    @C.deleter
    def C(self):
        self._C = None

    @property
    def random_state(self) -> Optional[int]: return self._random_state

    @random_state.setter
    def random_state(self, value: Optional[int]): self._random_state = value

    @random_state.deleter
    def random_state(self):
        self._random_state = None

    @property
    def models(self) -> list: return self._models

    @models.setter
    def models(self, value: list): self._models = value

    @models.deleter
    def models(self):
        self._models = None

from .ID3Classifier import ID3Classifier
from sklearn.svm import LinearSVC
import numpy as np
from typing import Optional, Union, Tuple


class HybridSVMForest():
    def __init__(self, estimator_count: int = 10, p_svm: float = 0.5, C: float = 1.0,
                 random_state: Optional[int] = None, max_features: Union[str, int, float] = 'sqrt'):
        self._estimator_count = estimator_count
        self._p_svm = p_svm
        self._C = C
        self._random_state = random_state
        self._max_features = max_features
        self._models = []
        self._feature_indices = []
        self._classes_array = None
        self._majority_class = None

    def fit(self, features: Tuple[np.ndarray, np.ndarray], labels: np.ndarray) -> 'HybridSVMForest':
        self._classes_array = np.unique(labels)
        self._models.clear()
        self._feature_indices.clear()

        if len(labels) == 0:
            raise ValueError("Labels array is empty.")

        self._majority_class = np.bincount(labels.astype(int)).argmax()

        rng = np.random.default_rng(self._random_state)
        n_samples = features[0].shape[0]

        for _ in range(self._estimator_count):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            y_sample = labels[indices]

            is_svm = rng.random() < self._p_svm

            if is_svm:
                X_full = features[1]
                model = LinearSVC(C=self._C, random_state=self._random_state, dual=False, max_iter=2000)
            else:
                X_full = features[0]
                model = ID3Classifier()

            X_sample_rows = X_full[indices]
            n_features_total = X_sample_rows.shape[1]

            if isinstance(self._max_features, int):
                k = min(self._max_features, n_features_total)
            elif isinstance(self._max_features, float):
                k = max(1, int(self._max_features * n_features_total))
            elif self._max_features == 'sqrt':
                k = int(np.sqrt(n_features_total))
            elif self._max_features == 'log2':
                k = int(np.log2(n_features_total))
            else:
                k = n_features_total

            if n_features_total <= 10:
                k = max(k, int(n_features_total * 0.7))
                k = max(k, 1)

            if is_svm and n_features_total > 20:
                k = max(k, int(n_features_total * 0.5))

            f_idx = rng.choice(n_features_total, size=k, replace=False)
            X_sample_final = X_sample_rows[:, f_idx]

            model.fit(X_sample_final, y_sample)

            self._models.append(model)
            self._feature_indices.append(f_idx)

        return self

    def predict(self, features: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
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

            votes_clean = [v if v is not None else most_common_class for v in votes]

            if not votes_clean:
                final_predictions.append(most_common_class)
                continue

            try:
                votes_int = np.array(votes_clean, dtype=np.int64)
                counts = np.bincount(votes_int, minlength=len(self._classes_array))
                final_predictions.append(np.argmax(counts))
            except ValueError:
                final_predictions.append(most_common_class)

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

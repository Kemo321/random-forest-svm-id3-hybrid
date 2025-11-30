from .ID3Classifier import ID3Classifier
from sklearn.svm import LinearSVC
import numpy as np
from typing import Optional, Any


class HybridSVMForest():
    def __init__(self, n_estimators: int = 10, p_svm: float = 0.5, C: float = 1.0, random_state: Optional[int] = None):
        self._n_estimators = n_estimators
        self._p_svm = p_svm
        self._C = C
        self._random_state = random_state
        self._models = []
        self._classes_array = None

    def fit(self, X: tuple, y: np.ndarray) -> 'HybridSVMForest':
        self._models = []
        self._classes_array = np.unique(y)

        rng = np.random.default_rng(self._random_state)
        n_samples = X[0].shape[0]

        for _ in range(self._n_estimators):

            indices = rng.choice(n_samples, size=n_samples, replace=True)
            y_sample = y[indices]

            if rng.random() < self._p_svm:
                X_sample = X[1][indices]
                model = LinearSVC(
                    C=self._C,
                    random_state=self._random_state,
                    dual=False,
                    max_iter=2000
                )
            else:
                X_sample = X[0][indices]
                model = ID3Classifier()

            model.fit(X_sample, y_sample)
            self._models.append(model)

        return self

    def predict(self, X: tuple) -> np.ndarray:
        if not self._models:
            raise Exception("The model has not been trained yet. Please call 'fit' before 'predict'.")

        predictions = []

        for model in self._models:
            if isinstance(model, LinearSVC):
                current_preds = model.predict(X[1])
            elif isinstance(model, ID3Classifier):
                current_preds = model.predict(X[0])
            else:
                current_preds = model.predict(X[0])

            predictions.append(current_preds)

        predictions = np.array(predictions)
        final_predictions = []
        
        try:
             most_common_class = self._classes_array[np.argmax(np.bincount(self._classes_array))]
        except:
             most_common_class = 0


        for i in range(X[0].shape[0]):
            votes = predictions[:, i]

            votes_clean = np.array([v if v is not None else most_common_class for v in votes])

            try:
                votes_int = votes_clean.astype(np.int64)
            except ValueError as e:
                raise TypeError("Predicted class labels must be convertible to integers for majority voting.") from e

            counts = np.bincount(votes_int, minlength=len(self._classes_array))
            final_predictions.append(np.argmax(counts))

        return np.array(final_predictions)

    @property
    def n_estimators(self) -> int: return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, value: int): self._n_estimators = value

    @property
    def p_svm(self) -> float: return self._p_svm

    @p_svm.setter
    def p_svm(self, value: float): self._p_svm = value

    @property
    def C(self) -> float: return self._C

    @C.setter
    def C(self, value: float): self._C = value

    @property
    def random_state(self) -> Optional[int]: return self._random_state

    @random_state.setter
    def random_state(self, value: Optional[int]): self._random_state = value

    @property
    def models(self) -> list: return self._models

    @models.setter
    def models(self, value: list): self._models = value

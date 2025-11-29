from .ID3Classifier import ID3Classifier
import sklearn.svm as svm
import numpy as np
from typing import Optional


class HybridSVMForest():
    """
    Hybrid ensemble classifier combining custom ID3 trees and SVMs using Bagging.
    """

    def __init__(self, n_estimators: int = 10, p_svm: float = 0.5, C: float = 1.0, random_state: Optional[int] = None):
        """
        Initializes the hybrid forest parameters.
        """
        self.n_estimators = n_estimators
        self.p_svm = p_svm
        self.C = C
        self.random_state = random_state
        self.models = []
        self._classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HybridSVMForest':
        """
        Trains the ensemble by creating bootstrap samples and training either SVM or ID3 models.
        """
        self.models = []
        self.classes = np.unique(y)

        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):

            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            if rng.random() < self.p_svm:
                model = svm.SVC(C=self.C, probability=False, random_state=self.random_state)
            else:
                model = ID3Classifier()

            model.fit(X_sample, y_sample)
            self.models.append(model)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for X using majority voting from all trained base estimators.
        """

        if not self.models:
            raise Exception("The model has not been trained yet. Please call 'fit' before 'predict'.")

        predictions = np.array([model.predict(X) for model in self.models])
        final_predictions = []

        for i in range(X.shape[0]):
            votes = predictions[:, i]
            counts = np.bincount(votes, minlength=len(self.classes))
            final_predictions.append(np.argmax(counts))

        return np.array(final_predictions)

    """ Getters and Setters """

    @property
    def n_estimators(self) -> int:
        return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, value: int):
        self._n_estimators = value

    @n_estimators.deleter
    def n_estimators(self):
        self._n_estimators = None

    @property
    def p_svm(self) -> float:
        return self._p_svm

    @p_svm.setter
    def p_svm(self, value: float):
        self._p_svm = value

    @p_svm.deleter
    def p_svm(self):
        self._p_svm = None

    @property
    def C(self) -> float:
        return self._C

    @C.setter
    def C(self, value: float):
        self._C = value

    @C.deleter
    def C(self):
        self._C = None

    @property
    def random_state(self) -> Optional[int]:
        return self._random_state

    @random_state.setter
    def random_state(self, value: Optional[int]):
        self._random_state = value

    @random_state.deleter
    def random_state(self):
        self._random_state = None

    @property
    def models(self) -> list:
        return self._models

    @models.setter
    def models(self, value: list):
        self._models = value

    @models.deleter
    def models(self):
        self._models = None

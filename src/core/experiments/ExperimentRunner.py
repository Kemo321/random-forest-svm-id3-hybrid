import time
from typing import Dict, Any, Type, List, Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


class ExperimentRunner:
    def __init__(self, n_repeats: int = 25, n_splits: int = 5) -> None:
        self.n_repeats: int = n_repeats
        self.n_splits: int = n_splits

    def run_cv(
        self,
        X_id3: np.ndarray,
        X_svm: np.ndarray,
        y: np.ndarray,
        model_class: Type[BaseEstimator],
        model_params: Dict[str, Any],
    ) -> Dict[str, float]:
        accuracies: List[float] = []
        base_seed: int = int(time.time())

        for i in range(self.n_repeats):
            current_seed: int = base_seed + i
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=current_seed)

            fold_accuracies: List[float] = []
            for train_index, test_index in skf.split(X_id3, y):
                y_train, y_test = y[train_index], y[test_index]

                X_train_tuple = (X_id3[train_index], X_svm[train_index])
                X_test_tuple = (X_id3[test_index], X_svm[test_index])

                if "random_state" in model_class.__init__.__code__.co_varnames:
                    model: BaseEstimator = model_class(**model_params, random_state=current_seed)
                else:
                    model = model_class(**model_params)

                model.fit(X_train_tuple, y_train)
                preds = model.predict(X_test_tuple)
                acc: float = accuracy_score(y_test, preds)
                fold_accuracies.append(acc)

            accuracies.append(float(np.mean(fold_accuracies)))

        return {
            "mean_acc": float(np.mean(accuracies)),
            "std_acc": float(np.std(accuracies)),
            "min_acc": float(np.min(accuracies)),
            "max_acc": float(np.max(accuracies)),
        }

    def run_single_with_confusion_matrix(
        self,
        X_id3: np.ndarray,
        X_svm: np.ndarray,
        y: np.ndarray,
        model_class: Type[BaseEstimator],
        model_params: Dict[str, Any],
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        X_id3_tr, X_id3_te, y_train, y_test = train_test_split(
            X_id3, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_svm_tr, X_svm_te, _, _ = train_test_split(
            X_svm, y, test_size=test_size, random_state=random_state, stratify=y
        )

        if "random_state" in model_class.__init__.__code__.co_varnames:
            model = model_class(**model_params, random_state=random_state)
        else:
            model = model_class(**model_params)

        model.fit((X_id3_tr, X_svm_tr), y_train)
        preds = model.predict((X_id3_te, X_svm_te))

        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        return {
            "accuracy": float(acc),
            "confusion_matrix": cm,
            "y_test": y_test,
            "y_pred": preds,
        }

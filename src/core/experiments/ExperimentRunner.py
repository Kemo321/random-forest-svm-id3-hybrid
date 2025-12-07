import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


class ExperimentRunner:
    def __init__(self, n_repeats=25, n_splits=5):
        self.n_repeats = n_repeats
        self.n_splits = n_splits

    def run_cv(self, X_id3, X_svm, y, model_class, model_params):
        accuracies = []
        base_seed = int(time.time())

        for i in range(self.n_repeats):
            current_seed = base_seed + i
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=current_seed)

            fold_accuracies = []
            for train_index, test_index in skf.split(X_id3, y):
                y_train, y_test = y[train_index], y[test_index]

                X_train_tuple = (X_id3[train_index], X_svm[train_index])
                X_test_tuple = (X_id3[test_index], X_svm[test_index])

                if 'random_state' in model_class.__init__.__code__.co_varnames:
                    model = model_class(**model_params, random_state=current_seed)
                else:
                    model = model_class(**model_params)

                model.fit(X_train_tuple, y_train)
                preds = model.predict(X_test_tuple)
                acc = accuracy_score(y_test, preds)
                fold_accuracies.append(acc)

            accuracies.append(np.mean(fold_accuracies))

        return {
            "mean_acc": np.mean(accuracies),
            "std_acc": np.std(accuracies),
            "min_acc": np.min(accuracies),
            "max_acc": np.max(accuracies),
        }

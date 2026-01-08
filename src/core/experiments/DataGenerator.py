"""
Data Generator - generates CSV results for all experimental scenarios.

This module handles:
- Scenario 1: Impact of p_svm (T=20, C=1.0, p_svm ∈ {0.0, 0.2, 0.5, 0.8, 1.0})
- Scenario 2: Impact of T (p_svm=0.5, C=1.0, T ∈ {10, 20, 50, 100})
- Scenario 3: Impact of C (p_svm=1.0, T=20, C ∈ {0.1, 1.0, 10.0, 50.0})
- Heatmaps: Confusion matrices (T=20, p_svm=0.5, C=1.0)
- Overfitting: Train vs Test accuracy comparison
"""

import os
from typing import Dict, Any, Type, List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix


class DataGenerator:
    """Generates experimental results as CSV files."""

    # Default experimental parameters from badania.txt
    DEFAULT_N_REPEATS = 25
    DEFAULT_N_SPLITS = 5

    # Scenario 1: p_svm impact
    SCENARIO1_T = 20
    SCENARIO1_C = 1.0
    SCENARIO1_P_SVM_VALUES = [0.0, 0.2, 0.5, 0.8, 1.0]

    # Scenario 2: T impact
    SCENARIO2_P_SVM = 0.5
    SCENARIO2_C = 1.0
    SCENARIO2_T_VALUES = [10, 20, 50, 100]

    # Scenario 3: C impact
    SCENARIO3_P_SVM = 1.0
    SCENARIO3_T = 20
    SCENARIO3_C_VALUES = [0.1, 1.0, 10.0, 50.0]

    # Heatmaps and Overfitting params (reuse from Scenario 1 p_svm=0.5)
    COMMON_T = 20
    COMMON_P_SVM = 0.5
    COMMON_C = 1.0

    def __init__(
        self,
        n_repeats: int = DEFAULT_N_REPEATS,
        n_splits: int = DEFAULT_N_SPLITS,
        results_dir: str = "./results"
    ) -> None:
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Cache for reusing computed results
        self._cached_results: Dict[str, Any] = {}

    def _get_cache_key(self, ds_name: str, T: int, p_svm: float, C: float) -> str:
        """Generate unique cache key for model parameters."""
        return f"{ds_name}_T{T}_p{p_svm}_C{C}"

    def run_cv(
        self,
        X_id3: np.ndarray,
        X_svm: np.ndarray,
        y: np.ndarray,
        model_class: Type[BaseEstimator],
        model_params: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Run cross-validation experiment with specified parameters.

        Returns dict with: mean_acc, std_acc, min_acc, max_acc
        """
        accuracies: List[float] = []
        base_seed: int = 42  # Fixed seed for reproducibility

        for i in range(self.n_repeats):
            current_seed: int = base_seed + i
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=current_seed
            )

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

    def run_cv_with_train_test(
        self,
        X_id3: np.ndarray,
        X_svm: np.ndarray,
        y: np.ndarray,
        model_class: Type[BaseEstimator],
        model_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run cross-validation with both train and test accuracy measurement.
        Used for overfitting analysis and confusion matrix generation.

        Performs n_repeats independent CV runs, each with n_splits folds.
        Returns aggregated train/test accuracies and confusion matrix.
        """
        train_accs: List[float] = []  # Per-repeat mean train accuracy
        test_accs: List[float] = []   # Per-repeat mean test accuracy
        all_y_test: List[np.ndarray] = []
        all_preds: List[np.ndarray] = []
        base_seed: int = 42

        for i in range(self.n_repeats):
            current_seed: int = base_seed + i
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=current_seed
            )

            fold_train_accs: List[float] = []
            fold_test_accs: List[float] = []

            for train_index, test_index in skf.split(X_id3, y):
                y_train, y_test = y[train_index], y[test_index]

                X_train_tuple = (X_id3[train_index], X_svm[train_index])
                X_test_tuple = (X_id3[test_index], X_svm[test_index])

                if "random_state" in model_class.__init__.__code__.co_varnames:
                    model = model_class(**model_params, random_state=current_seed)
                else:
                    model = model_class(**model_params)

                model.fit(X_train_tuple, y_train)

                # Train accuracy (on training fold)
                train_preds = model.predict(X_train_tuple)
                train_acc = accuracy_score(y_train, train_preds)
                fold_train_accs.append(train_acc)

                # Test accuracy (on test fold)
                test_preds = model.predict(X_test_tuple)
                test_acc = accuracy_score(y_test, test_preds)
                fold_test_accs.append(test_acc)

                # Collect for confusion matrix
                all_y_test.append(y_test)
                all_preds.append(test_preds)

            # Mean accuracy for this repeat (across folds)
            train_accs.append(float(np.mean(fold_train_accs)))
            test_accs.append(float(np.mean(fold_test_accs)))

        # Aggregate confusion matrix from all folds of all repeats
        y_test_combined = np.concatenate(all_y_test)
        preds_combined = np.concatenate(all_preds)
        cm = confusion_matrix(y_test_combined, preds_combined)

        mean_train = float(np.mean(train_accs))
        mean_test = float(np.mean(test_accs))

        return {
            "train_acc_mean": mean_train,
            "train_acc_std": float(np.std(train_accs)),
            "train_acc_min": float(np.min(train_accs)),
            "train_acc_max": float(np.max(train_accs)),
            "test_acc_mean": mean_test,
            "test_acc_std": float(np.std(test_accs)),
            "test_acc_min": float(np.min(test_accs)),
            "test_acc_max": float(np.max(test_accs)),
            "delta": mean_train - mean_test,
            "confusion_matrix": cm,
        }


    def generate_scenario1_results(
        self,
        datasets_config: List[Dict[str, Any]],
        model_class: Type[BaseEstimator],
    ) -> pd.DataFrame:
        """
        Scenario 1 (4.2): Impact of p_svm on quality.
        T=20, C=1.0, p_svm ∈ {0.0, 0.2, 0.5, 0.8, 1.0}
        """
        print("\n" + "=" * 60)
        print("Scenario 1: Impact of p_svm (T=20, C=1.0)")
        print("=" * 60)

        results = []

        for ds in datasets_config:
            ds_name = ds["name"]
            print(f"\n  Dataset: {ds_name}")

            try:
                loaded_data = ds["loader"]()
                if len(loaded_data) == 3:
                    X_id3, X_svm, y = loaded_data
                else:
                    X_id3, y = loaded_data
                    X_svm = X_id3

                for p_svm in self.SCENARIO1_P_SVM_VALUES:
                    print(f"    p_svm={p_svm}...", end=" ", flush=True)

                    params = {
                        "estimator_count": self.SCENARIO1_T,
                        "p_svm": p_svm,
                        "C": self.SCENARIO1_C
                    }

                    stats = self.run_cv(X_id3, X_svm, y, model_class, params)

                    # Cache result with common params for reuse
                    cache_key = self._get_cache_key(
                        ds_name, self.SCENARIO1_T, p_svm, self.SCENARIO1_C
                    )
                    self._cached_results[cache_key] = stats

                    results.append({
                        "dataset": ds_name,
                        "p_svm": p_svm,
                        "T": self.SCENARIO1_T,
                        "C": self.SCENARIO1_C,
                        **stats
                    })
                    print(f"acc={stats['mean_acc']:.4f}")

            except Exception as e:
                print(f"    Error: {e}")
                continue

        df = pd.DataFrame(results)
        csv_path = os.path.join(self.results_dir, "results_impact_p_svm.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        return df

    def generate_scenario2_results(
        self,
        datasets_config: List[Dict[str, Any]],
        model_class: Type[BaseEstimator],
    ) -> pd.DataFrame:
        """
        Scenario 2 (4.3): Impact of T (estimator count) on quality.
        p_svm=0.5, C=1.0, T ∈ {10, 20, 50, 100}
        """
        print("\n" + "=" * 60)
        print("Scenario 2: Impact of T (p_svm=0.5, C=1.0)")
        print("=" * 60)

        results = []

        for ds in datasets_config:
            ds_name = ds["name"]
            print(f"\n  Dataset: {ds_name}")

            try:
                loaded_data = ds["loader"]()
                if len(loaded_data) == 3:
                    X_id3, X_svm, y = loaded_data
                else:
                    X_id3, y = loaded_data
                    X_svm = X_id3

                for T in self.SCENARIO2_T_VALUES:
                    # Check cache first (T=20 should be cached from Scenario 1)
                    cache_key = self._get_cache_key(
                        ds_name, T, self.SCENARIO2_P_SVM, self.SCENARIO2_C
                    )

                    if cache_key in self._cached_results:
                        print(f"    T={T}... (cached)", end=" ")
                        stats = self._cached_results[cache_key]
                    else:
                        print(f"    T={T}...", end=" ", flush=True)
                        params = {
                            "estimator_count": T,
                            "p_svm": self.SCENARIO2_P_SVM,
                            "C": self.SCENARIO2_C
                        }
                        stats = self.run_cv(X_id3, X_svm, y, model_class, params)
                        self._cached_results[cache_key] = stats

                    results.append({
                        "dataset": ds_name,
                        "estimator_count": T,
                        "p_svm": self.SCENARIO2_P_SVM,
                        "C": self.SCENARIO2_C,
                        **stats
                    })
                    print(f"acc={stats['mean_acc']:.4f}")

            except Exception as e:
                print(f"    Error: {e}")
                continue

        df = pd.DataFrame(results)
        csv_path = os.path.join(self.results_dir, "results_impact_estimator_count.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        return df

    def generate_scenario3_results(
        self,
        datasets_config: List[Dict[str, Any]],
        model_class: Type[BaseEstimator],
    ) -> pd.DataFrame:
        """
        Scenario 3 (4.4): Impact of C (SVM regularization) on quality.
        p_svm=1.0, T=20, C ∈ {0.1, 1.0, 10.0, 50.0}
        """
        print("\n" + "=" * 60)
        print("Scenario 3: Impact of C (p_svm=1.0, T=20)")
        print("=" * 60)

        results = []

        for ds in datasets_config:
            ds_name = ds["name"]
            print(f"\n  Dataset: {ds_name}")

            try:
                loaded_data = ds["loader"]()
                if len(loaded_data) == 3:
                    X_id3, X_svm, y = loaded_data
                else:
                    X_id3, y = loaded_data
                    X_svm = X_id3

                for C in self.SCENARIO3_C_VALUES:
                    # Check cache first (C=1.0 with p_svm=1.0 should be from Scenario 1)
                    cache_key = self._get_cache_key(
                        ds_name, self.SCENARIO3_T, self.SCENARIO3_P_SVM, C
                    )

                    if cache_key in self._cached_results:
                        print(f"    C={C}... (cached)", end=" ")
                        stats = self._cached_results[cache_key]
                    else:
                        print(f"    C={C}...", end=" ", flush=True)
                        params = {
                            "estimator_count": self.SCENARIO3_T,
                            "p_svm": self.SCENARIO3_P_SVM,
                            "C": C
                        }
                        stats = self.run_cv(X_id3, X_svm, y, model_class, params)
                        self._cached_results[cache_key] = stats

                    results.append({
                        "dataset": ds_name,
                        "C": C,
                        "estimator_count": self.SCENARIO3_T,
                        "p_svm": self.SCENARIO3_P_SVM,
                        **stats
                    })
                    print(f"acc={stats['mean_acc']:.4f}")

            except Exception as e:
                print(f"    Error: {e}")
                continue

        df = pd.DataFrame(results)
        csv_path = os.path.join(self.results_dir, "results_impact_C.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        return df

    def generate_overfitting_results(
        self,
        datasets_config: List[Dict[str, Any]],
        model_class: Type[BaseEstimator],
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Generate overfitting analysis (Train vs Test accuracy).
        Also generates confusion matrices for heatmaps.

        Parameters: T=20, p_svm=0.5, C=1.0 (same as default config)

        Returns:
            - DataFrame with train/test accuracy comparison
            - Dict of confusion matrices by dataset name
        """
        print("\n" + "=" * 60)
        print("Overfitting Analysis & Confusion Matrices")
        print(f"(T={self.COMMON_T}, p_svm={self.COMMON_P_SVM}, C={self.COMMON_C})")
        print("=" * 60)

        results = []
        confusion_matrices = {}
        cm_dir = os.path.join(self.results_dir, "confusion_matrices")
        os.makedirs(cm_dir, exist_ok=True)

        for ds in datasets_config:
            ds_name = ds["name"]
            print(f"\n  Dataset: {ds_name}...", end=" ", flush=True)

            try:
                loaded_data = ds["loader"]()
                if len(loaded_data) == 3:
                    X_id3, X_svm, y = loaded_data
                else:
                    X_id3, y = loaded_data
                    X_svm = X_id3

                params = {
                    "estimator_count": self.COMMON_T,
                    "p_svm": self.COMMON_P_SVM,
                    "C": self.COMMON_C
                }

                eval_result = self.run_cv_with_train_test(
                    X_id3, X_svm, y, model_class, params
                )

                results.append({
                    "dataset": ds_name,
                    "T": self.COMMON_T,
                    "p_svm": self.COMMON_P_SVM,
                    "C": self.COMMON_C,
                    "train_acc_mean": eval_result["train_acc_mean"],
                    "train_acc_std": eval_result["train_acc_std"],
                    "train_acc_min": eval_result["train_acc_min"],
                    "train_acc_max": eval_result["train_acc_max"],
                    "test_acc_mean": eval_result["test_acc_mean"],
                    "test_acc_std": eval_result["test_acc_std"],
                    "test_acc_min": eval_result["test_acc_min"],
                    "test_acc_max": eval_result["test_acc_max"],
                    "delta": eval_result["delta"],
                })

                # Save confusion matrix
                cm = eval_result["confusion_matrix"]
                confusion_matrices[ds_name] = cm

                ds_name_clean = ds_name.replace(' ', '_').replace('-', '_')
                cm_path = os.path.join(cm_dir, f"cm_hybrid_{ds_name_clean}.csv")
                np.savetxt(cm_path, cm, delimiter=',', fmt='%d')

                print(f"Train={eval_result['train_acc_mean']:.4f}, "
                      f"Test={eval_result['test_acc_mean']:.4f}, "
                      f"Δ={eval_result['delta']:.4f}")

            except Exception as e:
                print(f"Error: {e}")
                continue

        df = pd.DataFrame(results)
        csv_path = os.path.join(self.results_dir, "results_overfitting.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        return df, confusion_matrices

    def generate_all(
        self,
        datasets_config: List[Dict[str, Any]],
        model_class: Type[BaseEstimator],
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate all experimental results.

        Returns dict of DataFrames with keys:
        - 'p_svm': Scenario 1 results
        - 'estimator_count': Scenario 2 results
        - 'C': Scenario 3 results
        - 'overfitting': Train vs Test comparison
        """
        print("\n" + "=" * 70)
        print("DATA GENERATION - Starting all experiments")
        print("=" * 70)

        results = {}

        # Run Scenario 1 first - this caches results for p_svm=0.5, T=20, C=1.0
        results['p_svm'] = self.generate_scenario1_results(datasets_config, model_class)

        # Scenario 2 - will use cached T=20 result
        results['estimator_count'] = self.generate_scenario2_results(datasets_config, model_class)

        # Scenario 3 - will use cached p_svm=1.0, T=20, C=1.0 from Scenario 1
        results['C'] = self.generate_scenario3_results(datasets_config, model_class)

        # Overfitting analysis with confusion matrices
        results['overfitting'], _ = self.generate_overfitting_results(datasets_config, model_class)

        print("\n" + "=" * 70)
        print("DATA GENERATION COMPLETE")
        print("=" * 70)

        return results

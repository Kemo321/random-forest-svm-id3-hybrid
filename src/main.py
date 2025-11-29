import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearnex import patch_sklearn
from core.models.HybridSVMForest import HybridSVMForest
from core.utils.DataLoader import DataLoader

patch_sklearn()


def run_single_experiment(X, y, model_class, model_params, n_repeats=25, n_splits=5):
    """Run n_repeats of n_splits stratified cross-validation and return accuracy stats."""
    accuracies = []

    for i in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)

        fold_accuracies = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = model_class(**model_params, random_state=i)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            fold_accuracies.append(acc)

        accuracies.append(np.mean(fold_accuracies))

    return {
        "mean_acc": np.mean(accuracies),
        "std_acc": np.std(accuracies),
        "min_acc": np.min(accuracies),
        "max_acc": np.max(accuracies),
    }


def main():
    datasets_config = [
        {
            "name": "Mushroom Data Set",
            "type": "Discrete",
            "loader": lambda: DataLoader.load_mushroom_data(),
        },
        {
            "name": "Wisconsin Breast Cancer",
            "type": "Continuous (Discretized)",
            "loader": lambda: DataLoader.load_breast_cancer_data(n_bins=5),
        },
        {
            "name": "Wine Quality - Red",
            "type": "Continuous (Discretized)",
            "loader": lambda: DataLoader.load_wine_data(n_bins=5),
        },
    ]

    all_results = []

    fixed_n_estimators = 20
    p_svm_values = [0.0, 0.2, 0.5, 0.8, 1.0]

    for ds in datasets_config:
        ds_name = ds["name"]
        print("\n" + "=" * 80)
        print(f"PROCESSING DATASET: {ds_name} ({ds['type']})")
        print("=" * 80)

        try:
            X, y = ds["loader"]()
            print(f"Data loaded successfully. X shape: {X.shape}, Number of classes: {len(np.unique(y))}")
        except Exception as e:
            print(f"❌ ERROR loading dataset {ds_name}: {e}")
            continue

        print("\n--- Investigating impact of SVM share (p_svm) [N={}] ---".format(fixed_n_estimators))
        for p in p_svm_values:
            print(f"  > Testing p_svm = {p:<3} ...", end=" ", flush=True)

            stats = run_single_experiment(
                X,
                y,
                HybridSVMForest,
                {"n_estimators": fixed_n_estimators, "p_svm": p, "C": 1.0},
            )
            print(f"Result: {stats['mean_acc']:.4f} (std: {stats['std_acc']:.4f})")

            all_results.append(
                {
                    "dataset": ds_name,
                    "experiment": "impact_p_svm",
                    "p_svm": p,
                    "n_estimators": fixed_n_estimators,
                    "model_type": "HybridSVMForest",
                    **stats,
                }
            )

        print("\n--- Reference method: RandomForestClassifier ---")
        print("  > Testing Random Forest ...", end=" ", flush=True)

        stats_rf = run_single_experiment(X, y, RandomForestClassifier, {"n_estimators": fixed_n_estimators})
        print(f"Result: {stats_rf['mean_acc']:.4f}")

        all_results.append(
            {
                "dataset": ds_name,
                "experiment": "reference",
                "p_svm": 0.0,
                "n_estimators": fixed_n_estimators,
                "model_type": "RandomForestClassifier",
                **stats_rf,
            }
        )

    if all_results:
        df_results = pd.DataFrame(all_results)

        print("\n\n" + "#" * 80)
        print("AGGREGATED SUMMARY")
        print("#" * 80)

        for ds in datasets_config:
            name = ds["name"]
            print(f"\nResults for: {name}")
            subset = df_results[df_results["dataset"] == name]
            if not subset.empty:
                cols = ["model_type", "p_svm", "mean_acc", "std_acc", "min_acc", "max_acc"]
                print(subset[cols].to_string(index=False))
            else:
                print("(No results - loading error?)")
    else:
        print("\nNo experiments were completed.")


if __name__ == "__main__":
    main()

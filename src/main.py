import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearnex import patch_sklearn
from core.models.HybridSVMForest import HybridSVMForest
from core.utils.DataLoader import DataLoader

patch_sklearn()


def run_single_experiment(X, y, model_class, model_params, n_repeats=25, n_splits=5):
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


def generate_plots(df_results, output_dir="./plots"):
    if df_results.empty:
        return
    os.makedirs(output_dir, exist_ok=True)
    datasets = df_results["dataset"].unique()
    for ds_name in datasets:
        subset = df_results[df_results["dataset"] == ds_name]
        hybrid_data = subset[subset["model_type"] == "HybridSVMForest"].sort_values("p_svm")
        ref_data = subset[subset["model_type"] == "RandomForestClassifier"]
        plt.figure(figsize=(10, 6))
        if not hybrid_data.empty:
            plt.errorbar(
                hybrid_data["p_svm"],
                hybrid_data["mean_acc"],
                yerr=hybrid_data["std_acc"],
                fmt='-o',
                label='Hybrid SVM Forest',
                capsize=5
            )
        if not ref_data.empty:
            ref_acc = ref_data["mean_acc"].iloc[0]
            plt.axhline(y=ref_acc, color='r', linestyle='--', label=f'Random Forest Ref ({ref_acc:.4f})')
        plt.title(f"Impact of SVM Share on Accuracy - Dataset: {ds_name}")
        plt.xlabel("Share of SVM classifiers (p_svm)")
        plt.ylabel("Mean Accuracy")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        safe_name = ds_name.replace(" ", "_").replace("-", "").lower()
        filename = os.path.join(output_dir, f"plot_{safe_name}.png")
        plt.savefig(filename)
        plt.close()


def main():
    datasets_config = [
        {
            "name": "Mushroom Data Set",
            "type": "Discrete",
            "loader": lambda: DataLoader.load_mushroom_data(),
        },
        {
            "name": "Wisconsin Breast Cancer",
            "type": "Continuous Discretized",
            "loader": lambda: DataLoader.load_breast_cancer_data(n_bins=5),
        },
        {
            "name": "Wine Quality - Red",
            "type": "Continuous Discretized",
            "loader": lambda: DataLoader.load_wine_data(n_bins=5),
        },
    ]

    all_results = []
    fixed_n_estimators = 20
    p_svm_values = [0.0, 0.2, 0.5, 0.8, 1.0]

    for ds in datasets_config:
        ds_name = ds["name"]
        try:
            X, y = ds["loader"]()
        except Exception as e:
            print(f"Error loading dataset {ds_name}: {e}")
            continue

        for p in p_svm_values:
            stats = run_single_experiment(
                X,
                y,
                HybridSVMForest,
                {"n_estimators": fixed_n_estimators, "p_svm": p, "C": 1.0},
            )
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

        stats_rf = run_single_experiment(X, y, RandomForestClassifier, {"n_estimators": fixed_n_estimators})
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
        for ds in datasets_config:
            name = ds["name"]
            subset = df_results[df_results["dataset"] == name]
            if not subset.empty:
                cols = ["model_type", "p_svm", "mean_acc", "std_acc", "min_acc", "max_acc"]
                print(f"\nResults for: {name}")
                print(subset[cols].to_string(index=False))
        generate_plots(df_results)
    else:
        print("No experiments completed.")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

from core.models.HybridSVMForest import HybridSVMForest
from core.models.ID3Classifier import ID3Classifier
from core.utils.DataLoader import DataLoader


def run_single_experiment(X_id3, X_svm, y, model_class, model_params, n_repeats=25, n_splits=5):
    """
    Runs a rigorous evaluation with n_repeats of Stratified CV, passing
    a tuple of (X_id3, X_svm) to the model.
    """
    accuracies = []
    base_seed = int(time.time())

    for i in range(n_repeats):
        current_seed = base_seed + i
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=current_seed) 
        
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


def run_verification_experiment(datasets_config):
    """
    Verifies Custom ID3 vs Sklearn Decision Tree (Entropy).
    """
    print("\n" + "="*50)
    print("VERIFICATION EXPERIMENT: Custom ID3 vs Sklearn Tree")
    print("="*50)
    
    results = []
    
    for ds in datasets_config:
        ds_name = ds["name"]
        print(f"Verifying on {ds_name}...")
        try:
            loaded_data = ds["loader"]()
            if ds_name == "Mushroom Data Set":
                X_id3, _, y = loaded_data
            else:
                X_id3, y = loaded_data
            
            X_train, X_test, y_train, y_test = train_test_split(X_id3, y, test_size=0.3, random_state=42, stratify=y)
            
            # 1. Custom ID3
            id3 = ID3Classifier()
            id3.fit(X_train, y_train)
            pred_id3 = id3.predict(X_test)
            acc_id3 = accuracy_score(y_test, pred_id3)
            
            # 2. Sklearn Tree (Reference)
            dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
            dt.fit(X_train, y_train)
            pred_dt = dt.predict(X_test)
            acc_dt = accuracy_score(y_test, pred_dt)
            
            results.append({
                "Dataset": ds_name,
                "Custom ID3 Acc": f"{acc_id3:.4f}",
                "Sklearn Tree Acc": f"{acc_dt:.4f}",
                "Diff": f"{acc_id3 - acc_dt:.4f}"
            })
            
        except Exception as e:
            print(f"Error in verification for {ds_name}: {e}")

    df_ver = pd.DataFrame(results)
    print(df_ver.to_string(index=False))
    print("\nNote: Differences may arise from implementation details (e.g., tie-breaking, handling of continuous variables).")


def generate_generic_plot(df_results, x_col, x_label, title_suffix, output_dir="./plots"):
    if df_results.empty:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    datasets = df_results["dataset"].unique()
    
    for ds_name in datasets:
        subset = df_results[df_results["dataset"] == ds_name]
        subset = subset.sort_values(x_col)
        
        plt.figure(figsize=(10, 6))
        
        hybrid_data = subset[subset["model_type"] == "HybridSVMForest"]
        if not hybrid_data.empty:
            plt.errorbar(
                hybrid_data[x_col],
                hybrid_data["mean_acc"],
                yerr=hybrid_data["std_acc"],
                fmt='-o',
                label='Hybrid SVM Forest',
                capsize=5,
                linewidth=2
            )
            
        plt.title(f"{title_suffix} - {ds_name}")
        plt.xlabel(x_label)
        plt.ylabel("Mean Accuracy (CV)")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        
        safe_name = f"{ds_name}_{x_col}".replace(" ", "_").replace("-", "").lower()
        filename = os.path.join(output_dir, f"exp_{safe_name}.png")
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved: {filename}")


def main():
    datasets_config = [
        {
            "name": "Mushroom Data Set",
            "loader": lambda: DataLoader.load_mushroom_data(),
        },
        {
            "name": "Wisconsin Breast Cancer",
            "loader": lambda: DataLoader.load_breast_cancer_data(n_bins=5),
        },
        {
            "name": "Wine Quality - Red",
            "loader": lambda: DataLoader.load_wine_data(n_bins=5),
        },
    ]

    results_exp_p_svm = []
    results_exp_n_est = []
    results_exp_C = []

    run_verification_experiment(datasets_config)

    print("\n" + "="*50)
    print("STARTING MAIN EXPERIMENTS")
    print("="*50)

    for ds in datasets_config:
        ds_name = ds["name"]
        print(f"\nProcessing Dataset: {ds_name}")
        
        try:
            loaded_data = ds["loader"]()

            if ds_name == "Mushroom Data Set":
                X_id3, X_svm, y = loaded_data
            else:
                X_id3, y = loaded_data
                X_svm = X_id3

        except Exception as e:
            print(f"Skipping {ds_name} due to load error: {e}")
            continue

        print("   Running Exp 1: Impact of p_svm...")
        fixed_n = 20
        fixed_C = 1.0
        p_svm_values = [0.0, 0.2, 0.5, 0.8, 1.0]
        
        for p in p_svm_values:
            stats = run_single_experiment(
                X_id3, X_svm, y, HybridSVMForest,
                {"n_estimators": fixed_n, "p_svm": p, "C": fixed_C}
            )
            results_exp_p_svm.append({
                "dataset": ds_name, "model_type": "HybridSVMForest",
                "p_svm": p, "mean_acc": stats["mean_acc"], "std_acc": stats["std_acc"]
            })

        print("   Running Exp 2: Impact of n_estimators...")
        fixed_p_svm_for_T = 0.5
        n_est_values = [10, 20, 50, 100]
        
        for n in n_est_values:
            stats = run_single_experiment(
                X_id3, X_svm, y, HybridSVMForest,
                {"n_estimators": n, "p_svm": fixed_p_svm_for_T, "C": fixed_C}
            )
            results_exp_n_est.append({
                "dataset": ds_name, "model_type": "HybridSVMForest",
                "n_estimators": n, "mean_acc": stats["mean_acc"], "std_acc": stats["std_acc"]
            })
            
        print("   Running Exp 3: Impact of C (Regularization)...")
        fixed_p_svm_for_C = 1.0
        C_values = [0.1, 1.0, 10.0, 50.0]
        
        for c_val in C_values:
            stats = run_single_experiment(
                X_id3, X_svm, y, HybridSVMForest,
                {"n_estimators": fixed_n, "p_svm": fixed_p_svm_for_C, "C": c_val}
            )
            results_exp_C.append({
                "dataset": ds_name, "model_type": "HybridSVMForest",
                "C": c_val, "mean_acc": stats["mean_acc"], "std_acc": stats["std_acc"]
            })

    if results_exp_p_svm:
        df1 = pd.DataFrame(results_exp_p_svm)
        print("\n--- Results: Impact of p_svm ---")
        print(df1[["dataset", "p_svm", "mean_acc"]].to_string(index=False))
        generate_generic_plot(
            df1, "p_svm", "Share of SVM (p_svm)", 
            "Impact of SVM Share"
        )

    if results_exp_n_est:
        df2 = pd.DataFrame(results_exp_n_est)
        print("\n--- Results: Impact of n_estimators ---")
        print(df2[["dataset", "n_estimators", "mean_acc"]].to_string(index=False))
        generate_generic_plot(
            df2, "n_estimators", "Number of Estimators (T)", 
            "Ensemble Size Stability"
        )

    if results_exp_C:
        df3 = pd.DataFrame(results_exp_C)
        print("\n--- Results: Impact of C parameter ---")
        print(df3[["dataset", "C", "mean_acc"]].to_string(index=False))
        generate_generic_plot(
            df3, "C", "SVM Regularization (C)", 
            "Impact of C parameter"
        )

    print("\nDone. Check the './plots' directory.")


if __name__ == "__main__":
    main()
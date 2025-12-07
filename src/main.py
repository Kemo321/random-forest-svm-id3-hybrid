import pandas as pd
import warnings

# Sklearn optimizations
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

from core.models.HybridSVMForest import HybridSVMForest
from core.utils.DataLoader import DataLoader
from core.experiments.ExperimentRunner import ExperimentRunner
from core.experiments.VerificationRunner import VerificationRunner
from core.utils.Visualizer import Visualizer

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

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

    verifier = VerificationRunner()
    runner = ExperimentRunner(n_repeats=25, n_splits=5)
    visualizer = Visualizer(output_dir="./plots")

    verifier.run(datasets_config)

    print("\n" + "="*50)
    print("STARTING MAIN EXPERIMENTS")
    print("="*50)

    results_exp_p_svm = []
    results_exp_n_est = []
    results_exp_C = []

    for ds in datasets_config:
        ds_name = ds["name"]
        print(f"\nProcessing Dataset: {ds_name}")
        
        try:
            loaded_data = ds["loader"]()
            if len(loaded_data) == 3:
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
        for p in [0.0, 0.2, 0.5, 0.8, 1.0]:
            stats = runner.run_cv(
                X_id3, X_svm, y, HybridSVMForest,
                {"n_estimators": fixed_n, "p_svm": p, "C": fixed_C}
            )
            results_exp_p_svm.append({
                "dataset": ds_name, "model_type": "HybridSVMForest",
                "p_svm": p, **stats
            })

        print("   Running Exp 2: Impact of n_estimators...")
        fixed_p_for_T = 0.5
        for n in [10, 20, 50, 100]:
            stats = runner.run_cv(
                X_id3, X_svm, y, HybridSVMForest,
                {"n_estimators": n, "p_svm": fixed_p_for_T, "C": fixed_C}
            )
            results_exp_n_est.append({
                "dataset": ds_name, "model_type": "HybridSVMForest",
                "n_estimators": n, **stats
            })

        print("   Running Exp 3: Impact of C (Regularization)...")
        fixed_p_for_C = 1.0
        for c_val in [0.1, 1.0, 10.0, 50.0]:
            stats = runner.run_cv(
                X_id3, X_svm, y, HybridSVMForest,
                {"n_estimators": fixed_n, "p_svm": fixed_p_for_C, "C": c_val}
            )
            results_exp_C.append({
                "dataset": ds_name, "model_type": "HybridSVMForest",
                "C": c_val, **stats
            })

    
    if results_exp_p_svm:
        df = pd.DataFrame(results_exp_p_svm)
        print("\n--- Results: Impact of p_svm ---")
        print(df[["dataset", "p_svm", "mean_acc"]].to_string(index=False))
        visualizer.plot_experiment(df, "p_svm", "Share of SVM (p_svm)", "Impact of SVM Share")

    if results_exp_n_est:
        df = pd.DataFrame(results_exp_n_est)
        print("\n--- Results: Impact of n_estimators ---")
        print(df[["dataset", "n_estimators", "mean_acc"]].to_string(index=False))
        visualizer.plot_experiment(df, "n_estimators", "Number of Estimators (T)", "Ensemble Size Stability")

    if results_exp_C:
        df = pd.DataFrame(results_exp_C)
        print("\n--- Results: Impact of C parameter ---")
        print(df[["dataset", "C", "mean_acc"]].to_string(index=False))
        visualizer.plot_experiment(df, "C", "SVM Regularization (C)", "Impact of C parameter")

    print(f"\nDone. Check the '{visualizer.output_dir}' directory.")

if __name__ == "__main__":
    main()

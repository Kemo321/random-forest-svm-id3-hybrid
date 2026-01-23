# Authors: Jan Szwagierczak, Tomasz Okoń

import argparse
import warnings
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Sklearn optimizations
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

from core.models.HybridSVMForest import HybridSVMForest
from core.utils.DataLoader import DataLoader
from core.experiments.DataGenerator import DataGenerator
from core.experiments.PlotGenerator import PlotGenerator
from core.experiments.VerificationRunner import VerificationRunner

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def get_datasets_config(dataset_filter=None):
    all_datasets = [
        {
            "name": "Mushroom Data Set",
            "loader": lambda: DataLoader.load_mushroom_data(),
            "class_labels": ["edible", "poisonous"],
        },
        {
            "name": "Wisconsin Breast Cancer",
            "loader": lambda: DataLoader.load_breast_cancer_data(n_bins=5),
            "class_labels": ["malignant", "benign"],
        },
        {
            "name": "Wine Quality - Red",
            "loader": lambda: DataLoader.load_wine_quality_red_data(n_bins=5),
            "class_labels": ["low", "high"],
        },
        {
            "name": "Car Evaluation",
            "loader": lambda: DataLoader.load_car_data(),
            "class_labels": ["unacc", "acc", "good", "vgood"],
        },
        {
            "name": "Synthetic Diagonal",
            "loader": lambda: DataLoader.load_diagonal_data(n_samples=1000),
            "class_labels": ["below", "above"],
        }
    ]

    if dataset_filter:
        filtered = [ds for ds in all_datasets
                    if dataset_filter.lower() in ds["name"].lower()]
        if not filtered:
            available = [ds["name"] for ds in all_datasets]
            raise ValueError(f"Dataset '{dataset_filter}' not found. Available: {available}")
        return filtered

    return all_datasets


def run_data_generation(datasets_config, results_dir="./results"):
    print("\n" + "=" * 70)
    print("STARTING DATA GENERATION")
    print("=" * 70)

    generator = DataGenerator(
        n_repeats=25,
        n_splits=5,
        results_dir=results_dir
    )

    results = generator.generate_all(datasets_config, HybridSVMForest)

    print("\nGenerated CSV files:")
    for key, df in results.items():
        print(f"  - {key}: {len(df)} rows")

    return results


def run_plot_generation(datasets_config, results_dir="./results", plots_dir="./plots"):
    print("\n" + "=" * 70)
    print("STARTING PLOT GENERATION")
    print("=" * 70)

    class_labels = {}
    for ds in datasets_config:
        if "class_labels" in ds:
            class_labels[ds["name"]] = ds["class_labels"]

    plotter = PlotGenerator(output_dir=plots_dir, results_dir=results_dir)
    results = plotter.generate_all_plots(class_labels=class_labels)

    total_plots = sum(len(v) for v in results.values() if isinstance(v, list))
    print(f"\nGenerated {total_plots} plots in {plots_dir}/")

    return results


def run_verification(datasets_config, results_dir="./results"):
    verifier = VerificationRunner(results_dir=results_dir)
    verifier.run(datasets_config)


def get_default_paths():
    return (
        os.path.join(PROJECT_ROOT, "results"),
        os.path.join(PROJECT_ROOT, "plots")
    )


def main():
    default_results_dir, default_plots_dir = get_default_paths()

    parser = argparse.ArgumentParser(description="Hybrid SVM-Forest Experiments")
    parser.add_argument('--data-only', action='store_true',
                        help='Generate only CSV results, no plots')
    parser.add_argument('--plots-only', action='store_true',
                        help='Generate only plots from existing CSVs')
    parser.add_argument('--verification', action='store_true',
                        help='Run verification experiment only')
    parser.add_argument('--results-dir', type=str, default=default_results_dir,
                        help='Directory for CSV results')
    parser.add_argument('--plots-dir', type=str, default=default_plots_dir,
                        help='Directory for plots')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Run only for specific dataset (partial name match)')

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    datasets_config = get_datasets_config(args.dataset)

    if args.verification:
        run_verification(datasets_config, args.results_dir)
        return

    if args.plots_only:
        run_plot_generation(datasets_config, args.results_dir, args.plots_dir)
        return

    if args.data_only:
        run_data_generation(datasets_config, args.results_dir)
        return

    run_verification(datasets_config, args.results_dir)
    run_data_generation(datasets_config, args.results_dir)
    run_plot_generation(datasets_config, args.results_dir, args.plots_dir)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nResults saved to: {args.results_dir}/")
    print(f"Plots saved to: {args.plots_dir}/")


if __name__ == "__main__":
    main()

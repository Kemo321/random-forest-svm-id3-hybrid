"""
Plot Generator - creates visualizations from CSV results.

This module handles:
- Scenario 1 plots: Accuracy vs p_svm for each dataset
- Scenario 2 plots: Accuracy vs T (estimator count) for each dataset
- Scenario 3 plots: Accuracy vs C for each dataset
- Heatmaps: Confusion matrices for all datasets
- Overfitting: Train vs Test comparison tables/plots
"""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


class PlotGenerator:
    """Generates plots from experimental CSV results."""

    def __init__(self, output_dir: str = "./plots", results_dir: str = "./results") -> None:
        self.output_dir = output_dir
        self.results_dir = results_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Set style for all plots
        plt.style.use('seaborn-v0_8-whitegrid')

        # Formatter for decimal separator (period -> comma)
        self.decimal_formatter = ticker.FuncFormatter(lambda x, p: f'{x:.2f}'.replace('.', ','))

    def _save_figure(self, fig: plt.Figure, filename: str, dpi: int = 150) -> str:
        """Save figure and return the path."""
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {filepath}")
        return filepath

    def plot_scenario1_p_svm(self, csv_path: Optional[str] = None) -> List[str]:
        """
        Generate plots for Scenario 1: Impact of p_svm.
        Creates one plot per dataset showing accuracy vs p_svm.
        """
        print("\nGenerating Scenario 1 plots (p_svm impact)...")

        if csv_path is None:
            csv_path = os.path.join(self.results_dir, "results_impact_p_svm.csv")

        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found")
            return []

        df = pd.read_csv(csv_path)
        datasets = df["dataset"].unique()
        saved_files = []

        for ds_name in datasets:
            subset = df[df["dataset"] == ds_name].sort_values("p_svm")

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.errorbar(
                subset["p_svm"],
                subset["mean_acc"],
                yerr=subset["std_acc"],
                fmt='-o',
                capsize=5,
                linewidth=2,
                markersize=8,
                color='#2196F3',
                ecolor='#90CAF9',
                label='Mean Accuracy ± Std'
            )

            # Add min/max as shaded area
            ax.fill_between(
                subset["p_svm"],
                subset["min_acc"],
                subset["max_acc"],
                alpha=0.2,
                color='#2196F3',
                label='Min-Max Range'
            )

            ax.set_title(f"Impact of p_svm on Accuracy - {ds_name}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Share of SVM Estimators (p_svm)", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.legend(loc='best')
            ax.set_xticks(subset["p_svm"])
            ax.yaxis.set_major_formatter(self.decimal_formatter)
            ax.grid(True, linestyle=':', alpha=0.6)

            safe_name = ds_name.replace(" ", "_").replace("-", "_").lower()
            filename = f"scenario1_p_svm_{safe_name}.png"
            saved_files.append(self._save_figure(fig, filename))

        return saved_files

    def plot_scenario2_estimator_count(self, csv_path: Optional[str] = None) -> List[str]:
        """
        Generate plots for Scenario 2: Impact of T (estimator count).
        Creates one plot per dataset showing accuracy vs T.
        """
        print("\nGenerating Scenario 2 plots (T impact)...")

        if csv_path is None:
            csv_path = os.path.join(self.results_dir, "results_impact_estimator_count.csv")

        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found")
            return []

        df = pd.read_csv(csv_path)
        datasets = df["dataset"].unique()
        saved_files = []

        for ds_name in datasets:
            subset = df[df["dataset"] == ds_name].sort_values("estimator_count")

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.errorbar(
                subset["estimator_count"],
                subset["mean_acc"],
                yerr=subset["std_acc"],
                fmt='-s',
                capsize=5,
                linewidth=2,
                markersize=8,
                color='#4CAF50',
                ecolor='#A5D6A7',
                label='Mean Accuracy ± Std'
            )

            ax.fill_between(
                subset["estimator_count"],
                subset["min_acc"],
                subset["max_acc"],
                alpha=0.2,
                color='#4CAF50',
                label='Min-Max Range'
            )

            ax.set_title(f"Impact of Estimator Count on Accuracy - {ds_name}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Number of Estimators (T)", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.legend(loc='best')
            ax.set_xticks(subset["estimator_count"])
            ax.yaxis.set_major_formatter(self.decimal_formatter)
            ax.grid(True, linestyle=':', alpha=0.6)

            safe_name = ds_name.replace(" ", "_").replace("-", "_").lower()
            filename = f"scenario2_estimator_count_{safe_name}.png"
            saved_files.append(self._save_figure(fig, filename))

        return saved_files

    def plot_scenario3_C(self, csv_path: Optional[str] = None) -> List[str]:
        """
        Generate plots for Scenario 3: Impact of C (SVM regularization).
        Creates one plot per dataset showing accuracy vs C.
        """
        print("\nGenerating Scenario 3 plots (C impact)...")

        if csv_path is None:
            csv_path = os.path.join(self.results_dir, "results_impact_C.csv")

        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found")
            return []

        df = pd.read_csv(csv_path)
        datasets = df["dataset"].unique()
        saved_files = []

        for ds_name in datasets:
            subset = df[df["dataset"] == ds_name].sort_values("C")

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.errorbar(
                subset["C"],
                subset["mean_acc"],
                yerr=subset["std_acc"],
                fmt='-^',
                capsize=5,
                linewidth=2,
                markersize=8,
                color='#FF5722',
                ecolor='#FFAB91',
                label='Mean Accuracy ± Std'
            )

            ax.fill_between(
                subset["C"],
                subset["min_acc"],
                subset["max_acc"],
                alpha=0.2,
                color='#FF5722',
                label='Min-Max Range'
            )

            ax.set_title(f"Impact of SVM Regularization (C) on Accuracy - {ds_name}", fontsize=14, fontweight='bold')
            ax.set_xlabel("SVM Regularization Parameter (C)", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.legend(loc='best')
            ax.set_xscale('log')
            ax.yaxis.set_major_formatter(self.decimal_formatter)
            ax.grid(True, linestyle=':', alpha=0.6)

            safe_name = ds_name.replace(" ", "_").replace("-", "_").lower()
            filename = f"scenario3_C_{safe_name}.png"
            saved_files.append(self._save_figure(fig, filename))

        return saved_files

    def plot_confusion_matrices(
        self,
        class_labels: Optional[Dict[str, List[str]]] = None
    ) -> List[str]:
        """
        Generate heatmaps for confusion matrices.
        Creates one heatmap per dataset from saved CSV files.

        Args:
            class_labels: Optional dict mapping dataset name to list of class labels
        """
        print("\nGenerating confusion matrix heatmaps...")

        cm_dir = os.path.join(self.results_dir, "confusion_matrices")
        if not os.path.exists(cm_dir):
            print(f"  Warning: {cm_dir} not found")
            return []

        saved_files = []

        # Find all hybrid confusion matrix files
        cm_files = [f for f in os.listdir(cm_dir) if f.startswith("cm_hybrid_") and f.endswith(".csv")]

        for cm_file in cm_files:
            ds_name_clean = cm_file.replace("cm_hybrid_", "").replace(".csv", "")
            ds_name = ds_name_clean.replace("_", " ")

            cm_path = os.path.join(cm_dir, cm_file)
            cm = np.loadtxt(cm_path, delimiter=',', dtype=int)

            fig, ax = plt.subplots(figsize=(8, 6))

            # Get class labels if provided
            labels = None
            if class_labels and ds_name in class_labels:
                labels = class_labels[ds_name]

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax,
                xticklabels=labels if labels else 'auto',
                yticklabels=labels if labels else 'auto',
                cbar_kws={'label': 'Count'}
            )

            ax.set_title(f"Confusion Matrix - {ds_name}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Predicted Label", fontsize=12)
            ax.set_ylabel("True Label", fontsize=12)

            filename = f"heatmap_{ds_name_clean}.png"
            saved_files.append(self._save_figure(fig, filename))

        return saved_files

    def plot_confusion_matrices_grid(
        self,
        class_labels: Optional[Dict[str, List[str]]] = None
    ) -> str:
        """
        Generate 2x2 grid of confusion matrix heatmaps for all datasets.
        """
        print("\nGenerating confusion matrices grid (2x2)...")

        cm_dir = os.path.join(self.results_dir, "confusion_matrices")
        if not os.path.exists(cm_dir):
            print(f"  Warning: {cm_dir} not found")
            return ""

        # Find all hybrid confusion matrix files
        cm_files = sorted([f for f in os.listdir(cm_dir) if f.startswith("cm_hybrid_") and f.endswith(".csv")])

        if len(cm_files) < 4:
            print(f"  Warning: Expected 4 confusion matrices, found {len(cm_files)}")
            if len(cm_files) == 0:
                return ""

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, cm_file in enumerate(cm_files[:4]):
            ds_name_clean = cm_file.replace("cm_hybrid_", "").replace(".csv", "")
            ds_name = ds_name_clean.replace("_", " ")

            cm_path = os.path.join(cm_dir, cm_file)
            cm = np.loadtxt(cm_path, delimiter=',', dtype=int)

            labels = None
            if class_labels and ds_name in class_labels:
                labels = class_labels[ds_name]

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=axes[idx],
                xticklabels=labels if labels else 'auto',
                yticklabels=labels if labels else 'auto',
                cbar_kws={'label': 'Count'}
            )

            axes[idx].set_title(ds_name, fontsize=12, fontweight='bold')
            axes[idx].set_xlabel("Predicted", fontsize=10)
            axes[idx].set_ylabel("True", fontsize=10)

        fig.suptitle("Confusion Matrices for All Datasets\n(T=20, p_svm=0.5, C=1.0)",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        filename = "heatmaps_grid_2x2.png"
        return self._save_figure(fig, filename)

    def plot_overfitting_analysis(self, csv_path: Optional[str] = None) -> List[str]:
        """
        Generate overfitting analysis visualization.
        Creates bar plot comparing Train vs Test accuracy for each dataset.
        """
        print("\nGenerating overfitting analysis plot...")

        if csv_path is None:
            csv_path = os.path.join(self.results_dir, "results_overfitting.csv")

        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found")
            return []

        df = pd.read_csv(csv_path)
        saved_files = []

        # Bar plot comparing train vs test
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(df))
        width = 0.35

        bars1 = ax.bar(x - width/2, df["train_acc_mean"], width,
                       yerr=df["train_acc_std"], label='Train',
                       color='#2196F3', capsize=5, alpha=0.8)
        bars2 = ax.bar(x + width/2, df["test_acc_mean"], width,
                       yerr=df["test_acc_std"], label='Test',
                       color='#FF5722', capsize=5, alpha=0.8)

        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Overfitting Analysis: Train vs Test Accuracy\n(T=20, p_svm=0.5, C=1.0)",
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df["dataset"], rotation=15, ha='right')
        ax.yaxis.set_major_formatter(self.decimal_formatter)
        ax.legend()
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)

        # Add delta values as text
        for i, (train, test, delta) in enumerate(zip(df["train_acc_mean"], df["test_acc_mean"], df["delta"])):
            max_y = max(train, test)
            ax.annotate(f'Δ={delta:.3f}'.replace('.', ','), xy=(i, max_y + 0.02),
                       ha='center', fontsize=9, color='gray')

        plt.tight_layout()
        saved_files.append(self._save_figure(fig, "overfitting_train_vs_test.png"))

        # Also create individual plots per dataset
        for _, row in df.iterrows():
            ds_name = row["dataset"]

            fig, ax = plt.subplots(figsize=(8, 5))

            categories = ['Train', 'Test']
            means = [row["train_acc_mean"], row["test_acc_mean"]]
            stds = [row["train_acc_std"], row["test_acc_std"]]
            colors = ['#2196F3', '#FF5722']

            bars = ax.bar(categories, means, yerr=stds, capsize=10,
                         color=colors, alpha=0.8, edgecolor='black')

            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title(f"Train vs Test Accuracy - {ds_name}\n(Δ = {row['delta']:.4f})".replace('.', ','),
                        fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.yaxis.set_major_formatter(self.decimal_formatter)
            ax.grid(True, axis='y', linestyle=':', alpha=0.6)

            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                ax.annotate(f'{mean:.4f}\n±{std:.4f}'.replace('.', ','),
                           xy=(bar.get_x() + bar.get_width()/2, mean),
                           ha='center', va='bottom', fontsize=10)

            safe_name = ds_name.replace(" ", "_").replace("-", "_").lower()
            filename = f"overfitting_{safe_name}.png"
            saved_files.append(self._save_figure(fig, filename))

        return saved_files

    def generate_all_plots(self, class_labels: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
        """
        Generate all plots from existing CSV files.

        Returns dict mapping plot category to list of saved file paths.
        """
        print("\n" + "=" * 70)
        print("PLOT GENERATION - Creating all visualizations")
        print("=" * 70)

        results = {}

        results['scenario1'] = self.plot_scenario1_p_svm()
        results['scenario2'] = self.plot_scenario2_estimator_count()
        results['scenario3'] = self.plot_scenario3_C()
        results['heatmaps'] = self.plot_confusion_matrices(class_labels)
        results['heatmaps_grid'] = [self.plot_confusion_matrices_grid(class_labels)]
        results['overfitting'] = self.plot_overfitting_analysis()

        print("\n" + "=" * 70)
        print("PLOT GENERATION COMPLETE")
        print("=" * 70)

        return results

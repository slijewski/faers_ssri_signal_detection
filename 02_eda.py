import logging
"""
==============================================================================
02_eda.py — Exploratory Data Analysis of SSRI Adverse Event Profiles
==============================================================================

This script performs comprehensive EDA on the FAERS data collected in Step 01.
It generates publication-quality visualizations that reveal the adverse event
landscape of SSRI antidepressants.

Visualizations Generated:
    1. Top 15 adverse events per SSRI (faceted horizontal bar charts)
    2. Suicidality-related term comparison across all SSRIs (grouped bars)
    3. Drug × Reaction heatmap (top 30 shared reactions)
    4. Total report volume comparison (proportional bar chart)

Prerequisites:
    Run 01_data_collection.py first to generate the data/ directory.

Output:
    All figures are saved to the outputs/ directory as PNG files.

Author: Sebastian Lijewski, PhD
==============================================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_context("talk", font_scale=0.9)

DRUG_COLORS = {
    "fluoxetine":   "#E63946",
    "sertraline":   "#457B9D",
    "paroxetine":   "#2A9D8F",
    "escitalopram": "#E9C46A",
    "citalopram":   "#F4A261",
    "fluvoxamine":  "#264653",
}

SUICIDALITY_TERMS = [
    "Suicidal ideation",
    "Suicide attempt",
    "Completed suicide",
    "Self-injurious ideation",
    "Intentional self-injury",
    "Suicidal behaviour",
    "Depression suicidal",
]



def load_data() -> tuple:
    """
    Load all CSV files produced by 01_data_collection.py.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        reaction_counts, totals, background DataFrames.
    """
    df_reactions = pd.read_csv(os.path.join(DATA_DIR, "ssri_reaction_counts.csv"))
    df_totals = pd.read_csv(os.path.join(DATA_DIR, "ssri_totals.csv"))
    df_background = pd.read_csv(os.path.join(DATA_DIR, "background_totals.csv"))

    logging.info(f"Loaded {len(df_reactions):,} reaction records "
          f"across {df_reactions['generic_name'].nunique()} SSRIs")
    logging.info(f"Total reports per drug:\n{df_totals[['display_name', 'total_reports']].to_string(index=False)}\n")

    return df_reactions, df_totals, df_background



def plot_total_reports(df_totals: pd.DataFrame):
    """
    Bar chart comparing total FAERS report volume per SSRI.

    This contextualizes the signal detection results: drugs with more reports
    are not necessarily more dangerous — they may simply be prescribed more.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    df_sorted = df_totals.sort_values("total_reports", ascending=True)

    colors = [DRUG_COLORS[name] for name in df_sorted["generic_name"]]
    bars = ax.barh(
        df_sorted["display_name"],
        df_sorted["total_reports"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, value in zip(bars, df_sorted["total_reports"]):
        ax.text(
            bar.get_width() + max(df_sorted["total_reports"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.0f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Total Adverse Event Reports in FAERS", fontsize=12)
    ax.set_title(
        "FAERS Reporting Volume by SSRI Drug",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_total_reports_by_ssri.png"), dpi=150)
    plt.close()
    logging.info("  ✓ Saved: 01_total_reports_by_ssri.png")


def plot_top_reactions_per_drug(df_reactions: pd.DataFrame, top_n: int = 15):
    """
    Faceted horizontal bar charts showing the top N adverse events per SSRI.

    Each subplot represents one drug, allowing side-by-side comparison
    of the most frequently reported adverse events.
    """
    drugs = sorted(df_reactions["generic_name"].unique())
    n_drugs = len(drugs)
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()

    for idx, drug in enumerate(drugs):
        ax = axes[idx]
        df_drug = (
            df_reactions[df_reactions["generic_name"] == drug]
            .nlargest(top_n, "count")
            .sort_values("count", ascending=True)
        )

        color = DRUG_COLORS[drug]
        display_name = df_drug["display_name"].iloc[0]

        ax.barh(
            df_drug["reaction_term"],
            df_drug["count"],
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.set_title(display_name, fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )

    for idx in range(n_drugs, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(
        f"Top {top_n} Adverse Events per SSRI (FAERS Database)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "02_top_reactions_per_ssri.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    logging.info("  ✓ Saved: 02_top_reactions_per_ssri.png")


def plot_suicidality_comparison(df_reactions: pd.DataFrame):
    """
    Grouped bar chart comparing suicidality-related adverse event counts
    across all SSRIs.

    This is the key clinical visualization — it directly addresses the
    research question about differential suicidality signaling.
    """
    df_suicidal = df_reactions[
        df_reactions["reaction_term"].isin(SUICIDALITY_TERMS)
    ].copy()

    if df_suicidal.empty:
        logging.info("  ⚠ No suicidality terms found in the data. Skipping plot.")
        return

    df_pivot = df_suicidal.pivot_table(
        index="reaction_term",
        columns="display_name",
        values="count",
        fill_value=0,
    )

    df_pivot["_total"] = df_pivot.sum(axis=1)
    df_pivot = df_pivot.sort_values("_total", ascending=False).drop("_total", axis=1)

    fig, ax = plt.subplots(figsize=(14, 8))

    n_terms = len(df_pivot)
    n_drugs = len(df_pivot.columns)
    bar_width = 0.12
    x = np.arange(n_terms)

    drug_order = list(DRUG_COLORS.keys())
    display_order = [
        col for drug in drug_order
        for col in df_pivot.columns if drug in col.lower()
    ]

    for i, drug_col in enumerate(display_order):
        if drug_col in df_pivot.columns:
            generic = [k for k, v in
                       {d: n for d, n in zip(DRUG_COLORS.keys(),
                        [c for c in display_order])}
                       .items() if v == drug_col]
            generic_name = drug_col.split(" (")[0].lower()
            color = DRUG_COLORS.get(generic_name, "#999999")
            ax.bar(
                x + i * bar_width,
                df_pivot[drug_col],
                bar_width,
                label=drug_col,
                color=color,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )

    ax.set_xticks(x + bar_width * (n_drugs - 1) / 2)
    ax.set_xticklabels(df_pivot.index, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Number of FAERS Reports", fontsize=12)
    ax.set_title(
        "Suicidality-Related Adverse Events by SSRI Drug\n"
        "(FDA FAERS Database)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.legend(
        title="SSRI Drug",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=9,
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "03_suicidality_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    logging.info("  ✓ Saved: 03_suicidality_comparison.png")


def plot_reaction_heatmap(df_reactions: pd.DataFrame, top_n: int = 30):
    """
    Heatmap showing the relationship between drugs and their most common
    adverse events (log-scaled).

    This reveals patterns: which drugs share similar adverse event profiles,
    and which have unique signatures.
    """
    top_reactions = (
        df_reactions.groupby("reaction_term")["count"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )

    df_top = df_reactions[df_reactions["reaction_term"].isin(top_reactions)]

    heatmap_data = df_top.pivot_table(
        index="reaction_term",
        columns="display_name",
        values="count",
        fill_value=0,
    )

    heatmap_data["_total"] = heatmap_data.sum(axis=1)
    heatmap_data = heatmap_data.sort_values("_total", ascending=False).drop(
        "_total", axis=1
    )

    heatmap_log = np.log10(heatmap_data.replace(0, 1))

    fig, ax = plt.subplots(figsize=(12, 14))
    sns.heatmap(
        heatmap_log,
        cmap="YlOrRd",
        annot=heatmap_data.values,
        fmt=".0f",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "log₁₀(Count)", "shrink": 0.6},
        annot_kws={"fontsize": 8},
    )

    ax.set_title(
        f"Top {top_n} Adverse Events × SSRI Drug\n"
        "Heatmap of FAERS Report Counts",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=10, rotation=30)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "04_reaction_heatmap.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    logging.info("  ✓ Saved: 04_reaction_heatmap.png")



def main():
    """Execute the full EDA pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info("=" * 70)
    logging.info("EXPLORATORY DATA ANALYSIS — SSRI Adverse Events (FAERS)")
    logging.info("=" * 70)

    df_reactions, df_totals, df_background = load_data()

    logging.info("\nGenerating visualizations...\n")
    plot_total_reports(df_totals)
    plot_top_reactions_per_drug(df_reactions)
    plot_suicidality_comparison(df_reactions)
    plot_reaction_heatmap(df_reactions)

    logging.info(f"\n{'=' * 70}")
    logging.info("EDA COMPLETE — All figures saved to: outputs/")
    logging.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()

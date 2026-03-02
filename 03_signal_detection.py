import logging
"""
Pharmacovigilance signal detection for SSRI suicidality using
PRR, ROR, and Chi-squared metrics.
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

PRR_THRESHOLD = 2.0
CHI2_THRESHOLD = 4.0
MIN_REPORTS = 3

SUICIDALITY_TERMS = [
    "SUICIDAL IDEATION",
    "SUICIDE ATTEMPT",
    "COMPLETED SUICIDE",
    "SELF-INJURIOUS IDEATION",
    "INTENTIONAL SELF-INJURY",
    "SUICIDAL BEHAVIOUR",
    "DEPRESSION SUICIDAL",
]

SIGNAL_COLORS = {
    "SIGNAL": "#E63946",
    "NO SIGNAL": "#457B9D",
    "INSUFFICIENT DATA": "#ADB5BD",
}

DRUG_COLORS = {
    "fluoxetine":   "#E63946",
    "sertraline":   "#457B9D",
    "paroxetine":   "#2A9D8F",
    "escitalopram": "#E9C46A",
    "citalopram":   "#F4A261",
    "fluvoxamine":  "#264653",
}



def calculate_prr(a: int, b: int, c: int, d: int) -> dict:
    """
    Calculate the Proportional Reporting Ratio (PRR) and its 95% CI.

    The PRR compares the proportion of a specific adverse event among all
    reports for a drug of interest against the proportion of that event
    among all reports for all other drugs.

    Formula:
        PRR = [a / (a + b)] / [c / (c + d)]

    95% CI (log-normal):
        ln(PRR) ± 1.96 × √(1/a - 1/(a+b) + 1/c - 1/(c+d))

    Parameters
    ----------
    a : int — Drug + Reaction of interest
    b : int — Drug + Other reactions
    c : int — Other drugs + Reaction of interest
    d : int — Other drugs + Other reactions

    Returns
    -------
    dict with keys: 'prr', 'prr_lower', 'prr_upper'
    """
    if a == 0 or (a + b) == 0 or c == 0 or (c + d) == 0:
        return {"prr": 0.0, "prr_lower": 0.0, "prr_upper": 0.0}

    prr = (a / (a + b)) / (c / (c + d))

    try:
        se_ln = math.sqrt(1/a - 1/(a + b) + 1/c - 1/(c + d))
        ln_prr = math.log(prr)
        prr_lower = math.exp(ln_prr - 1.96 * se_ln)
        prr_upper = math.exp(ln_prr + 1.96 * se_ln)
    except (ValueError, ZeroDivisionError):
        prr_lower = 0.0
        prr_upper = 0.0

    return {"prr": prr, "prr_lower": prr_lower, "prr_upper": prr_upper}


def calculate_ror(a: int, b: int, c: int, d: int) -> dict:
    """
    Calculate the Reporting Odds Ratio (ROR) and its 95% CI.

    The ROR is conceptually similar to an odds ratio from epidemiology.
    It compares the odds of a specific reaction being reported for a drug
    versus the odds for all other drugs.

    Formula:
        ROR = (a × d) / (b × c)

    95% CI (log-normal):
        ln(ROR) ± 1.96 × √(1/a + 1/b + 1/c + 1/d)

    Parameters
    ----------
    a, b, c, d : int — values from the 2×2 contingency table

    Returns
    -------
    dict with keys: 'ror', 'ror_lower', 'ror_upper'
    """
    if a == 0 or b == 0 or c == 0 or d == 0:
        return {"ror": 0.0, "ror_lower": 0.0, "ror_upper": 0.0}

    ror = (a * d) / (b * c)

    try:
        se_ln = math.sqrt(1/a + 1/b + 1/c + 1/d)
        ln_ror = math.log(ror)
        ror_lower = math.exp(ln_ror - 1.96 * se_ln)
        ror_upper = math.exp(ln_ror + 1.96 * se_ln)
    except (ValueError, ZeroDivisionError):
        ror_lower = 0.0
        ror_upper = 0.0

    return {"ror": ror, "ror_lower": ror_lower, "ror_upper": ror_upper}


def calculate_chi_squared(a: int, b: int, c: int, d: int) -> float:
    """
    Calculate Yates-corrected χ² for a 2×2 contingency table.

    Yates' correction is applied because we are dealing with a single 2×2
    table, and it reduces the tendency to reject H₀ too aggressively
    with small sample sizes.

    Formula:
        χ² = N × (|ad - bc| - N/2)² / [(a+b)(c+d)(a+c)(b+d)]

    Parameters
    ----------
    a, b, c, d : int — values from the 2×2 table

    Returns
    -------
    float
        Yates-corrected χ² statistic. Values ≥ 4.0 ≈ p < 0.05.
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    n = a + b + c + d
    numerator = n * (abs(a * d - b * c) - n / 2) ** 2
    denominator = (a + b) * (c + d) * (a + c) * (b + d)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def classify_signal(prr: float, chi2: float, n_cases: int) -> str:
    """
    Classify whether a drug-event combination meets signal criteria.

    Evans et al. (2001) criteria:
    - PRR ≥ 2.0
    - χ² ≥ 4.0
    - N ≥ 3

    Parameters
    ----------
    prr : float — Proportional Reporting Ratio
    chi2 : float — Chi-squared statistic
    n_cases : int — Number of reports (cell 'a')

    Returns
    -------
    str — 'SIGNAL', 'NO SIGNAL', or 'INSUFFICIENT DATA'
    """
    if n_cases < MIN_REPORTS:
        return "INSUFFICIENT DATA"
    if prr >= PRR_THRESHOLD and chi2 >= CHI2_THRESHOLD:
        return "SIGNAL"
    return "NO SIGNAL"



def run_signal_detection() -> pd.DataFrame:
    """
    Execute the full disproportionality analysis for all SSRI × suicidality
    term combinations.

    Constructs the 2×2 contingency table for each (drug, reaction) pair
    using the count data from openFDA, then calculates PRR, ROR, and χ².

    Returns
    -------
    pd.DataFrame
        Results with columns: drug, reaction, a, b, c, d, PRR, ROR, χ², signal.
    """
    df_reactions = pd.read_csv(os.path.join(DATA_DIR, "ssri_reaction_counts.csv"))
    df_totals = pd.read_csv(os.path.join(DATA_DIR, "ssri_totals.csv"))
    df_background = pd.read_csv(os.path.join(DATA_DIR, "background_totals.csv"))

    total_faers = df_background["total_faers_reports"].iloc[0]

    drug_totals = dict(zip(df_totals["generic_name"], df_totals["total_reports"]))

    reaction_lookup = {}
    for _, row in df_reactions.iterrows():
        key = (row["generic_name"], row["reaction_term"].upper())
        reaction_lookup[key] = row["count"]

    bg_lookup = {
        term.upper(): count
        for term, count in zip(df_background["reaction_term"],
                               df_background["background_count"])
    }

    results = []

    for drug in drug_totals.keys():
        drug_total = drug_totals[drug]
        display_name = df_totals.loc[
            df_totals["generic_name"] == drug, "display_name"
        ].iloc[0]

        for reaction in SUICIDALITY_TERMS:
            a = reaction_lookup.get((drug, reaction), 0)

            b = drug_total - a

            total_reaction = bg_lookup.get(reaction, 0)
            c = total_reaction - a

            d = total_faers - a - b - c

            a, b, c, d = max(a, 0), max(b, 0), max(c, 0), max(d, 0)

            prr_result = calculate_prr(a, b, c, d)
            ror_result = calculate_ror(a, b, c, d)
            chi2 = calculate_chi_squared(a, b, c, d)
            signal_status = classify_signal(prr_result["prr"], chi2, a)

            results.append({
                "generic_name": drug,
                "display_name": display_name,
                "reaction_term": reaction,
                "n_cases (a)": a,
                "n_drug_other (b)": b,
                "n_other_reaction (c)": c,
                "n_other_other (d)": d,
                "PRR": round(prr_result["prr"], 3),
                "PRR_lower_95CI": round(prr_result["prr_lower"], 3),
                "PRR_upper_95CI": round(prr_result["prr_upper"], 3),
                "ROR": round(ror_result["ror"], 3),
                "ROR_lower_95CI": round(ror_result["ror_lower"], 3),
                "ROR_upper_95CI": round(ror_result["ror_upper"], 3),
                "Chi_squared": round(chi2, 2),
                "Signal_Status": signal_status,
            })

    df_results = pd.DataFrame(results)

    df_results.to_csv(
        os.path.join(OUTPUT_DIR, "signal_detection_results.csv"), index=False
    )

    return df_results



def plot_forest_prr(df: pd.DataFrame):
    """
    Forest plot of PRR values with 95% CI for 'Suicidal ideation' across SSRIs.

    Forest plots are the standard visualization in pharmacoepidemiology
    for comparing effect measures across drugs.
    """
    df_si = df[df["reaction_term"] == "SUICIDAL IDEATION"].copy()
    df_si = df_si.sort_values("PRR", ascending=True)

    if df_si.empty:
        logging.info("  ⚠ No 'Suicidal ideation' data. Skipping forest plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions = range(len(df_si))
    colors = [DRUG_COLORS.get(name, "#999") for name in df_si["generic_name"]]

    for i, (_, row) in enumerate(df_si.iterrows()):
        ax.plot(
            [row["PRR_lower_95CI"], row["PRR_upper_95CI"]],
            [i, i],
            color=colors[i],
            linewidth=2,
            solid_capstyle="round",
        )
        ax.scatter(
            row["PRR"],
            i,
            color=colors[i],
            s=120,
            zorder=5,
            marker="D",
            edgecolor="white",
            linewidth=1,
        )
        ax.text(
            row["PRR_upper_95CI"] + 0.05,
            i,
            f'{row["PRR"]:.2f} [{row["PRR_lower_95CI"]:.2f}–{row["PRR_upper_95CI"]:.2f}]',
            va="center",
            fontsize=9,
        )

    ax.axvline(x=1, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="PRR = 1 (no signal)")
    ax.axvline(x=PRR_THRESHOLD, color="#E63946", linestyle=":", linewidth=1.5,
               alpha=0.7, label=f"PRR = {PRR_THRESHOLD} (signal threshold)")

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(df_si["display_name"], fontsize=11)
    ax.set_xlabel("Proportional Reporting Ratio (PRR) with 95% CI", fontsize=12)
    ax.set_title(
        'PRR Forest Plot: "Suicidal Ideation" Signal by SSRI\n'
        "(Evans et al. criteria: PRR ≥ 2, χ² ≥ 4, N ≥ 3)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "05_prr_forest_plot.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    logging.info("  ✓ Saved: 05_prr_forest_plot.png")


def plot_forest_ror(df: pd.DataFrame):
    """
    Forest plot of ROR values with 95% CI for 'Suicidal ideation'.
    """
    df_si = df[df["reaction_term"] == "SUICIDAL IDEATION"].copy()
    df_si = df_si.sort_values("ROR", ascending=True)

    if df_si.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    y_positions = range(len(df_si))
    colors = [DRUG_COLORS.get(name, "#999") for name in df_si["generic_name"]]

    for i, (_, row) in enumerate(df_si.iterrows()):
        ax.plot(
            [row["ROR_lower_95CI"], row["ROR_upper_95CI"]],
            [i, i],
            color=colors[i],
            linewidth=2,
            solid_capstyle="round",
        )
        ax.scatter(
            row["ROR"], i, color=colors[i], s=120, zorder=5,
            marker="D", edgecolor="white", linewidth=1,
        )
        ax.text(
            row["ROR_upper_95CI"] + 0.05, i,
            f'{row["ROR"]:.2f} [{row["ROR_lower_95CI"]:.2f}–{row["ROR_upper_95CI"]:.2f}]',
            va="center", fontsize=9,
        )

    ax.axvline(x=1, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="ROR = 1 (null)")
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(df_si["display_name"], fontsize=11)
    ax.set_xlabel("Reporting Odds Ratio (ROR) with 95% CI", fontsize=12)
    ax.set_title(
        'ROR Forest Plot: "Suicidal Ideation" by SSRI\n'
        "(FDA FAERS Database)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "06_ror_forest_plot.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    logging.info("  ✓ Saved: 06_ror_forest_plot.png")


def plot_signal_summary(df: pd.DataFrame):
    """
    Summary heatmap-style table showing signal status for all drug×reaction
    combinations with color coding.
    """
    df_pivot = df.pivot_table(
        index="display_name",
        columns="reaction_term",
        values="PRR",
        aggfunc="first",
    )

    signal_pivot = df.pivot_table(
        index="display_name",
        columns="reaction_term",
        values="Signal_Status",
        aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(16, 8))

    signal_numeric = signal_pivot.replace(
        {"SIGNAL": 2, "NO SIGNAL": 1, "INSUFFICIENT DATA": 0}
    ).astype(float)

    cmap = plt.cm.colors.ListedColormap(["#ADB5BD", "#457B9D", "#E63946"])

    sns.heatmap(
        signal_numeric,
        cmap=cmap,
        annot=df_pivot.values,
        fmt=".2f",
        linewidths=1,
        linecolor="white",
        ax=ax,
        cbar=False,
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        vmin=0,
        vmax=2,
    )

    legend_elements = [
        mpatches.Patch(color="#E63946", label="SIGNAL (PRR≥2, χ²≥4, N≥3)"),
        mpatches.Patch(color="#457B9D", label="NO SIGNAL"),
        mpatches.Patch(color="#ADB5BD", label="INSUFFICIENT DATA"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=10,
    )

    ax.set_title(
        "Pharmacovigilance Signal Detection Summary\n"
        "SSRI × Suicidality-Related Adverse Events (PRR values shown)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", labelsize=11)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "07_signal_summary_table.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    logging.info("  ✓ Saved: 07_signal_summary_table.png")


def print_key_findings(df: pd.DataFrame):
    """Print a formatted summary of key findings to the console."""
    logging.info("\n" + "=" * 70)
    logging.info("KEY FINDINGS — Suicidal Ideation Signal Detection")
    logging.info("=" * 70)

    df_si = df[df["reaction_term"] == "SUICIDAL IDEATION"].sort_values(
        "PRR", ascending=False
    )

    for _, row in df_si.iterrows():
        status_icon = "🔴" if row["Signal_Status"] == "SIGNAL" else "🟢"
        logging.info(f"\n  {status_icon} {row['display_name']}")
        logging.info(f"     PRR = {row['PRR']:.3f} "
              f"[{row['PRR_lower_95CI']:.3f}–{row['PRR_upper_95CI']:.3f}]")
        logging.info(f"     ROR = {row['ROR']:.3f} "
              f"[{row['ROR_lower_95CI']:.3f}–{row['ROR_upper_95CI']:.3f}]")
        logging.info(f"     χ²  = {row['Chi_squared']:.1f}  |  N = {row['n_cases (a)']}")
        logging.info(f"     Status: {row['Signal_Status']}")

    signals = df_si[df_si["Signal_Status"] == "SIGNAL"]
    logging.info(f"\n{'=' * 70}")
    if not signals.empty:
        top_signal = signals.iloc[0]
        logging.info(f"⚠  {len(signals)} SSRI(s) meet Evans et al. signal criteria "
              f"for suicidal ideation.")
        logging.info(f"   Strongest signal: {top_signal['display_name']} "
              f"(PRR = {top_signal['PRR']:.3f})")
    else:
        logging.info("✓  No SSRIs met all three Evans et al. signal criteria "
              "for suicidal ideation.")
    logging.info(f"{'=' * 70}")



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info("=" * 70)
    logging.info("SIGNAL DETECTION — Disproportionality Analysis")
    logging.info("SSRIs × Suicidality-Related Adverse Events")
    logging.info("=" * 70)

    df_results = run_signal_detection()

    logging.info(f"\nCalculated PRR/ROR for {len(df_results)} drug-event combinations.")
    logging.info(f"Results saved: outputs/signal_detection_results.csv\n")

    logging.info("Generating visualizations...\n")
    plot_forest_prr(df_results)
    plot_forest_ror(df_results)
    plot_signal_summary(df_results)

    print_key_findings(df_results)


if __name__ == "__main__":
    main()

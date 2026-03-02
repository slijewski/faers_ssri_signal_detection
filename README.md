# 💊 SSRI Suicidality Signal Detection (FDA FAERS)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Data: openFDA](https://img.shields.io/badge/Data-openFDA-0071BC.svg)](https://open.fda.gov/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

## Overview

This project applies **quantitative pharmacovigilance methods** to the FDA Adverse Event Reporting System (FAERS) to investigate the disproportionality of suicidality-related adverse events across six Selective Serotonin Reuptake Inhibitors (SSRIs).

**Research Question:**
> *Which SSRI drugs show the strongest disproportionality signal for suicidal ideation in the FDA FAERS database?*

## 🧬 Scientific Background

The FDA issued a **Black Box Warning** in 2004 linking SSRIs to increased suicidality in pediatric patients, later extended to young adults (18–24). However, the relative risk across individual SSRIs remains a subject of ongoing pharmacovigilance surveillance.

**Disproportionality Analysis (DA)** is the standard quantitative method used by regulatory agencies and pharmaceutical companies to detect potential safety **signals** in spontaneous reporting databases. A *signal* does not establish causation — it identifies drug–event combinations that are reported more frequently than expected, warranting further investigation.

## 🛠️ Methodology

### Data Source

All data is sourced from the **openFDA Drug Adverse Event API** (`api.fda.gov/drug/event.json`), which provides access to millions of adverse event reports submitted to the FDA.

### SSRIs Analyzed

| Drug | Brand Name |
|---|---|
| Fluoxetine | Prozac |
| Sertraline | Zoloft |
| Paroxetine | Paxil |
| Escitalopram | Lexapro |
| Citalopram | Celexa |
| Fluvoxamine | Luvox |

### Statistical Methods

#### Proportional Reporting Ratio (PRR)

The PRR compares the proportion of a specific adverse reaction for a drug of interest against the proportion of that reaction across all other drugs in the database.

PRR = (a / (a + b)) / (c / (c + d))

#### Reporting Odds Ratio (ROR)

The ROR is the odds-ratio analogue, comparing the odds of a reaction being reported for a specific drug vs. all other drugs.

ROR = (a * d) / (b * c)

#### Signal Criteria (Evans et al., 2001)

A signal is flagged when **all three** conditions are met:

- **PRR ≥ 2.0** — the reaction is reported at least twice as often
- **χ² ≥ 4.0** — statistically significant (p < 0.05)
- **N ≥ 3** — at least 3 cases to avoid spurious signals

Where `a`, `b`, `c`, `d` are cells of the 2×2 contingency table:

|  | Reaction of Interest | All Other Reactions | Total |
|---|---|---|---|
| **Drug of Interest** | a | b | a + b |
| **All Other Drugs** | c | d | c + d |

### MedDRA Terms Analyzed

The following suicidality-related Preferred Terms are evaluated:

- Suicidal ideation
- Suicide attempt
- Completed suicide
- Self-injurious ideation
- Intentional self-injury
- Suicidal behaviour
- Depression suicidal

## 🖥️ Interactive Dashboard

The Streamlit application provides four views:

1. **Overview** — Research context, methodology, and key metrics
2. **Exploratory Analysis** — Interactive visualizations of adverse event profiles
3. **Signal Detection** — PRR/ROR results with forest plots and signal flags
4. **Network View** — Drug–reaction network with adjustable thresholds

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Internet connection (for openFDA API access during data collection)

### Installation

1. Navigate to the project directory:

   ```bash
   cd faers_ssri_signal_detection
   ```

2. Install dependencies:

   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

### Usage

Run the pipeline in order:

```bash
# Step 1: Collect data from openFDA API (~2 minutes)
# Using uv (recommended)
uv run 01_data_collection.py
# Or using python directly
python 01_data_collection.py

# Step 2: Generate EDA visualizations
# Using uv (recommended)
uv run 02_eda.py
# Or using python directly
python 02_eda.py

# Step 3: Run disproportionality analysis
# Using uv (recommended)
uv run 03_signal_detection.py
# Or using python directly
python 03_signal_detection.py

# Step 4: Launch interactive dashboard
# Using uv (recommended)
uv run streamlit run app.py
# Or using streamlit directly
streamlit run app.py
```

## 📁 Repository Structure

```text
├── .python-version           # Python version pin (uv)
├── 01_data_collection.py      # openFDA API data extraction
├── 02_eda.py                  # Exploratory data analysis & visualization
├── 03_signal_detection.py     # PRR/ROR disproportionality analysis
├── app.py                     # Streamlit interactive dashboard
├── requirements.txt           # Python dependencies
├── uv.lock                   # Lockfile for reproducible environment
├── data/                      # Collected data (generated)
│   ├── ssri_reaction_counts.csv
│   ├── ssri_totals.csv
│   └── background_totals.csv
├── outputs/                   # Analysis outputs (generated)
│   ├── signal_detection_results.csv
│   ├── 01_total_reports_by_ssri.png
│   ├── 02_top_reactions_per_ssri.png
│   ├── 03_suicidality_comparison.png
│   ├── 04_reaction_heatmap.png
│   ├── 05_prr_forest_plot.png
│   ├── 06_ror_forest_plot.png
│   └── 07_signal_summary_table.png
└── README.md
```

## ⚠️ Limitations

- **Spontaneous reporting bias:** FAERS is subject to underreporting, stimulated reporting (e.g., after media attention), and notoriety bias.
- **No causality:** Disproportionality signals indicate statistical associations, not causal relationships.
- **Confounding:** Patients on SSRIs have underlying conditions (depression) that independently increase suicidality risk.
- **Reporting denominator unknown:** FAERS does not capture the total number of patients exposed to each drug, making true incidence rates impossible to calculate.

## 📚 References

1. Evans, S.J.W., Waller, P.C., & Davis, S. (2001). Use of proportional reporting ratios (PRRs) for signal generation from spontaneous adverse drug reaction reports. *Pharmacoepidemiology and Drug Safety*, 10(6), 483–486.
2. van Puijenbroek, E.P., et al. (2002). A comparison of measures of disproportionality for signal detection in spontaneous reporting systems. *Pharmacoepidemiology and Drug Safety*, 11(1), 3–10.
3. FDA. (2018). Questions and Answers on FDA's Adverse Event Reporting System (FAERS). Retrieved from [fda.gov](https://www.fda.gov/drugs/surveillance/fda-adverse-event-reporting-system-faers).

## 📜 Disclaimer

> **CLINICAL DISCLAIMER:** This project is designed for **educational and research purposes only**. The analysis uses spontaneous reporting data (FAERS), which has inherent limitations including reporting bias and confounding. Results **must not** be used as a substitute for professional medical judgment, clinical decision-making, or to guide pharmacotherapy choices.

---

## 👨‍🔬Author

Sebastian Lijewski
PhD in Pharmaceutical Sciences

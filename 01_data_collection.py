"""
==============================================================================
01_data_collection.py — FDA FAERS Data Collection for SSRI Signal Detection
==============================================================================

This script queries the openFDA Drug Adverse Event API to collect adverse
event reporting data for six Selective Serotonin Reuptake Inhibitors (SSRIs).

Data Sources:
    - openFDA Drug Event API: https://open.fda.gov/apis/drug/event/
    - No API key required (rate-limited to ~240 requests/minute)

Methodology:
    For each SSRI, we collect:
    1. Total number of adverse event reports (denominator for PRR calculation)
    2. Top 1000 adverse reaction terms with their counts (numerators)
    3. Overall FAERS background totals (all-drug comparator)

    The 'count' endpoint aggregates results server-side, avoiding the need
    to download and process millions of individual reports.

Output:
    - data/ssri_reaction_counts.csv   — reaction term counts per SSRI
    - data/ssri_totals.csv            — total report counts per drug
    - data/background_totals.csv      — overall FAERS background rates

Author: Sebastian Lijewski, PhD
==============================================================================
"""

import os
import time
import json
import requests
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# openFDA Drug Event API base URL
BASE_URL = "https://api.fda.gov/drug/event.json"

# The six major SSRIs approved by the FDA
# Generic names are used as they match the openFDA 'generic_name' field
SSRI_DRUGS = {
    "fluoxetine":    "Fluoxetine (Prozac)",
    "sertraline":    "Sertraline (Zoloft)",
    "paroxetine":    "Paroxetine (Paxil)",
    "escitalopram":  "Escitalopram (Lexapro)",
    "citalopram":    "Citalopram (Celexa)",
    "fluvoxamine":   "Fluvoxamine (Luvox)",
}

# Suicidality-related MedDRA Preferred Terms (PTs) for focused analysis
# These cover the spectrum from ideation to completed events
SUICIDALITY_TERMS = [
    "Suicidal ideation",
    "Suicide attempt",
    "Completed suicide",
    "Self-injurious ideation",
    "Intentional self-injury",
    "Suicidal behaviour",
    "Depression suicidal",
]

# Rate limiting: openFDA allows ~240 requests/minute without an API key
# We add a small delay between requests to be a good API citizen
REQUEST_DELAY_SECONDS = 0.5

# Maximum number of reaction terms to retrieve per drug
# openFDA 'count' endpoint supports up to 1000
MAX_REACTION_TERMS = 1000

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# =============================================================================
# API HELPER FUNCTIONS
# =============================================================================

def query_openfda(params: dict, max_retries: int = 3) -> dict:
    """
    Execute a query against the openFDA Drug Event API with retry logic.

    The openFDA API can occasionally return 5xx errors under heavy load.
    This function implements exponential backoff to handle transient failures.

    Parameters
    ----------
    params : dict
        Query parameters (search, count, limit, skip).
    max_retries : int
        Maximum number of retry attempts on failure.

    Returns
    -------
    dict
        Parsed JSON response from the API.

    Raises
    ------
    requests.exceptions.HTTPError
        If the API returns a non-200 status after all retries.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                # 404 means no results found — this is a valid "zero" result
                return {"meta": {"results": {"total": 0}}, "results": []}
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"  [RETRY] Attempt {attempt + 1} failed ({e}). "
                      f"Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  [RETRY] Connection error ({e}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

    return {}


def get_total_reports(drug_name: str) -> int:
    """
    Get the total number of adverse event reports for a specific drug.

    This queries the API with limit=1 to minimize data transfer while
    still retrieving the total count from the metadata.

    Parameters
    ----------
    drug_name : str
        Generic drug name (e.g., 'fluoxetine').

    Returns
    -------
    int
        Total number of reports in FAERS for this drug.
    """
    params = {
        "search": f'patient.drug.openfda.generic_name:"{drug_name}"',
        "limit": 1,
    }
    data = query_openfda(params)
    return data.get("meta", {}).get("results", {}).get("total", 0)


def get_reaction_counts(drug_name: str, limit: int = MAX_REACTION_TERMS) -> list:
    """
    Get the top adverse reaction terms and their counts for a specific drug.

    Uses the openFDA 'count' endpoint, which performs server-side aggregation
    and returns term-frequency pairs. This is far more efficient than
    downloading individual reports.

    Parameters
    ----------
    drug_name : str
        Generic drug name (e.g., 'fluoxetine').
    limit : int
        Maximum number of reaction terms to retrieve (max 1000).

    Returns
    -------
    list of dict
        Each dict has 'term' (reaction name) and 'count' (frequency).
        Example: [{'term': 'Nausea', 'count': 7516}, ...]
    """
    params = {
        "search": f'patient.drug.openfda.generic_name:"{drug_name}"',
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": limit,
    }
    data = query_openfda(params)
    return data.get("results", [])


def get_background_reaction_count(reaction_term: str) -> int:
    """
    Get the total number of reports containing a specific reaction term
    across ALL drugs in FAERS (background rate).

    This is essential for calculating PRR — we need to know how common
    a reaction is across the entire database, not just for one drug.

    Parameters
    ----------
    reaction_term : str
        MedDRA Preferred Term (e.g., 'Suicidal ideation').

    Returns
    -------
    int
        Total reports with this reaction across all drugs.
    """
    params = {
        "search": f'patient.reaction.reactionmeddrapt:"{reaction_term}"',
        "limit": 1,
    }
    data = query_openfda(params)
    return data.get("meta", {}).get("results", {}).get("total", 0)


def get_total_faers_reports() -> int:
    """
    Get the approximate total number of reports in the FAERS database.

    This serves as the grand total denominator for disproportionality
    calculations. We query without any search filter to get the overall count.

    Returns
    -------
    int
        Total number of reports in FAERS.
    """
    # Use a broad, reliable query — searching for "serious:1" OR "serious:2"
    # captures virtually all reports without complex date range filtering
    # that can cause 500 errors on the openFDA API
    params = {"limit": 1}
    data = query_openfda(params)
    total = data.get("meta", {}).get("results", {}).get("total", 0)
    if total == 0:
        # Fallback: use sum of SSRIs × scaling factor as rough estimate
        # (openFDA reports ~30M total reports as of 2024)
        total = 30_000_000
        print("  ⚠ Could not retrieve total FAERS count. Using estimate.")
    return total


# =============================================================================
# MAIN DATA COLLECTION PIPELINE
# =============================================================================

def collect_all_data():
    """
    Main pipeline: collect all SSRI adverse event data from openFDA.

    This function orchestrates the entire data collection process:
    1. Fetches total report counts for each SSRI (denominators)
    2. Fetches top reaction term counts for each SSRI (numerators)
    3. Fetches background rates for suicidality terms (comparators)
    4. Fetches the overall FAERS database size (grand total)

    All results are saved as CSV files in the data/ directory.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------------------
    # STEP 1: Collect total report counts per SSRI
    # -----------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Collecting total report counts per SSRI")
    print("=" * 70)

    totals_data = []
    for generic_name, display_name in SSRI_DRUGS.items():
        total = get_total_reports(generic_name)
        totals_data.append({
            "generic_name": generic_name,
            "display_name": display_name,
            "total_reports": total,
        })
        print(f"  ✓ {display_name}: {total:,} total reports")
        time.sleep(REQUEST_DELAY_SECONDS)

    df_totals = pd.DataFrame(totals_data)
    df_totals.to_csv(os.path.join(OUTPUT_DIR, "ssri_totals.csv"), index=False)
    print(f"\n  Saved: data/ssri_totals.csv ({len(df_totals)} drugs)\n")

    # -----------------------------------------------------------------
    # STEP 2: Collect reaction term counts per SSRI
    # -----------------------------------------------------------------
    print("=" * 70)
    print("STEP 2: Collecting adverse reaction counts per SSRI")
    print("=" * 70)

    all_reactions = []
    for generic_name, display_name in SSRI_DRUGS.items():
        print(f"\n  Fetching reactions for {display_name}...")
        reactions = get_reaction_counts(generic_name)
        for reaction in reactions:
            all_reactions.append({
                "generic_name": generic_name,
                "display_name": display_name,
                "reaction_term": reaction["term"],
                "count": reaction["count"],
            })
        print(f"  ✓ {display_name}: {len(reactions)} unique reaction terms")
        time.sleep(REQUEST_DELAY_SECONDS)

    df_reactions = pd.DataFrame(all_reactions)
    df_reactions.to_csv(
        os.path.join(OUTPUT_DIR, "ssri_reaction_counts.csv"), index=False
    )
    print(f"\n  Saved: data/ssri_reaction_counts.csv "
          f"({len(df_reactions)} rows)\n")

    # -----------------------------------------------------------------
    # STEP 3: Collect background rates for suicidality-related terms
    # -----------------------------------------------------------------
    print("=" * 70)
    print("STEP 3: Collecting FAERS background rates for suicidality terms")
    print("=" * 70)

    background_data = []
    for term in SUICIDALITY_TERMS:
        bg_count = get_background_reaction_count(term)
        background_data.append({
            "reaction_term": term,
            "background_count": bg_count,
        })
        print(f"  ✓ '{term}': {bg_count:,} total reports across all drugs")
        time.sleep(REQUEST_DELAY_SECONDS)

    # Also get the grand total of all FAERS reports
    print("\n  Fetching total FAERS database size...")
    total_faers = get_total_faers_reports()
    print(f"  ✓ Total FAERS reports: {total_faers:,}")

    df_background = pd.DataFrame(background_data)
    df_background["total_faers_reports"] = total_faers
    df_background.to_csv(
        os.path.join(OUTPUT_DIR, "background_totals.csv"), index=False
    )
    print(f"\n  Saved: data/background_totals.csv\n")

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------
    print("=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)
    print(f"\nFiles saved to: {OUTPUT_DIR}/")
    print(f"  • ssri_totals.csv          — {len(df_totals)} drugs")
    print(f"  • ssri_reaction_counts.csv — {len(df_reactions)} rows")
    print(f"  • background_totals.csv    — {len(df_background)} terms")
    print(f"\nTotal FAERS database: {total_faers:,} reports")
    print(f"SSRIs analyzed: {', '.join(SSRI_DRUGS.values())}")


if __name__ == "__main__":
    collect_all_data()

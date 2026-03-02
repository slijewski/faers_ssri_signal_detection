import logging
import os
import time
import json
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')



BASE_URL = "https://api.fda.gov/drug/event.json"

SSRI_DRUGS = {
    "fluoxetine":    "Fluoxetine (Prozac)",
    "sertraline":    "Sertraline (Zoloft)",
    "paroxetine":    "Paroxetine (Paxil)",
    "escitalopram":  "Escitalopram (Lexapro)",
    "citalopram":    "Citalopram (Celexa)",
    "fluvoxamine":   "Fluvoxamine (Luvox)",
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

REQUEST_DELAY_SECONDS = 0.5

MAX_REACTION_TERMS = 1000

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")



def query_openfda(params: dict, max_retries: int = 3) -> dict:

    for attempt in range(max_retries):
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                return {"meta": {"results": {"total": 0}}, "results": []}
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logging.info(f"  [RETRY] Attempt {attempt + 1} failed ({e}). "
                      f"Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logging.info(f"  [RETRY] Connection error ({e}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

    return {}


def get_total_reports(drug_name: str) -> int:

    params = {
        "search": f'patient.drug.openfda.generic_name:"{drug_name}"',
        "limit": 1,
    }
    data = query_openfda(params)
    return data.get("meta", {}).get("results", {}).get("total", 0)


def get_reaction_counts(drug_name: str, limit: int = MAX_REACTION_TERMS) -> list:

    params = {
        "search": f'patient.drug.openfda.generic_name:"{drug_name}"',
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": limit,
    }
    data = query_openfda(params)
    return data.get("results", [])


def get_background_reaction_count(reaction_term: str) -> int:

    params = {
        "search": f'patient.reaction.reactionmeddrapt:"{reaction_term}"',
        "limit": 1,
    }
    data = query_openfda(params)
    return data.get("meta", {}).get("results", {}).get("total", 0)


def get_total_faers_reports() -> int:

    params = {"limit": 1}
    data = query_openfda(params)
    total = data.get("meta", {}).get("results", {}).get("total", 0)
    if total == 0:
        total = 30_000_000
        logging.info("  ⚠ Could not retrieve total FAERS count. Using estimate.")
    return total



def collect_all_data():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info("=" * 70)
    logging.info("STEP 1: Collecting total report counts per SSRI")
    logging.info("=" * 70)

    totals_data = []
    for generic_name, display_name in SSRI_DRUGS.items():
        total = get_total_reports(generic_name)
        totals_data.append({
            "generic_name": generic_name,
            "display_name": display_name,
            "total_reports": total,
        })
        logging.info(f"  ✓ {display_name}: {total:,} total reports")
        time.sleep(REQUEST_DELAY_SECONDS)

    df_totals = pd.DataFrame(totals_data)
    df_totals.to_csv(os.path.join(OUTPUT_DIR, "ssri_totals.csv"), index=False)
    logging.info(f"\n  Saved: data/ssri_totals.csv ({len(df_totals)} drugs)\n")

    logging.info("=" * 70)
    logging.info("STEP 2: Collecting adverse reaction counts per SSRI")
    logging.info("=" * 70)

    all_reactions = []
    for generic_name, display_name in SSRI_DRUGS.items():
        logging.info(f"\n  Fetching reactions for {display_name}...")
        reactions = get_reaction_counts(generic_name)
        for reaction in reactions:
            all_reactions.append({
                "generic_name": generic_name,
                "display_name": display_name,
                "reaction_term": reaction["term"],
                "count": reaction["count"],
            })
        logging.info(f"  ✓ {display_name}: {len(reactions)} unique reaction terms")
        time.sleep(REQUEST_DELAY_SECONDS)

    df_reactions = pd.DataFrame(all_reactions)
    df_reactions.to_csv(
        os.path.join(OUTPUT_DIR, "ssri_reaction_counts.csv"), index=False
    )
    logging.info(f"\n  Saved: data/ssri_reaction_counts.csv "
          f"({len(df_reactions)} rows)\n")

    logging.info("=" * 70)
    logging.info("STEP 3: Collecting FAERS background rates for suicidality terms")
    logging.info("=" * 70)

    background_data = []
    for term in SUICIDALITY_TERMS:
        bg_count = get_background_reaction_count(term)
        background_data.append({
            "reaction_term": term,
            "background_count": bg_count,
        })
        logging.info(f"  ✓ '{term}': {bg_count:,} total reports across all drugs")
        time.sleep(REQUEST_DELAY_SECONDS)

    logging.info("\n  Fetching total FAERS database size...")
    total_faers = get_total_faers_reports()
    logging.info(f"  ✓ Total FAERS reports: {total_faers:,}")

    df_background = pd.DataFrame(background_data)
    df_background["total_faers_reports"] = total_faers
    df_background.to_csv(
        os.path.join(OUTPUT_DIR, "background_totals.csv"), index=False
    )
    logging.info(f"\n  Saved: data/background_totals.csv\n")

    logging.info("=" * 70)
    logging.info("DATA COLLECTION COMPLETE")
    logging.info("=" * 70)
    logging.info(f"\nFiles saved to: {OUTPUT_DIR}/")
    logging.info(f"  • ssri_totals.csv          — {len(df_totals)} drugs")
    logging.info(f"  • ssri_reaction_counts.csv — {len(df_reactions)} rows")
    logging.info(f"  • background_totals.csv    — {len(df_background)} terms")
    logging.info(f"\nTotal FAERS database: {total_faers:,} reports")
    logging.info(f"SSRIs analyzed: {', '.join(SSRI_DRUGS.values())}")


if __name__ == "__main__":
    collect_all_data()

"""
==============================================================================
app.py — SSRI Pharmacovigilance Signal Detection Dashboard (Streamlit)
==============================================================================

Interactive dashboard for exploring FDA FAERS data on SSRI antidepressants
and suicidality-related adverse events.

Features:
    - Overview of the dataset and methodology
    - Interactive EDA visualizations
    - Signal detection results (PRR/ROR) with filtering
    - Drug-reaction network visualization

Usage:
    streamlit run app.py

Author: Sebastian Lijewski, PhD
==============================================================================
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="SSRI Signal Detection — FAERS",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# DATA LOADING (cached for performance)
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")


@st.cache_data
def load_data():
    """Load all project data files with caching."""
    reactions = pd.read_csv(os.path.join(DATA_DIR, "ssri_reaction_counts.csv"))
    totals = pd.read_csv(os.path.join(DATA_DIR, "ssri_totals.csv"))
    background = pd.read_csv(os.path.join(DATA_DIR, "background_totals.csv"))

    signal_path = os.path.join(OUTPUT_DIR, "signal_detection_results.csv")
    signals = pd.read_csv(signal_path) if os.path.exists(signal_path) else None

    return reactions, totals, background, signals


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar with navigation and information."""
    st.sidebar.title("💊 SSRI Signal Detection")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Overview", "📊 Exploratory Analysis", "🔬 Signal Detection",
         "🕸️ Network View"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Data Source:** [openFDA](https://open.fda.gov/)
        **Method:** Disproportionality Analysis
        **Author:** Sebastian Lijewski, PhD
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.warning(
        "⚠️ **Disclaimer:** This tool is for educational and research "
        "purposes only. Do not use for clinical decision-making."
    )

    return page


# =============================================================================
# PAGE: OVERVIEW
# =============================================================================

def page_overview(df_totals, df_background):
    """Render the Overview page."""
    st.title("🔬 SSRI Pharmacovigilance Signal Detection")
    st.markdown("### FDA Adverse Event Reporting System (FAERS) Analysis")

    st.markdown("---")

    # Key metrics
    total_faers = df_background["total_faers_reports"].iloc[0]
    total_ssri = df_totals["total_reports"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total FAERS Reports", f"{total_faers:,.0f}")
    col2.metric("SSRI Reports", f"{total_ssri:,.0f}")
    col3.metric("SSRIs Analyzed", f"{len(df_totals)}")
    col4.metric("Signal Method", "PRR / ROR")

    st.markdown("---")

    # Methodology section
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(
            """
            ## Research Question

            > **Which SSRI drugs show the strongest disproportionality signal
            > for suicidal ideation in the FDA FAERS database?**

            ## Methodology

            This project applies **Disproportionality Analysis (DA)**, the
            standard quantitative method used by regulatory agencies (FDA, EMA)
            and pharmaceutical companies for pharmacovigilance signal detection
            in spontaneous reporting systems.

            ### Metrics Calculated:
            - **PRR** (Proportional Reporting Ratio) — compares the proportion
              of a reaction for a specific drug vs. all other drugs
            - **ROR** (Reporting Odds Ratio) — odds-ratio analogue for
              pharmacovigilance data
            - **χ²** (Chi-squared) — statistical significance test

            ### Signal Criteria (Evans et al., 2001):
            A safety signal is flagged when **all three conditions** are met:
            - PRR ≥ 2.0
            - χ² ≥ 4.0 (p < 0.05)
            - N ≥ 3 cases
            """
        )

    with col_right:
        st.markdown("## SSRIs Analyzed")
        for _, row in df_totals.iterrows():
            st.markdown(
                f"**{row['display_name']}** — "
                f"{row['total_reports']:,.0f} reports"
            )

        st.markdown("---")
        st.markdown(
            """
            ## The 2×2 Contingency Table
            |  | Reaction | Other |
            |---|---|---|
            | **Drug** | a | b |
            | **Other Drugs** | c | d |

            *PRR = [a/(a+b)] / [c/(c+d)]*
            """
        )


# =============================================================================
# PAGE: EDA
# =============================================================================

def page_eda(df_reactions, df_totals):
    """Render the Exploratory Data Analysis page."""
    st.title("📊 Exploratory Data Analysis")

    # --- Total reports bar chart ---
    st.markdown("### FAERS Reporting Volume by SSRI")

    df_sorted = df_totals.sort_values("total_reports", ascending=True)
    fig_totals = px.bar(
        df_sorted,
        x="total_reports",
        y="display_name",
        orientation="h",
        color="display_name",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"total_reports": "Total Reports", "display_name": "SSRI Drug"},
    )
    fig_totals.update_layout(
        showlegend=False,
        height=400,
        xaxis_tickformat=",",
    )
    st.plotly_chart(fig_totals, use_container_width=True)

    st.markdown("---")

    # --- Interactive: select drug to see top reactions ---
    st.markdown("### Top Adverse Events by Drug")
    col1, col2 = st.columns([1, 3])

    with col1:
        selected_drug = st.selectbox(
            "Select SSRI:",
            df_reactions["display_name"].unique(),
        )
        top_n = st.slider("Number of reactions:", 5, 30, 15)

    with col2:
        df_drug = (
            df_reactions[df_reactions["display_name"] == selected_drug]
            .nlargest(top_n, "count")
            .sort_values("count", ascending=True)
        )

        fig_top = px.bar(
            df_drug,
            x="count",
            y="reaction_term",
            orientation="h",
            color_discrete_sequence=["#E63946"],
            labels={"count": "Number of Reports", "reaction_term": "Adverse Event"},
        )
        fig_top.update_layout(height=max(400, top_n * 25), showlegend=False)
        st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("---")

    # --- Heatmap ---
    st.markdown("### Drug × Reaction Heatmap (Top 25 Reactions)")

    top_reactions = (
        df_reactions.groupby("reaction_term")["count"]
        .sum()
        .nlargest(25)
        .index.tolist()
    )
    df_top_all = df_reactions[df_reactions["reaction_term"].isin(top_reactions)]
    heatmap_data = df_top_all.pivot_table(
        index="reaction_term",
        columns="display_name",
        values="count",
        fill_value=0,
    )
    heatmap_data = heatmap_data.loc[
        heatmap_data.sum(axis=1).sort_values(ascending=False).index
    ]

    fig_heat = px.imshow(
        np.log10(heatmap_data.replace(0, 1)),
        labels=dict(x="SSRI Drug", y="Adverse Event", color="log₁₀(Count)"),
        color_continuous_scale="YlOrRd",
        aspect="auto",
        text_auto=False,
    )
    fig_heat.update_layout(height=700)

    # Add text annotations with actual counts
    for i, row_name in enumerate(heatmap_data.index):
        for j, col_name in enumerate(heatmap_data.columns):
            val = heatmap_data.loc[row_name, col_name]
            fig_heat.add_annotation(
                x=j, y=i,
                text=f"{val:,.0f}",
                showarrow=False,
                font=dict(size=8, color="black" if val < 5000 else "white"),
            )

    st.plotly_chart(fig_heat, use_container_width=True)


# =============================================================================
# PAGE: SIGNAL DETECTION
# =============================================================================

def page_signal_detection(df_signals):
    """Render the Signal Detection results page."""
    st.title("🔬 Signal Detection Results")

    if df_signals is None:
        st.error(
            "⚠️ Signal detection results not found. "
            "Run `python 03_signal_detection.py` first."
        )
        return

    st.markdown(
        """
        Results of disproportionality analysis using **PRR** and **ROR**
        with Evans et al. signal criteria.
        """
    )

    # --- Filters ---
    col1, col2 = st.columns(2)
    with col1:
        selected_reaction = st.selectbox(
            "Select reaction term:",
            df_signals["reaction_term"].unique(),
            index=0,
        )
    with col2:
        show_signals_only = st.checkbox("Show signals only", value=False)

    df_filtered = df_signals[df_signals["reaction_term"] == selected_reaction]
    if show_signals_only:
        df_filtered = df_filtered[df_filtered["Signal_Status"] == "SIGNAL"]

    # --- Results table with color coding ---
    st.markdown(f"### Results for: *{selected_reaction}*")

    df_display = df_filtered[[
        "display_name", "n_cases (a)", "PRR", "PRR_lower_95CI",
        "PRR_upper_95CI", "ROR", "ROR_lower_95CI", "ROR_upper_95CI",
        "Chi_squared", "Signal_Status",
    ]].copy()
    df_display = df_display.sort_values("PRR", ascending=False)
    df_display.columns = [
        "Drug", "N Cases", "PRR", "PRR Lower", "PRR Upper",
        "ROR", "ROR Lower", "ROR Upper", "χ²", "Signal"
    ]

    # Color-code the signal column
    def color_signal(val):
        if val == "SIGNAL":
            return "background-color: #FECDD3; color: #991B1B; font-weight: bold"
        elif val == "NO SIGNAL":
            return "background-color: #DBEAFE; color: #1E40AF"
        return "background-color: #F3F4F6; color: #6B7280"

    styled = df_display.style.map(color_signal, subset=["Signal"])
    styled = styled.format({
        "PRR": "{:.3f}", "PRR Lower": "{:.3f}", "PRR Upper": "{:.3f}",
        "ROR": "{:.3f}", "ROR Lower": "{:.3f}", "ROR Upper": "{:.3f}",
        "χ²": "{:.1f}", "N Cases": "{:,.0f}",
    })

    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Interactive Forest Plot (PRR) ---
    st.markdown("### PRR Forest Plot")

    df_plot = df_filtered.sort_values("PRR", ascending=True).copy()

    fig_forest = go.Figure()

    # Confidence intervals
    for _, row in df_plot.iterrows():
        color = "#E63946" if row["Signal_Status"] == "SIGNAL" else "#457B9D"
        fig_forest.add_trace(go.Scatter(
            x=[row["PRR_lower_95CI"], row["PRR_upper_95CI"]],
            y=[row["display_name"], row["display_name"]],
            mode="lines",
            line=dict(color=color, width=3),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_forest.add_trace(go.Scatter(
            x=[row["PRR"]],
            y=[row["display_name"]],
            mode="markers",
            marker=dict(
                color=color, size=12, symbol="diamond",
                line=dict(color="white", width=1),
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{row['display_name']}</b><br>"
                f"PRR = {row['PRR']:.3f}<br>"
                f"95% CI: [{row['PRR_lower_95CI']:.3f} – "
                f"{row['PRR_upper_95CI']:.3f}]<br>"
                f"χ² = {row['Chi_squared']:.1f}<br>"
                f"N = {row['n_cases (a)']}<extra></extra>"
            ),
        ))

    # Reference lines
    fig_forest.add_vline(x=1, line_dash="dash", line_color="gray",
                         annotation_text="PRR=1")
    fig_forest.add_vline(x=2, line_dash="dot", line_color="red",
                         annotation_text="Threshold")

    fig_forest.update_layout(
        title=f"PRR with 95% CI — {selected_reaction}",
        xaxis_title="Proportional Reporting Ratio (PRR)",
        height=400,
        xaxis=dict(range=[0, None]),
    )
    st.plotly_chart(fig_forest, use_container_width=True)

    # --- Full results across all reactions ---
    st.markdown("---")
    st.markdown("### Complete Signal Detection Results")

    df_all = df_signals[[
        "display_name", "reaction_term", "n_cases (a)", "PRR",
        "Chi_squared", "Signal_Status",
    ]].copy()
    df_all = df_all.sort_values(["reaction_term", "PRR"], ascending=[True, False])
    df_all.columns = ["Drug", "Reaction", "N Cases", "PRR", "χ²", "Signal"]

    styled_all = df_all.style.map(color_signal, subset=["Signal"])
    styled_all = styled_all.format({"PRR": "{:.3f}", "χ²": "{:.1f}"})

    st.dataframe(styled_all, use_container_width=True, hide_index=True, height=500)


# =============================================================================
# PAGE: NETWORK VIEW
# =============================================================================

def page_network(df_reactions):
    """Render a drug-reaction network as a Plotly force-directed graph."""
    st.title("🕸️ Drug–Reaction Network")
    st.markdown(
        "Interactive network showing relationships between SSRIs and their "
        "most frequently reported adverse events."
    )

    # Allow user to filter by minimum count
    min_count = st.slider(
        "Minimum report count to display edge:",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
    )

    # Filter reactions above threshold
    df_net = df_reactions[df_reactions["count"] >= min_count].copy()

    # Get unique drugs and reactions
    drugs = df_net["display_name"].unique().tolist()
    reactions = df_net["reaction_term"].unique().tolist()
    all_nodes = drugs + reactions

    # Create adjacency for simple Plotly scatter layout
    # Use a circular layout for drugs and position reactions around them
    import math

    node_positions = {}

    # Place drugs in inner circle
    for i, drug in enumerate(drugs):
        angle = 2 * math.pi * i / len(drugs)
        node_positions[drug] = (math.cos(angle) * 2, math.sin(angle) * 2)

    # Place reactions in outer circle
    for i, reaction in enumerate(reactions):
        angle = 2 * math.pi * i / len(reactions)
        node_positions[reaction] = (math.cos(angle) * 5, math.sin(angle) * 5)

    # Build edges
    edge_x, edge_y = [], []
    for _, row in df_net.iterrows():
        x0, y0 = node_positions[row["display_name"]]
        x1, y1 = node_positions[row["reaction_term"]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#ADB5BD"),
        hoverinfo="none",
        showlegend=False,
    ))

    # Drug nodes
    drug_x = [node_positions[d][0] for d in drugs]
    drug_y = [node_positions[d][1] for d in drugs]
    fig.add_trace(go.Scatter(
        x=drug_x, y=drug_y,
        mode="markers+text",
        marker=dict(size=20, color="#E63946", line=dict(width=2, color="white")),
        text=drugs,
        textposition="top center",
        textfont=dict(size=10, color="#E63946"),
        hoverinfo="text",
        showlegend=False,
    ))

    # Reaction nodes (sized by total count)
    reaction_counts = df_net.groupby("reaction_term")["count"].sum()
    reaction_sizes = [
        max(8, min(25, reaction_counts.get(r, 0) / reaction_counts.max() * 25))
        for r in reactions
    ]
    reaction_x = [node_positions[r][0] for r in reactions]
    reaction_y = [node_positions[r][1] for r in reactions]

    fig.add_trace(go.Scatter(
        x=reaction_x, y=reaction_y,
        mode="markers+text",
        marker=dict(
            size=reaction_sizes,
            color="#457B9D",
            line=dict(width=1, color="white"),
        ),
        text=[r[:20] + "..." if len(r) > 20 else r for r in reactions],
        textposition="bottom center",
        textfont=dict(size=8, color="#457B9D"),
        hovertext=[
            f"{r}<br>Total: {reaction_counts.get(r, 0):,}"
            for r in reactions
        ],
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title="SSRI–Adverse Event Network (FAERS)",
        height=700,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Showing {len(df_net)} drug–reaction edges with ≥ {min_count} reports. "
        f"Red = SSRI drugs, Blue = adverse reactions (sized by frequency)."
    )


# =============================================================================
# MAIN APP ROUTER
# =============================================================================

def main():
    df_reactions, df_totals, df_background, df_signals = load_data()
    page = render_sidebar()

    if page == "🏠 Overview":
        page_overview(df_totals, df_background)
    elif page == "📊 Exploratory Analysis":
        page_eda(df_reactions, df_totals)
    elif page == "🔬 Signal Detection":
        page_signal_detection(df_signals)
    elif page == "🕸️ Network View":
        page_network(df_reactions)


if __name__ == "__main__":
    main()

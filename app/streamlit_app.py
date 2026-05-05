"""
Luxury DCX Client Intelligence, GenAI Clienteling & MLOps Framework
Streamlit Dashboard

Created by Charlotte Hsu

Run from the main project folder:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "dcx_client_next_best_action.csv"
FEATURE_IMPORTANCE_FILE = BASE_DIR / "data" / "model_feature_importance.csv"


# ---------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Luxury DCX AI Portfolio",
    page_icon="◆",
    layout="wide",
)


# ---------------------------------------------------------------------
# Luxury Styling
# ---------------------------------------------------------------------

LUXURY_BLACK = "#111111"
CHARCOAL = "#2B2B2B"
IVORY = "#F8F5EF"
SOFT_BEIGE = "#E8DDCE"
CHAMPAGNE = "#C6A664"
MUTED_GOLD = "#A8894F"
SOFT_GRAY = "#F2F0EB"
WHITE = "#FFFFFF"


st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {IVORY};
            color: {LUXURY_BLACK};
        }}

        h1, h2, h3, h4 {{
            color: {LUXURY_BLACK};
            letter-spacing: 0.02em;
        }}

        .block-container {{
            padding-top: 4.5rem;
            padding-bottom: 3rem;
        }}

        header[data-testid="stHeader"] {{
            background: rgba(248, 245, 239, 0.96);
            border-bottom: 1px solid {SOFT_BEIGE};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {IVORY};
            color: {LUXURY_BLACK};
            border-right: 1px solid {SOFT_BEIGE};
        }}

        section[data-testid="stSidebar"] * {{
            color: {LUXURY_BLACK};
        }}

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {{
            color: {LUXURY_BLACK} !important;
        }}

        .sidebar-title {{
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-weight: 700;
            color: {MUTED_GOLD};
            margin-bottom: 1rem;
            padding-top: 0.5rem;
        }}

        .sidebar-note {{
            font-size: 0.78rem;
            color: {CHARCOAL};
            line-height: 1.4;
            margin-bottom: 1.25rem;
        }}

        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background-color: {WHITE} !important;
            border: 1px solid {SOFT_BEIGE} !important;
            border-radius: 12px !important;
            color: {LUXURY_BLACK} !important;
            box-shadow: 0 1px 6px rgba(17, 17, 17, 0.04) !important;
            min-height: 44px !important;
        }}

        section[data-testid="stSidebar"] div[data-baseweb="select"] input {{
            color: {LUXURY_BLACK} !important;
        }}

        section[data-testid="stSidebar"] div[data-baseweb="select"] input::placeholder {{
            color: {CHARCOAL} !important;
        }}

        section[data-testid="stSidebar"] span[data-baseweb="tag"] {{
            background-color: {SOFT_BEIGE} !important;
            border: 1px solid {CHAMPAGNE} !important;
            border-radius: 999px !important;
            color: {LUXURY_BLACK} !important;
        }}

        section[data-testid="stSidebar"] span[data-baseweb="tag"] span {{
            color: {LUXURY_BLACK} !important;
        }}

        section[data-testid="stSidebar"] span[data-baseweb="tag"] svg {{
            fill: {MUTED_GOLD} !important;
        }}

        section[data-testid="stSidebar"] div[data-baseweb="select"] svg {{
            fill: {MUTED_GOLD} !important;
        }}

        section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {{
            border: 1px solid {CHAMPAGNE} !important;
            box-shadow: 0 0 0 1px {CHAMPAGNE} !important;
        }}

        div[data-baseweb="popover"] ul {{
            background-color: {WHITE} !important;
            border: 1px solid {SOFT_BEIGE} !important;
            border-radius: 12px !important;
            box-shadow: 0 8px 24px rgba(17, 17, 17, 0.12) !important;
        }}

        div[data-baseweb="popover"] li {{
            color: {LUXURY_BLACK} !important;
            background-color: {WHITE} !important;
        }}

        div[data-baseweb="popover"] li:hover {{
            background-color: {SOFT_GRAY} !important;
            color: {LUXURY_BLACK} !important;
        }}

        div[data-testid="stMetric"] {{
            background-color: {WHITE};
            border: 1px solid {SOFT_BEIGE};
            padding: 18px;
            border-radius: 14px;
            box-shadow: 0 2px 10px rgba(17, 17, 17, 0.05);
        }}

        div[data-testid="stMetricValue"] {{
            color: {LUXURY_BLACK};
            font-weight: 700;
        }}

        div[data-testid="stMetricLabel"] {{
            color: {CHARCOAL};
        }}

        .luxury-card {{
            background-color: {WHITE};
            padding: 22px 26px;
            border-radius: 16px;
            border: 1px solid {SOFT_BEIGE};
            box-shadow: 0 2px 12px rgba(17, 17, 17, 0.06);
            margin-bottom: 18px;
        }}

        .luxury-card h3 {{
            margin-top: 0;
            margin-bottom: 0.75rem;
        }}

        .luxury-card p {{
            color: {CHARCOAL};
            line-height: 1.6;
            font-size: 0.98rem;
        }}

        .luxury-subtitle {{
            color: {MUTED_GOLD};
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}

        .luxury-divider {{
            border-top: 1px solid {SOFT_BEIGE};
            margin: 1.5rem 0;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
            margin-bottom: 1.5rem;
        }}

        .summary-card {{
            background-color: {WHITE};
            border: 1px solid {SOFT_BEIGE};
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 2px 12px rgba(17, 17, 17, 0.05);
        }}

        .summary-card-title {{
            color: {MUTED_GOLD};
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.75rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}

        .summary-card-body {{
            color: {CHARCOAL};
            font-size: 0.94rem;
            line-height: 1.55;
        }}

        .takeaway-box {{
            background-color: {WHITE};
            color: {LUXURY_BLACK};
            border: 1px solid {SOFT_BEIGE};
            border-radius: 16px;
            padding: 22px 26px;
            margin-bottom: 18px;
            box-shadow: 0 2px 12px rgba(17, 17, 17, 0.06);
        }}

        .takeaway-box h3 {{
            color: {LUXURY_BLACK};
            margin-top: 0;
        }}

        .takeaway-box ul {{
            margin-bottom: 0;
        }}

        .takeaway-box li {{
            color: {CHARCOAL};
            margin-bottom: 0.5rem;
            line-height: 1.5;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: {WHITE};
            border: 1px solid {SOFT_BEIGE};
            border-radius: 999px;
            padding: 8px 18px;
            color: {CHARCOAL};
        }}

        .stTabs [aria-selected="true"] {{
            background-color: {LUXURY_BLACK};
            color: {IVORY};
            border: 1px solid {LUXURY_BLACK};
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid {SOFT_BEIGE};
            border-radius: 12px;
        }}

        div[data-testid="stAlert"] {{
            border-radius: 12px;
        }}

        textarea {{
            border-radius: 12px !important;
            border: 1px solid {SOFT_BEIGE} !important;
            background-color: {WHITE} !important;
            color: {LUXURY_BLACK} !important;
            font-family: Georgia, 'Times New Roman', serif !important;
            line-height: 1.55 !important;
        }}

        @media (max-width: 900px) {{
            .summary-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        st.error(
            "Could not find data/dcx_client_next_best_action.csv. "
            "Please run the full pipeline first:\n\n"
            "python3 01_generate_dcx_client_data.py\n"
            "python3 02_client_segmentation.py\n"
            "python3 03_predictive_modeling.py\n"
            "python3 04_next_best_action.py"
        )
        st.stop()

    return pd.read_csv(DATA_FILE)


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    if not FEATURE_IMPORTANCE_FILE.exists():
        return pd.DataFrame(columns=["feature", "importance"])

    return pd.read_csv(FEATURE_IMPORTANCE_FILE)


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def apply_chart_style(fig, height: int = 420):
    fig.update_layout(
        height=height,
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(color=LUXURY_BLACK, size=12),
        margin=dict(l=20, r=80, t=35, b=35),
        xaxis=dict(
            showgrid=True,
            gridcolor=SOFT_GRAY,
            zeroline=False,
            title_font=dict(color=CHARCOAL),
            tickfont=dict(color=CHARCOAL),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            title_font=dict(color=CHARCOAL),
            tickfont=dict(color=CHARCOAL),
            automargin=True,
        ),
    )
    return fig


def horizontal_bar(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_title: str,
    y_title: str,
    height: int = 420,
    text_format: str | None = None,
):
    fig = px.bar(
        data,
        x=x_col,
        y=y_col,
        orientation="h",
        text=x_col,
        color_discrete_sequence=[CHAMPAGNE],
    )

    if text_format:
        fig.update_traces(
            texttemplate=text_format,
            textposition="outside",
            cliponaxis=False,
            marker_line_color=MUTED_GOLD,
            marker_line_width=0.5,
        )
    else:
        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            marker_line_color=MUTED_GOLD,
            marker_line_width=0.5,
        )

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
    )

    return apply_chart_style(fig, height)


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_pct(value: float) -> str:
    return f"{value:.1%}"


def generate_clienteling_message(
    client_segment: str,
    primary_division: str,
    product_interest: str,
    next_best_action: str,
    outreach_channel: str,
    business_priority: str,
    service_risk: str,
    tone: str,
) -> str:
    channel = str(outreach_channel)

    if "SMS" in channel:
        greeting = "Hello [Client Name],"
        closing = "Warmly, [Advisor Name]"
    elif "Phone" in channel:
        greeting = "Suggested call opening:"
        closing = "Suggested call close: Offer assistance without pressure and confirm the client’s preferred next step."
    else:
        greeting = "Dear [Client Name],"
        closing = "Warm regards,\n[Advisor Name]"

    tone_phrase = {
        "Warm and Personal": "I hope you have been well.",
        "Elegant and Concise": "I hope this note finds you well.",
        "Service-Focused": "I wanted to personally follow up and make sure your experience has been seamless.",
        "Exclusive Invitation": "I wanted to share something special that may be of interest to you.",
    }.get(tone, "I hope this note finds you well.")

    action_key = str(next_best_action)

    if "Private Preview" in action_key:
        body = (
            f"We would be pleased to invite you to preview selected pieces from our "
            f"{primary_division} collection, including styles aligned with your interest "
            f"in {product_interest}."
        )
    elif "Boutique Appointment" in action_key:
        body = (
            f"Based on your interest in {product_interest}, I would be happy to arrange "
            f"a private appointment to explore selected pieces from our {primary_division} "
            f"collection at your convenience."
        )
    elif "WFJ" in action_key or "Watch" in action_key or "Fine Jewelry" in action_key:
        body = (
            f"I wanted to follow up regarding your interest in Watches & Fine Jewelry. "
            f"We would be happy to provide a more personalized consultation around "
            f"{product_interest}."
        )
    elif "Fashion Styling" in action_key:
        body = (
            f"I would be delighted to arrange a styling appointment focused on "
            f"{product_interest} and selected pieces from our latest {primary_division} "
            f"collection."
        )
    elif "Beauty" in action_key or "Replenishment" in action_key:
        body = (
            f"I wanted to check whether you may be ready to replenish or revisit your "
            f"preferred beauty selections, especially related to {product_interest}."
        )
    elif "Service" in action_key or "Recovery" in action_key:
        body = (
            "I wanted to personally follow up regarding your recent experience and ensure "
            "that we address any concerns with the level of care you expect from us."
        )
    elif "Lifecycle" in action_key or "Nurture" in action_key:
        body = (
            f"We would be pleased to continue sharing selected updates and recommendations "
            f"related to {product_interest} as new pieces and experiences become available."
        )
    elif "Digital" in action_key or "Retargeting" in action_key:
        body = (
            f"We noticed your interest in {product_interest} and would be happy to assist "
            f"with any questions or help you explore available options."
        )
    elif "Do Not Contact" in action_key or "Monitor" in action_key:
        body = (
            "At this time, no direct client outreach is recommended. The client should remain "
            "in a monitoring or nurture status unless their contactability or engagement status changes."
        )
    else:
        body = f"We would be happy to assist you with selected recommendations related to {product_interest}."

    urgency = {
        "Immediate": "Given the timing, I wanted to reach out personally.",
        "This Week": "I wanted to share this with you this week in case it is helpful.",
        "This Month": "I wanted to share this as part of our current client follow-up.",
        "Monitor": "I wanted to stay connected and share this when the timing feels right.",
    }.get(business_priority, "")

    service_note = ""
    if service_risk == "High":
        service_note = (
            "\n\nBefore sharing any product recommendation, I would first ensure that any "
            "prior service concern has been fully addressed."
        )
    elif service_risk == "Medium":
        service_note = "\n\nI would also be happy to assist with any service-related questions or follow-up needs."

    return f"""{greeting}

{tone_phrase} {urgency}

{body}{service_note}

{closing}
"""


def build_experimentation_dataset(data: pd.DataFrame) -> pd.DataFrame:
    experiment_df = data.copy()

    experiment_df["experiment_group"] = experiment_df["business_priority"].map(
        {
            "Immediate": "Treatment - Model NBA",
            "This Week": "Treatment - Model NBA",
            "This Month": "Treatment - Model NBA",
            "Monitor": "Control - Standard Outreach",
        }
    ).fillna("Control - Standard Outreach")

    base_prob = experiment_df["predicted_repeat_purchase_probability"].clip(0.01, 0.95)
    treatment_lift = experiment_df["experiment_group"].eq("Treatment - Model NBA").map({True: 0.08, False: 0.00})
    contactability_penalty = experiment_df["client_contactable_flag"].map({1: 0.00, 0: -0.20}).fillna(-0.10)
    service_penalty = experiment_df["service_risk_score"].apply(lambda x: -0.05 if x >= 70 else 0.00)

    experiment_df["simulated_conversion_probability"] = (
        base_prob + treatment_lift + contactability_penalty + service_penalty
    ).clip(0.01, 0.95)

    experiment_df["simulated_converted"] = (
        experiment_df["simulated_conversion_probability"]
        >= experiment_df["simulated_conversion_probability"].median()
    ).astype(int)

    experiment_df["simulated_revenue_after_action"] = (
        experiment_df["simulated_converted"]
        * experiment_df["estimated_action_revenue"]
        * experiment_df["experiment_group"].map(
            {
                "Treatment - Model NBA": 1.08,
                "Control - Standard Outreach": 0.92,
            }
        )
    )

    return experiment_df


def calculate_experiment_summary(experiment_df: pd.DataFrame) -> pd.DataFrame:
    return (
        experiment_df.groupby("experiment_group")
        .agg(
            clients=("client_id", "count"),
            conversion_rate=("simulated_converted", "mean"),
            revenue_per_client=("simulated_revenue_after_action", "mean"),
            total_revenue=("simulated_revenue_after_action", "sum"),
            avg_predicted_probability=("predicted_repeat_purchase_probability", "mean"),
        )
        .reset_index()
    )


def calculate_lift_metrics(experiment_summary: pd.DataFrame) -> dict:
    treatment = experiment_summary[experiment_summary["experiment_group"].eq("Treatment - Model NBA")]
    control = experiment_summary[experiment_summary["experiment_group"].eq("Control - Standard Outreach")]

    if treatment.empty or control.empty:
        return {"conversion_lift": 0, "revenue_lift": 0, "incremental_revenue": 0}

    treatment_conversion = treatment["conversion_rate"].iloc[0]
    control_conversion = control["conversion_rate"].iloc[0]
    treatment_rpc = treatment["revenue_per_client"].iloc[0]
    control_rpc = control["revenue_per_client"].iloc[0]
    treatment_clients = treatment["clients"].iloc[0]

    revenue_lift = treatment_rpc - control_rpc

    return {
        "conversion_lift": treatment_conversion - control_conversion,
        "revenue_lift": revenue_lift,
        "incremental_revenue": revenue_lift * treatment_clients,
    }


def get_model_monitoring_status(data: pd.DataFrame) -> dict:
    avg_score = data["predicted_repeat_purchase_probability"].mean()
    score_std = data["predicted_repeat_purchase_probability"].std()
    high_priority_share = data["model_priority_group"].eq("Top Priority").mean()
    contactable_share = data["client_contactable_flag"].mean()

    prediction_drift = "Needs Review" if avg_score < 0.15 or avg_score > 0.85 else "Healthy"
    score_distribution = "Needs Review" if score_std < 0.05 else "Healthy"
    priority_distribution = "Monitor" if high_priority_share > 0.45 else "Healthy"
    contactability_status = "Monitor" if contactable_share < 0.60 else "Healthy"

    overall_status = "Healthy"
    if "Needs Review" in [prediction_drift, score_distribution, priority_distribution, contactability_status]:
        overall_status = "Needs Review"
    elif "Monitor" in [prediction_drift, score_distribution, priority_distribution, contactability_status]:
        overall_status = "Monitor"

    return {
        "overall_status": overall_status,
        "prediction_drift": prediction_drift,
        "score_distribution": score_distribution,
        "priority_distribution": priority_distribution,
        "contactability_status": contactability_status,
        "avg_score": avg_score,
        "score_std": score_std,
        "high_priority_share": high_priority_share,
        "contactable_share": contactable_share,
    }


df = load_data()
feature_importance = load_feature_importance()


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.markdown(
    '<div class="luxury-subtitle">Created by Charlotte Hsu | Portfolio Project</div>',
    unsafe_allow_html=True,
)

st.title("Luxury DCX Intelligence, GenAI Clienteling & MLOps Framework")

st.markdown(
    """
    <div class="luxury-card">
        <h3>Predictive Modeling, Clienteling, Next-Best-Action Strategy, Brand-Safe GenAI, and Model Monitoring</h3>
        <p>
        This portfolio project demonstrates how a luxury retail analytics team could use
        data science to strengthen digital client experience, boutique clienteling, and
        commercial decision-making. The dashboard connects synthetic client data across
        boutique behavior, product interest, campaign response, appointment activity,
        service risk, purchase history, predictive repeat-purchase modeling, and
        next-best-action recommendations.
        </p>
        <p>
        The project also includes a brand-safe GenAI clienteling assistant prototype and an
        experimentation / MLOps framework that demonstrates how model-driven recommendations
        could be validated, monitored, governed, and scaled in a production environment.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="summary-grid">
        <div class="summary-card">
            <div class="summary-card-title">Project Description</div>
            <div class="summary-card-body">
                End-to-end luxury retail AI portfolio covering synthetic data design, client
                segmentation, predictive modeling, next-best-action, advisor-supported GenAI,
                experimentation, and model monitoring.
            </div>
        </div>
        <div class="summary-card">
            <div class="summary-card-title">Business Question</div>
            <div class="summary-card-body">
                Which clients should the brand prioritize, what action should be recommended,
                and how can predictive analytics and GenAI support more personalized,
                measurable, and brand-safe client engagement?
            </div>
        </div>
        <div class="summary-card">
            <div class="summary-card-title">Portfolio Focus</div>
            <div class="summary-card-body">
                Demonstrates a bridge between analytics, luxury client experience, segmentation,
                commercial planning, executive dashboards, responsible AI, experimentation,
                MLOps, and advisor-supported workflows.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Project Goal, Data Design, and Methodology", expanded=False):
    st.markdown(
        """
        ### Project Goal

        This project demonstrates how a Digital Client Experience data team could use analytics and machine learning to answer:

        1. Which clients are most likely to repeat purchase in the next 90 days?
        2. Which clients are high-value, VIP, at-risk, or strategically important?
        3. Which clients should receive advisor outreach, digital nurture, service recovery, or no direct outreach?
        4. Which next-best-actions have the largest estimated revenue opportunity?
        5. How can predictive analytics support a brand-safe AI roadmap for clienteling?
        6. How should the business validate and monitor model-driven recommendations after deployment?

        ### Data Design

        The dataset is synthetic and created for portfolio demonstration only. Each row represents one simulated luxury client linked to boutique/channel, advisor relationship, client tier, product interest, digital intent, campaign response, appointment conversion, contactability, service behavior, and repeat-purchase outcome.

        ### Modeling Approach

        The model predicts `repeat_purchase_90d`. The priority layer is intentionally separate from model score. It combines predicted probability with VIP status, lifetime value, WFJ opportunity, contactability, service risk, and business rules.

        ### GenAI Clienteling Approach

        The GenAI assistant is designed as a brand-safe workflow prototype. It supports advisor drafting, but does not automate client communication. Consent, contactability, brand voice, advisor review, and service sensitivity remain required controls.

        ### Experimentation and MLOps Approach

        The experimentation framework simulates treatment/control measurement for model-driven next-best-actions. The monitoring framework tracks score distribution, priority mix, contactability, drift indicators, and retraining readiness.
        """
    )


# ---------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------

st.sidebar.markdown(
    """
    <div class="sidebar-title">Filters</div>
    <div class="sidebar-note">
        Refine the client population by region, boutique/channel, division,
        segment, priority, and contact method.
    </div>
    """,
    unsafe_allow_html=True,
)

region_filter = st.sidebar.multiselect("Region", sorted(df["region"].dropna().unique()), default=sorted(df["region"].dropna().unique()))
boutique_filter = st.sidebar.multiselect("Boutique / Channel", sorted(df["boutique_name"].dropna().unique()), default=sorted(df["boutique_name"].dropna().unique()))
division_filter = st.sidebar.multiselect("Primary Division", sorted(df["primary_division"].dropna().unique()), default=sorted(df["primary_division"].dropna().unique()))
segment_filter = st.sidebar.multiselect("Client Segment", sorted(df["client_segment"].dropna().unique()), default=sorted(df["client_segment"].dropna().unique()))
priority_filter = st.sidebar.multiselect("Business Priority", sorted(df["business_priority"].dropna().unique()), default=sorted(df["business_priority"].dropna().unique()))
contact_filter = st.sidebar.multiselect("Contact Method", sorted(df["preferred_contact_method"].dropna().unique()), default=sorted(df["preferred_contact_method"].dropna().unique()))

filtered_df = df[
    df["region"].isin(region_filter)
    & df["boutique_name"].isin(boutique_filter)
    & df["primary_division"].isin(division_filter)
    & df["client_segment"].isin(segment_filter)
    & df["business_priority"].isin(priority_filter)
    & df["preferred_contact_method"].isin(contact_filter)
].copy()

if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust the sidebar filters.")
    st.stop()


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Executive Summary",
        "Data Overview",
        "Client Segmentation",
        "Predictive Model",
        "Next Best Action",
        "DCX / AI Roadmap",
        "GenAI Clienteling Assistant",
        "Experimentation & MLOps",
    ]
)


# ---------------------------------------------------------------------
# Tab 1: Executive Summary
# ---------------------------------------------------------------------

with tab1:
    st.subheader("Executive Summary")

    total_clients = len(filtered_df)
    repeat_purchase_rate = filtered_df["repeat_purchase_90d"].mean()
    avg_predicted_probability = filtered_df["predicted_repeat_purchase_probability"].mean()
    top_priority_clients = filtered_df["model_priority_group"].eq("Top Priority").sum()
    immediate_clients = filtered_df["business_priority"].eq("Immediate").sum()
    high_value_clients = filtered_df["high_value_client_flag"].sum()
    contactable_clients = filtered_df["client_contactable_flag"].sum()
    avg_lifetime_spend = filtered_df["lifetime_spend"].mean()
    total_estimated_revenue = filtered_df["estimated_action_revenue"].sum()

    st.markdown(
        f"""
        <div class="takeaway-box">
            <h3>Dashboard Summary</h3>
            <ul>
                <li>The selected client population includes <b>{total_clients:,.0f}</b> clients across boutique, digital, product, and behavioral dimensions.</li>
                <li>The average predicted repeat-purchase probability is <b>{format_pct(avg_predicted_probability)}</b>, compared with an actual repeat-purchase rate of <b>{format_pct(repeat_purchase_rate)}</b>.</li>
                <li>The filtered view identifies <b>{top_priority_clients:,.0f}</b> top-priority clients and <b>{immediate_clients:,.0f}</b> clients requiring immediate business action.</li>
                <li>The estimated action revenue opportunity for the selected population is <b>{format_currency(total_estimated_revenue)}</b>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Clients", f"{total_clients:,.0f}")
    c2.metric("Actual Repeat Purchase Rate", format_pct(repeat_purchase_rate))
    c3.metric("Avg Predicted Probability", format_pct(avg_predicted_probability))

    c4, c5, c6 = st.columns(3)
    c4.metric("Top Priority Clients", f"{top_priority_clients:,.0f}")
    c5.metric("Immediate Action Clients", f"{immediate_clients:,.0f}")
    c6.metric("High-Value Clients", f"{high_value_clients:,.0f}")

    c7, c8, c9 = st.columns(3)
    c7.metric("Contactable Clients", f"{contactable_clients:,.0f}")
    c8.metric("Avg Lifetime Spend", format_currency(avg_lifetime_spend))
    c9.metric("Estimated Action Revenue", format_currency(total_estimated_revenue))

    st.markdown('<div class="luxury-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="luxury-card">
            <h3>Key Takeaways</h3>
            <p>
            This dashboard moves beyond descriptive reporting by combining predictive modeling
            with business rules and clienteling logic. The output is not only a model score, but
            a practical action framework that can help boutique, CRM, digital, and commercial
            teams decide which clients to prioritize and how to engage them.
            </p>
            <p>
            The most important business insight is that repeat-purchase probability alone should
            not determine outreach priority. A luxury brand should also consider client value,
            VIP status, contactability, product interest, service risk, WFJ opportunity, and the
            expected commercial impact of each recommended action.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Business Priority Distribution")
        priority_counts = filtered_df["business_priority"].value_counts().rename_axis("Business Priority").reset_index(name="Client Count")
        order = ["Immediate", "This Week", "This Month", "Monitor"]
        priority_counts["Business Priority"] = pd.Categorical(priority_counts["Business Priority"], categories=order, ordered=True)
        priority_counts = priority_counts.sort_values("Business Priority", ascending=False)
        fig = horizontal_bar(priority_counts, "Client Count", "Business Priority", "Client Count", "Business Priority", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Estimated Revenue by Business Priority")
        revenue_priority = (
            filtered_df.groupby("business_priority")["estimated_action_revenue"]
            .sum()
            .reset_index()
            .rename(columns={"business_priority": "Business Priority", "estimated_action_revenue": "Estimated Revenue"})
        )
        revenue_priority["Business Priority"] = pd.Categorical(revenue_priority["Business Priority"], categories=order, ordered=True)
        revenue_priority = revenue_priority.sort_values("Business Priority", ascending=False)
        fig = horizontal_bar(revenue_priority, "Estimated Revenue", "Business Priority", "Estimated Revenue", "Business Priority", height=350, text_format="$%{text:,.0f}")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Tab 2: Data Overview
# ---------------------------------------------------------------------

with tab2:
    st.subheader("Data Overview")

    st.markdown(
        """
        <div class="luxury-card">
            <h3>Data Overview Summary</h3>
            <p>
            The synthetic dataset is designed to resemble a luxury retail client intelligence
            environment. It combines channel, boutique, region, product division, campaign,
            contactability, purchase history, and service-related attributes to support a
            more complete view of client behavior.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Boutiques / Channels", f"{filtered_df['boutique_name'].nunique():,.0f}")
    c2.metric("Regions", f"{filtered_df['region'].nunique():,.0f}")
    c3.metric("Primary Divisions", f"{filtered_df['primary_division'].nunique():,.0f}")
    c4.metric("Primary Categories", f"{filtered_df['primary_category'].nunique():,.0f}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Client Distribution by Boutique / Channel")
        boutique_summary = filtered_df["boutique_name"].value_counts().rename_axis("Boutique / Channel").reset_index(name="Client Count").sort_values("Client Count", ascending=True)
        fig = horizontal_bar(boutique_summary, "Client Count", "Boutique / Channel", "Client Count", "Boutique / Channel", height=470)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Client Distribution by Division")
        division_summary = filtered_df["primary_division"].value_counts().rename_axis("Primary Division").reset_index(name="Client Count").sort_values("Client Count", ascending=True)
        fig = horizontal_bar(division_summary, "Client Count", "Primary Division", "Client Count", "Primary Division", height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Category-Level Client Mix")
    category_summary = filtered_df["primary_category"].value_counts().rename_axis("Primary Category").reset_index(name="Client Count").sort_values("Client Count", ascending=True)
    fig = horizontal_bar(category_summary, "Client Count", "Primary Category", "Client Count", "Primary Category", height=560)
    st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Campaign Type Distribution")
        campaign_summary = filtered_df["last_campaign_type"].value_counts().rename_axis("Campaign Type").reset_index(name="Client Count").sort_values("Client Count", ascending=True)
        fig = horizontal_bar(campaign_summary, "Client Count", "Campaign Type", "Client Count", "Campaign Type", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("### Contactability Distribution")
        contact_summary = filtered_df["preferred_contact_method"].value_counts().rename_axis("Preferred Contact Method").reset_index(name="Client Count").sort_values("Client Count", ascending=True)
        fig = horizontal_bar(contact_summary, "Client Count", "Preferred Contact Method", "Client Count", "Preferred Contact Method", height=350)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Tab 3: Client Segmentation
# ---------------------------------------------------------------------

with tab3:
    st.subheader("Client Segmentation")

    st.markdown(
        """
        <div class="luxury-card">
            <h3>Segmentation Summary</h3>
            <p>
            Client segmentation helps convert raw customer records into commercially meaningful
            groups. In a luxury retail environment, segmentation should account for both value
            and relationship context: lifetime spend, client tier, product interest, digital
            engagement, service risk, and reactivation potential.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Segment Distribution")
        segment_counts = filtered_df["client_segment"].value_counts().rename_axis("Client Segment").reset_index(name="Client Count").sort_values("Client Count", ascending=True)
        fig = horizontal_bar(segment_counts, "Client Count", "Client Segment", "Client Count", "Client Segment", height=650)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Average Lifetime Spend by Segment")
        spend_by_segment = (
            filtered_df.groupby("client_segment")["lifetime_spend"]
            .mean()
            .sort_values(ascending=True)
            .reset_index()
            .rename(columns={"client_segment": "Client Segment", "lifetime_spend": "Average Lifetime Spend"})
        )
        fig = horizontal_bar(spend_by_segment, "Average Lifetime Spend", "Client Segment", "Average Lifetime Spend", "Client Segment", height=650, text_format="$%{text:,.0f}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Segment Performance Summary")
    segment_summary = (
        filtered_df.groupby("client_segment")
        .agg(
            clients=("client_id", "count"),
            avg_lifetime_spend=("lifetime_spend", "mean"),
            avg_predicted_probability=("predicted_repeat_purchase_probability", "mean"),
            avg_digital_intent=("digital_intent_score", "mean"),
            avg_service_risk=("service_risk_score", "mean"),
            avg_estimated_action_revenue=("estimated_action_revenue", "mean"),
        )
        .reset_index()
        .sort_values("avg_estimated_action_revenue", ascending=False)
    )

    st.dataframe(
        segment_summary.style.format(
            {
                "avg_lifetime_spend": "${:,.0f}",
                "avg_predicted_probability": "{:.1%}",
                "avg_digital_intent": "{:.1f}",
                "avg_service_risk": "{:.1f}",
                "avg_estimated_action_revenue": "${:,.0f}",
            }
        ),
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Tab 4: Predictive Model
# ---------------------------------------------------------------------

with tab4:
    st.subheader("Predictive Model")

    st.markdown(
        """
        <div class="luxury-card">
            <h3>Model Summary</h3>
            <p>
            The predictive model estimates the probability that a client will repeat purchase
            within 90 days. This score is used as one input into the business-priority framework,
            but it is not the only decision factor. The final prioritization also considers
            client value, VIP status, contactability, service risk, WFJ opportunity, and business rules.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Model Target", "Repeat Purchase in 90 Days")
    c2.metric("Best Model", "Logistic Regression")
    c3.metric("AUC", "0.824")

    st.info(
        "The model estimates repeat-purchase probability. Business priority is a separate layer "
        "that combines model probability with client value, VIP status, contactability, service risk, "
        "WFJ opportunity, and business rules."
    )

    st.markdown("### Top Feature Drivers")
    if not feature_importance.empty:
        top_features = feature_importance.head(25).copy()
        top_features = top_features.sort_values("importance", ascending=True)

        fig = px.bar(top_features, x="importance", y="feature", orientation="h", text="importance", color_discrete_sequence=[CHAMPAGNE])
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False, marker_line_color=MUTED_GOLD, marker_line_width=0.5)
        fig = apply_chart_style(fig, height=700)
        fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(feature_importance.head(25).style.format({"importance": "{:.4f}"}), use_container_width=True)
    else:
        st.warning("Feature importance file not found.")

    st.markdown("### Average Predicted Probability by Priority Group")
    priority_prob = (
        filtered_df.groupby("model_priority_group")["predicted_repeat_purchase_probability"]
        .mean()
        .reset_index()
        .rename(columns={"model_priority_group": "Priority Group", "predicted_repeat_purchase_probability": "Average Predicted Probability"})
    )
    order = ["Top Priority", "High Priority", "Medium Priority", "Low Priority"]
    priority_prob["Priority Group"] = pd.Categorical(priority_prob["Priority Group"], categories=order, ordered=True)
    priority_prob = priority_prob.sort_values("Priority Group", ascending=False)

    fig = horizontal_bar(priority_prob, "Average Predicted Probability", "Priority Group", "Average Predicted Probability", "Priority Group", height=350, text_format="%{text:.1%}")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Tab 5: Next Best Action
# ---------------------------------------------------------------------

with tab5:
    st.subheader("Next Best Action")

    st.markdown(
        """
        <div class="luxury-card">
            <h3>Next-Best-Action Summary</h3>
            <p>
            The next-best-action layer translates analytics into business execution. Each client
            receives a recommended action, suggested outreach channel, priority timing, rationale,
            and estimated revenue opportunity. This creates a bridge between data science and
            boutique/clienteling operations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Action Distribution")
        action_counts = filtered_df["next_best_action"].value_counts().rename_axis("Next Best Action").reset_index(name="Client Count").sort_values("Client Count", ascending=True)
        fig = horizontal_bar(action_counts, "Client Count", "Next Best Action", "Client Count", "Next Best Action", height=700)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Outreach Channel Distribution")
        channel_counts = filtered_df["recommended_outreach_channel"].value_counts().rename_axis("Outreach Channel").reset_index(name="Client Count").sort_values("Client Count", ascending=True)
        fig = horizontal_bar(channel_counts, "Client Count", "Outreach Channel", "Client Count", "Outreach Channel", height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Estimated Revenue by Next-Best-Action")
    action_revenue = (
        filtered_df.groupby("next_best_action")["estimated_action_revenue"]
        .sum()
        .sort_values(ascending=True)
        .reset_index()
        .rename(columns={"next_best_action": "Next Best Action", "estimated_action_revenue": "Estimated Revenue"})
    )
    fig = horizontal_bar(action_revenue, "Estimated Revenue", "Next Best Action", "Estimated Revenue", "Next Best Action", height=700, text_format="$%{text:,.0f}")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Client-Level Action Table")
    table_columns = [
        "client_id", "boutique_name", "region", "advisor_id", "primary_division", "primary_category",
        "client_tier", "luxury_value_band", "client_segment", "model_priority_group",
        "predicted_repeat_purchase_probability", "next_best_action", "recommended_outreach_channel",
        "business_priority", "estimated_action_revenue", "action_rationale",
    ]
    table_df = filtered_df[[c for c in table_columns if c in filtered_df.columns]].copy()

    priority_rank = {"Immediate": 1, "This Week": 2, "This Month": 3, "Monitor": 4}
    table_df["priority_rank"] = table_df["business_priority"].map(priority_rank)
    table_df = table_df.sort_values(["priority_rank", "estimated_action_revenue"], ascending=[True, False]).drop(columns="priority_rank")

    st.dataframe(
        table_df.style.format({"predicted_repeat_purchase_probability": "{:.1%}", "estimated_action_revenue": "${:,.0f}"}),
        use_container_width=True,
        height=580,
    )


# ---------------------------------------------------------------------
# Tab 6: AI Roadmap
# ---------------------------------------------------------------------

with tab6:
    st.subheader("DCX / AI Roadmap")

    st.markdown(
        """
        <div class="luxury-card">
            <h3>Roadmap Summary</h3>
            <p>
            This project can evolve from predictive reporting into a broader client intelligence
            operating model. The long-term opportunity is to connect model scoring, action
            recommendations, advisor workflows, controlled GenAI message drafting, experimentation,
            model monitoring, and governance into one brand-safe luxury clienteling ecosystem.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Phase 1: Predictive Client Prioritization

        Build a repeat-purchase model that identifies clients most likely to re-engage within 90 days.

        **Business output:** ranked client opportunity list by boutique, division, segment, predicted probability, client value, and contactability.

        ### Phase 2: Next-Best-Action Testing

        Convert model scores into recommended actions such as advisor follow-up, service recovery, private preview invitation, digital retargeting, and nurture campaigns.

        **Business output:** action type, outreach channel, priority timing, rationale, and estimated action revenue.

        ### Phase 3: Brand-Safe GenAI Clienteling Assistant

        Pilot a GenAI workflow that drafts advisor-ready outreach messages based on client segment, category interest, service context, and recommended action.

        **Guardrail:** human review is required before any client-facing communication.

        ### Phase 4: Experimentation and Business Impact Validation

        Use A/B testing and holdout design to measure whether recommended actions increase conversion, appointment activity, repeat purchase, and revenue per client.

        **Measurement focus:** revenue lift, conversion lift, contact fatigue, opt-out risk, service sensitivity, and brand experience.

        ### Phase 5: MLOps, Monitoring, Privacy, and Governance

        Establish model refresh cadence, drift monitoring, data privacy controls, consent rules, bias checks, version control, and business owner review.

        **Governance focus:** protect brand voice, client trust, personalization quality, and responsible use of AI.
        """
    )

    st.markdown("### Portfolio Positioning")
    st.write(
        "This project demonstrates a complete DCX AI workflow: synthetic data design, client segmentation, "
        "predictive modeling, business prioritization, next-best-action logic, estimated revenue opportunity, "
        "advisor-supported GenAI, experimentation, model monitoring, dashboard storytelling, and AI roadmap thinking."
    )

    st.markdown(
        """
        <div class="takeaway-box">
            <h3>Final Key Takeaways</h3>
            <ul>
                <li><b>Commercial relevance:</b> The dashboard connects client behavior to revenue opportunity and action planning.</li>
                <li><b>Luxury fit:</b> The logic respects client value, brand experience, personalized outreach, and selective engagement.</li>
                <li><b>Analytics maturity:</b> The project shows descriptive analytics, segmentation, predictive modeling, business rules, experimentation, and model monitoring.</li>
                <li><b>Executive usability:</b> Leadership can quickly understand who to prioritize, why they matter, what action should happen next, and how impact should be measured.</li>
                <li><b>AI readiness:</b> The roadmap creates a responsible path from model scoring to advisor-supported GenAI clienteling and production governance.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Tab 7: GenAI Clienteling Assistant
# ---------------------------------------------------------------------

with tab7:
    st.subheader("GenAI Clienteling Assistant")

    st.markdown(
        """
        <div class="luxury-card">
            <h3>Brand-Safe GenAI Clienteling Prototype</h3>
            <p>
            This prototype demonstrates how Generative AI could support boutique advisors
            by drafting personalized, advisor-ready outreach messages based on client segment,
            product interest, recommended action, contact method, business priority, and service context.
            </p>
            <p>
            The workflow is intentionally designed with luxury-brand guardrails: human review,
            contactability checks, brand voice control, service sensitivity, and no fully automated
            client-facing communication.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Client Context and Action Inputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_segment = st.selectbox("Client Segment", sorted(filtered_df["client_segment"].dropna().unique()))
        selected_division = st.selectbox("Primary Division", sorted(filtered_df["primary_division"].dropna().unique()))
        selected_product_interest = st.selectbox("Product Interest", sorted(filtered_df["primary_category"].dropna().unique()))

    with col2:
        selected_action = st.selectbox("Next Best Action", sorted(filtered_df["next_best_action"].dropna().unique()))
        selected_channel = st.selectbox("Outreach Channel", sorted(filtered_df["recommended_outreach_channel"].dropna().unique()))
        selected_priority = st.selectbox("Business Priority", ["Immediate", "This Week", "This Month", "Monitor"])

    with col3:
        selected_contactability = st.selectbox("Contactability Status", ["Contactable", "Do Not Contact", "Consent Review Needed"])
        selected_service_risk = st.selectbox("Service Risk Level", ["Low", "Medium", "High"])
        selected_tone = st.selectbox("Message Tone", ["Warm and Personal", "Elegant and Concise", "Service-Focused", "Exclusive Invitation"])

    st.markdown('<div class="luxury-divider"></div>', unsafe_allow_html=True)

    if selected_contactability == "Do Not Contact":
        st.warning(
            "This client is marked as Do Not Contact. No outreach message should be generated. "
            "The recommended action should be monitoring only unless consent status changes."
        )

        st.markdown(
            """
            <div class="takeaway-box">
                <h3>Governance Decision</h3>
                <ul>
                    <li>No client-facing outreach should be sent.</li>
                    <li>Client consent and contact preferences override model recommendations.</li>
                    <li>The advisor or CRM team should review consent status before taking action.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        if selected_contactability == "Consent Review Needed":
            st.info(
                "Consent review is needed before outreach. A draft can be prepared internally, "
                "but it should not be sent until contactability is confirmed."
            )
            label = "Advisor Draft"
            heading = "### Draft Message for Internal Review Only"
        else:
            label = "Draft Message"
            heading = "### Advisor-Ready Draft Message"

        generated_message = generate_clienteling_message(
            selected_segment,
            selected_division,
            selected_product_interest,
            selected_action,
            selected_channel,
            selected_priority,
            selected_service_risk,
            selected_tone,
        )

        st.markdown(heading)
        st.text_area(label, generated_message, height=280)

    st.markdown("### Why This Message Was Recommended")

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-card-title">Client Context</div>
                <div class="summary-card-body">
                    Segment: <b>{selected_segment}</b><br>
                    Division: <b>{selected_division}</b><br>
                    Product Interest: <b>{selected_product_interest}</b>
                </div>
            </div>
            <div class="summary-card">
                <div class="summary-card-title">Recommended Action</div>
                <div class="summary-card-body">
                    Action: <b>{selected_action}</b><br>
                    Channel: <b>{selected_channel}</b><br>
                    Priority: <b>{selected_priority}</b>
                </div>
            </div>
            <div class="summary-card">
                <div class="summary-card-title">Governance Context</div>
                <div class="summary-card-body">
                    Contactability: <b>{selected_contactability}</b><br>
                    Service Risk: <b>{selected_service_risk}</b><br>
                    Tone: <b>{selected_tone}</b>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Brand-Safety Checklist")

    checklist_col1, checklist_col2 = st.columns(2)

    with checklist_col1:
        st.markdown(
            """
            <div class="luxury-card">
                <h3>Required Before Sending</h3>
                <p>✓ Confirm client is contactable</p>
                <p>✓ Confirm preferred outreach channel</p>
                <p>✓ Review message for brand voice</p>
                <p>✓ Remove overly personal or sensitive assumptions</p>
                <p>✓ Advisor must approve before sending</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with checklist_col2:
        st.markdown(
            """
            <div class="luxury-card">
                <h3>AI Guardrails</h3>
                <p>✓ AI supports drafting, not final sending</p>
                <p>✓ No automated client communication</p>
                <p>✓ Consent rules override model score</p>
                <p>✓ Service-risk clients require extra care</p>
                <p>✓ All recommendations should be monitored for impact</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Production Roadmap")

    st.markdown(
        """
        <div class="takeaway-box">
            <h3>How This Could Scale in Production</h3>
            <ul>
                <li><b>Data layer:</b> Pull client features, consent status, product interest, and next-best-action from the CRM or data warehouse.</li>
                <li><b>AI layer:</b> Use approved prompt templates and brand voice rules to generate advisor-ready drafts.</li>
                <li><b>Workflow layer:</b> Route drafts to advisors for review, edit, and approval before any client communication.</li>
                <li><b>Governance layer:</b> Track model version, prompt version, approval status, consent status, and message performance.</li>
                <li><b>Measurement layer:</b> Evaluate impact through appointment booking, repeat purchase, revenue lift, response rate, and client opt-out risk.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Tab 8: Experimentation & MLOps
# ---------------------------------------------------------------------

with tab8:
    st.subheader("Experimentation & MLOps")

    st.markdown(
        """
        <div class="luxury-card">
            <h3>Experimentation and Model Monitoring Framework</h3>
            <p>
            This section demonstrates how a luxury DCX data science team could validate the
            business impact of model-driven next-best-action recommendations and monitor the
            model after deployment.
            </p>
            <p>
            The experimentation layer compares model-driven outreach against standard outreach,
            while the MLOps layer tracks model health, score distribution, contactability,
            drift indicators, and retraining readiness.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    experiment_df = build_experimentation_dataset(filtered_df)
    experiment_summary = calculate_experiment_summary(experiment_df)
    lift_metrics = calculate_lift_metrics(experiment_summary)
    monitoring_status = get_model_monitoring_status(filtered_df)

    st.markdown("### A/B Test Executive Summary")

    c1, c2, c3, c4 = st.columns(4)
    treatment_clients = experiment_df["experiment_group"].eq("Treatment - Model NBA").sum()
    control_clients = experiment_df["experiment_group"].eq("Control - Standard Outreach").sum()

    c1.metric("Treatment Clients", f"{treatment_clients:,.0f}")
    c2.metric("Control Clients", f"{control_clients:,.0f}")
    c3.metric("Conversion Lift", format_pct(lift_metrics["conversion_lift"]))
    c4.metric("Incremental Revenue", format_currency(lift_metrics["incremental_revenue"]))

    st.markdown(
        """
        <div class="luxury-card">
            <h3>Experiment Design</h3>
            <p>
            The simulated experiment compares clients receiving model-driven next-best-action
            recommendations against a control group receiving standard outreach. In a real
            production environment, the treatment and control groups would be randomly assigned
            or designed using a proper holdout strategy.
            </p>
            <p>
            The key business question is whether model-driven recommendations improve conversion,
            appointment activity, repeat purchase, and revenue per client while protecting client
            trust and avoiding over-contact.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Conversion Rate by Experiment Group")
        conversion_chart = experiment_summary.copy()
        conversion_chart["conversion_rate_label"] = conversion_chart["conversion_rate"]
        fig = px.bar(conversion_chart, x="experiment_group", y="conversion_rate", text="conversion_rate_label", color_discrete_sequence=[CHAMPAGNE])
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", marker_line_color=MUTED_GOLD, marker_line_width=0.5, cliponaxis=False)
        fig.update_layout(xaxis_title="Experiment Group", yaxis_title="Conversion Rate")
        fig = apply_chart_style(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Revenue per Client by Experiment Group")
        revenue_chart = experiment_summary.copy()
        revenue_chart["revenue_per_client_label"] = revenue_chart["revenue_per_client"]
        fig = px.bar(revenue_chart, x="experiment_group", y="revenue_per_client", text="revenue_per_client_label", color_discrete_sequence=[CHAMPAGNE])
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", marker_line_color=MUTED_GOLD, marker_line_width=0.5, cliponaxis=False)
        fig.update_layout(xaxis_title="Experiment Group", yaxis_title="Revenue per Client")
        fig = apply_chart_style(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Experiment Summary Table")
    st.dataframe(
        experiment_summary.style.format(
            {
                "conversion_rate": "{:.1%}",
                "revenue_per_client": "${:,.0f}",
                "total_revenue": "${:,.0f}",
                "avg_predicted_probability": "{:.1%}",
            }
        ),
        use_container_width=True,
    )

    st.markdown('<div class="luxury-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Action-Level Performance")

    action_performance = (
        experiment_df.groupby("next_best_action")
        .agg(
            clients=("client_id", "count"),
            conversion_rate=("simulated_converted", "mean"),
            revenue_per_client=("simulated_revenue_after_action", "mean"),
            total_revenue=("simulated_revenue_after_action", "sum"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=True)
    )

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Conversion Rate by Next-Best-Action")
        fig = horizontal_bar(action_performance, "conversion_rate", "next_best_action", "Conversion Rate", "Next-Best-Action", height=650, text_format="%{text:.1%}")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("### Total Revenue by Next-Best-Action")
        fig = horizontal_bar(action_performance, "total_revenue", "next_best_action", "Total Revenue", "Next-Best-Action", height=650, text_format="$%{text:,.0f}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Action Performance Table")
    st.dataframe(
        action_performance.sort_values("total_revenue", ascending=False).style.format(
            {
                "conversion_rate": "{:.1%}",
                "revenue_per_client": "${:,.0f}",
                "total_revenue": "${:,.0f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown('<div class="luxury-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Model Monitoring Summary")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Model Status", monitoring_status["overall_status"])
    m2.metric("Avg Model Score", format_pct(monitoring_status["avg_score"]))
    m3.metric("Score Std Dev", f"{monitoring_status['score_std']:.3f}")
    m4.metric("Contactable Share", format_pct(monitoring_status["contactable_share"]))

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-card-title">Prediction Drift</div>
                <div class="summary-card-body">
                    Status: <b>{monitoring_status["prediction_drift"]}</b><br>
                    Average model score is monitored to identify unusual shifts in predicted probability.
                </div>
            </div>
            <div class="summary-card">
                <div class="summary-card-title">Score Distribution</div>
                <div class="summary-card-body">
                    Status: <b>{monitoring_status["score_distribution"]}</b><br>
                    Score standard deviation is monitored to ensure the model continues to separate clients meaningfully.
                </div>
            </div>
            <div class="summary-card">
                <div class="summary-card-title">Priority Mix</div>
                <div class="summary-card-body">
                    Status: <b>{monitoring_status["priority_distribution"]}</b><br>
                    Top-priority share: <b>{format_pct(monitoring_status["high_priority_share"])}</b>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Model Score Distribution")
    fig = px.histogram(experiment_df, x="predicted_repeat_purchase_probability", nbins=30, color_discrete_sequence=[CHAMPAGNE])
    fig.update_traces(marker_line_color=MUTED_GOLD, marker_line_width=0.5)
    fig.update_layout(xaxis_title="Predicted Repeat-Purchase Probability", yaxis_title="Client Count")
    fig = apply_chart_style(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Model Score Decile Monitoring")
    decile_df = experiment_df.copy()
    decile_df["score_decile"] = pd.qcut(
        decile_df["predicted_repeat_purchase_probability"].rank(method="first"),
        q=10,
        labels=["D1 Lowest", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10 Highest"],
    )

    decile_summary = (
        decile_df.groupby("score_decile", observed=False)
        .agg(
            clients=("client_id", "count"),
            avg_predicted_probability=("predicted_repeat_purchase_probability", "mean"),
            simulated_conversion_rate=("simulated_converted", "mean"),
            revenue_per_client=("simulated_revenue_after_action", "mean"),
        )
        .reset_index()
    )

    fig = px.bar(decile_summary, x="score_decile", y="simulated_conversion_rate", text="simulated_conversion_rate", color_discrete_sequence=[CHAMPAGNE])
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", marker_line_color=MUTED_GOLD, marker_line_width=0.5, cliponaxis=False)
    fig.update_layout(xaxis_title="Model Score Decile", yaxis_title="Simulated Conversion Rate")
    fig = apply_chart_style(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        decile_summary.style.format(
            {
                "avg_predicted_probability": "{:.1%}",
                "simulated_conversion_rate": "{:.1%}",
                "revenue_per_client": "${:,.0f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown('<div class="luxury-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Production Readiness Checklist")
    readiness_col1, readiness_col2 = st.columns(2)

    with readiness_col1:
        st.markdown(
            """
            <div class="luxury-card">
                <h3>Experimentation Governance</h3>
                <p>✓ Define treatment and control groups</p>
                <p>✓ Track outreach exposure by client</p>
                <p>✓ Measure conversion, appointment, and revenue lift</p>
                <p>✓ Monitor opt-out and contact fatigue</p>
                <p>✓ Review results by division, boutique, and segment</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with readiness_col2:
        st.markdown(
            """
            <div class="luxury-card">
                <h3>MLOps Governance</h3>
                <p>✓ Version model and feature pipeline</p>
                <p>✓ Monitor score distribution and feature drift</p>
                <p>✓ Track actual outcomes after scoring</p>
                <p>✓ Define retraining cadence and approval workflow</p>
                <p>✓ Document model limitations and business rules</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="takeaway-box">
            <h3>Piece 3 Key Takeaway</h3>
            <ul>
                <li><b>Model scoring is not the finish line:</b> the business still needs to validate whether recommended actions create measurable lift.</li>
                <li><b>Experimentation connects analytics to impact:</b> treatment/control design helps quantify conversion, revenue, and client engagement outcomes.</li>
                <li><b>MLOps protects long-term reliability:</b> monitoring drift, score distribution, contactability, and actual outcomes helps determine when the model needs review or retraining.</li>
                <li><b>Luxury-specific success metrics matter:</b> the team should evaluate not only revenue lift, but also contact fatigue, opt-out risk, service sensitivity, and brand experience.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

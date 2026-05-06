import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Luxury Analytics & AI Roadmap Platform",
    page_icon="✨",
    layout="wide"
)

# ------------------------------------------------------------
# Luxury color system
# ------------------------------------------------------------
BG = "#F7F4EF"
CARD = "#FFFFFF"
BORDER = "#E5DDD2"
TEXT = "#111111"
SUBTEXT = "#7C6F63"
ACCENT = "#B79B7A"
ACCENT_LIGHT = "#D8C3A5"
ACCENT_DARK = "#8F7A66"
GRID = "#EEE7DE"

PHASE_COLORS = {
    "Phase 1: Quick Win": ACCENT_LIGHT,
    "Phase 2: Scalable Pilot": ACCENT,
    "Phase 3: Central Industrialization": ACCENT_DARK,
}

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG};
    }}

    .main {{
        background-color: {BG};
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}

    h1, h2, h3, h4 {{
        color: {TEXT};
        font-family: "Helvetica Neue", Arial, sans-serif;
        font-weight: 600;
        letter-spacing: 0.2px;
    }}

    p, div, span, label {{
        color: #2A2A2A;
        font-family: "Helvetica Neue", Arial, sans-serif;
    }}

    .metric-card {{
        background: {CARD};
        padding: 1.1rem;
        border-radius: 14px;
        border: 1px solid {BORDER};
        box-shadow: 0 2px 10px rgba(17,17,17,0.04);
    }}

    .small-label {{
        color: {SUBTEXT};
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        font-weight: 600;
    }}

    .big-number {{
        color: {TEXT};
        font-size: 1.85rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }}

    .insight-box {{
        background: #FCF8F3;
        border-left: 4px solid {ACCENT};
        padding: 1rem 1.1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #2A2A2A;
    }}

    .section-box {{
        background: {CARD};
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid {BORDER};
        box-shadow: 0 2px 10px rgba(17,17,17,0.03);
    }}

    div[data-testid="stDataFrame"] {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 0.25rem;
    }}

    div[data-testid="stMetric"] {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        padding: 0.8rem;
        border-radius: 12px;
    }}

    section[data-testid="stSidebar"] {{
        background-color: #F3EEE7;
        border-right: 1px solid {BORDER};
    }}

    .stDownloadButton button,
    .stButton button {{
        background-color: {TEXT};
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }}

    .stDownloadButton button:hover,
    .stButton button:hover {{
        background-color: #2A2A2A;
        color: white;
    }}

    /* Sidebar filter styling */
    section[data-testid="stSidebar"] label {{
        color: #111111 !important;
        font-weight: 600;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background-color: #FFFFFF !important;
        border: 1px solid #D8C3A5 !important;
        border-radius: 10px !important;
        color: #111111 !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="select"] span {{
        color: #111111 !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="tag"] {{
        background-color: #B79B7A !important;
        border-radius: 8px !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="tag"] span {{
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="tag"] svg {{
        fill: #FFFFFF !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stSlider"] div[role="slider"] {{
        background-color: #111111 !important;
        border: 2px solid #B79B7A !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stSlider"] div[data-testid="stTickBar"] {{
        background-color: #D8C3A5 !important;
    }}

    div[data-baseweb="popover"] {{
        background-color: #FFFFFF !important;
        border: 1px solid #E5DDD2 !important;
        border-radius: 10px !important;
    }}

    div[data-baseweb="popover"] li {{
        color: #111111 !important;
        background-color: #FFFFFF !important;
    }}

    div[data-baseweb="popover"] li:hover {{
        background-color: #FCF8F3 !important;
        color: #111111 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Plotly theme helper
# ------------------------------------------------------------
def apply_luxury_theme(fig):
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(
            family="Helvetica Neue, Arial, sans-serif",
            color=TEXT
        ),
        title_font=dict(size=20, color=TEXT),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#2A2A2A")
        ),
        margin=dict(l=30, r=30, t=60, b=30)
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#D9D0C5",
        tickfont=dict(color="#4A4A4A")
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        tickfont=dict(color="#4A4A4A")
    )
    return fig

# ------------------------------------------------------------
# Synthetic AI roadmap data
# ------------------------------------------------------------
@st.cache_data
def load_roadmap_data():
    data = [
        {
            "Use Case": "Client Next-Best-Action Engine",
            "Business Function": "CRM / Clienteling",
            "Business Value": 95,
            "Feasibility": 82,
            "Data Readiness": 78,
            "Risk Complexity": 38,
            "Global Scalability": 88,
            "Estimated ROI ($K)": 1250,
            "Time to Implement (Weeks)": 10,
            "Phase": "Phase 2: Scalable Pilot",
            "Owner": "CRM Analytics",
            "Success Metric": "Incremental conversion lift",
            "Baseline KPI": "8.5% conversion",
            "Target KPI": "11.0% conversion",
            "Model Output": "Recommended client action and priority score",
            "Business Action": "Advisor outreach, boutique appointment, private preview, nurture journey",
            "Data Inputs": "RFM, category affinity, digital intent, campaign response, service risk, contactability",
            "Validation Rules": "Check score distribution, decile lift, contact eligibility, and advisor feedback loop",
            "Monitoring Needs": "Score drift, conversion lift, action adoption, decile performance",
            "Central Handoff Notes": "Package scoring logic, feature definitions, KPI dictionary, and dashboard wireframes for central CRM data team."
        },
        {
            "Use Case": "Luxury Client LTV Prediction",
            "Business Function": "CRM / Finance",
            "Business Value": 92,
            "Feasibility": 76,
            "Data Readiness": 72,
            "Risk Complexity": 35,
            "Global Scalability": 86,
            "Estimated ROI ($K)": 980,
            "Time to Implement (Weeks)": 12,
            "Phase": "Phase 2: Scalable Pilot",
            "Owner": "Client Insights",
            "Success Metric": "Increase in retained high-value client revenue",
            "Baseline KPI": "$18.2M retained revenue",
            "Target KPI": "$20.5M retained revenue",
            "Model Output": "12-month predicted client value band",
            "Business Action": "Prioritize high-potential clients for personalized retention and growth actions",
            "Data Inputs": "Purchase frequency, lifetime spend, category mix, tenure, boutique engagement, digital behavior",
            "Validation Rules": "Back-test predicted vs. actual value bands and monitor calibration by segment",
            "Monitoring Needs": "Prediction error, segment migration, value band drift, revenue realization",
            "Central Handoff Notes": "Document feature logic and value-band definitions for enterprise client intelligence layer."
        },
        {
            "Use Case": "GenAI Clienteling Message Assistant",
            "Business Function": "Retail / Clienteling",
            "Business Value": 86,
            "Feasibility": 70,
            "Data Readiness": 68,
            "Risk Complexity": 62,
            "Global Scalability": 84,
            "Estimated ROI ($K)": 740,
            "Time to Implement (Weeks)": 14,
            "Phase": "Phase 3: Central Industrialization",
            "Owner": "Retail Excellence",
            "Success Metric": "Advisor productivity and approved message usage",
            "Baseline KPI": "22 messages/advisor/week",
            "Target KPI": "35 messages/advisor/week",
            "Model Output": "Brand-safe draft client message",
            "Business Action": "Advisor reviews, edits, and sends personalized outreach",
            "Data Inputs": "Client profile, purchase context, product interest, event eligibility, tone guidelines",
            "Validation Rules": "Human approval required; block restricted claims; check consent and personalization boundaries",
            "Monitoring Needs": "Approval rate, edit rate, response rate, blocked content, advisor adoption",
            "Central Handoff Notes": "Requires global brand, legal, CRM, and data privacy review before broader rollout."
        },
        {
            "Use Case": "Boutique Traffic Forecasting",
            "Business Function": "Retail Operations",
            "Business Value": 78,
            "Feasibility": 84,
            "Data Readiness": 80,
            "Risk Complexity": 24,
            "Global Scalability": 76,
            "Estimated ROI ($K)": 520,
            "Time to Implement (Weeks)": 8,
            "Phase": "Phase 1: Quick Win",
            "Owner": "Retail Operations",
            "Success Metric": "Forecast accuracy and staffing alignment",
            "Baseline KPI": "72% forecast accuracy",
            "Target KPI": "84% forecast accuracy",
            "Model Output": "Weekly boutique traffic forecast",
            "Business Action": "Adjust staffing, appointment availability, and service coverage",
            "Data Inputs": "Historical traffic, appointments, holidays, local events, weather proxy, campaign calendar",
            "Validation Rules": "Compare forecast vs. actual by boutique and week; monitor holiday/event outliers",
            "Monitoring Needs": "MAPE, under-forecast rate, staffing mismatch, appointment wait time",
            "Central Handoff Notes": "Can be standardized as a reusable forecasting asset for regional retail operations."
        },
        {
            "Use Case": "Inventory Risk & Stockout Prediction",
            "Business Function": "Supply Chain / Merchandising",
            "Business Value": 88,
            "Feasibility": 73,
            "Data Readiness": 66,
            "Risk Complexity": 31,
            "Global Scalability": 82,
            "Estimated ROI ($K)": 1100,
            "Time to Implement (Weeks)": 16,
            "Phase": "Phase 3: Central Industrialization",
            "Owner": "Merchandising Analytics",
            "Success Metric": "Reduction in stockout risk for priority products",
            "Baseline KPI": "14.2% priority-product stockout risk",
            "Target KPI": "9.5% priority-product stockout risk",
            "Model Output": "SKU / boutique stockout risk score",
            "Business Action": "Prioritize replenishment, transfer inventory, or adjust product visibility",
            "Data Inputs": "Inventory, sales velocity, lead time, boutique demand, launch calendar, product category",
            "Validation Rules": "Validate risk scores against actual stockouts and transfer outcomes",
            "Monitoring Needs": "Stockout precision, false positives, replenishment action rate, sell-through impact",
            "Central Handoff Notes": "Requires alignment with global merchandising, supply chain, and product master data standards."
        },
        {
            "Use Case": "Campaign Uplift Measurement",
            "Business Function": "Marketing / CRM",
            "Business Value": 81,
            "Feasibility": 88,
            "Data Readiness": 83,
            "Risk Complexity": 28,
            "Global Scalability": 80,
            "Estimated ROI ($K)": 690,
            "Time to Implement (Weeks)": 7,
            "Phase": "Phase 1: Quick Win",
            "Owner": "Marketing Analytics",
            "Success Metric": "Measured incremental campaign revenue",
            "Baseline KPI": "No standardized uplift view",
            "Target KPI": "Campaign lift tracked for 90% of major CRM campaigns",
            "Model Output": "Treatment/control lift and campaign incrementality estimate",
            "Business Action": "Optimize campaign targeting, cadence, and investment",
            "Data Inputs": "Campaign exposure, control groups, transactions, client segment, channel, timing",
            "Validation Rules": "Check balanced control group, eligibility rules, attribution window, and outlier campaigns",
            "Monitoring Needs": "Lift, conversion, revenue per contacted client, holdout quality, fatigue risk",
            "Central Handoff Notes": "Define standard experiment design and measurement template for CRM teams."
        },
        {
            "Use Case": "Advisor Productivity Intelligence",
            "Business Function": "Retail Excellence",
            "Business Value": 74,
            "Feasibility": 86,
            "Data Readiness": 79,
            "Risk Complexity": 41,
            "Global Scalability": 72,
            "Estimated ROI ($K)": 430,
            "Time to Implement (Weeks)": 9,
            "Phase": "Phase 1: Quick Win",
            "Owner": "Retail Performance",
            "Success Metric": "Clienteling conversion and advisor adoption",
            "Baseline KPI": "18% outreach-to-appointment conversion",
            "Target KPI": "23% outreach-to-appointment conversion",
            "Model Output": "Advisor activity and outcome intelligence dashboard",
            "Business Action": "Coach advisor behaviors and identify best practices",
            "Data Inputs": "Outreach volume, appointment creation, conversion, sales, segment mix, client tenure",
            "Validation Rules": "Normalize by boutique, advisor tenure, client book size, and segment mix",
            "Monitoring Needs": "Adoption, conversion, productivity distribution, exception flags",
            "Central Handoff Notes": "Ensure metric definitions are aligned with retail leadership and HR-sensitive governance."
        },
        {
            "Use Case": "Service Recovery Prioritization",
            "Business Function": "Client Care",
            "Business Value": 83,
            "Feasibility": 75,
            "Data Readiness": 70,
            "Risk Complexity": 44,
            "Global Scalability": 79,
            "Estimated ROI ($K)": 610,
            "Time to Implement (Weeks)": 11,
            "Phase": "Phase 2: Scalable Pilot",
            "Owner": "Client Care Analytics",
            "Success Metric": "Reduction in unresolved high-value service cases",
            "Baseline KPI": "21% unresolved after 14 days",
            "Target KPI": "13% unresolved after 14 days",
            "Model Output": "Service escalation priority score",
            "Business Action": "Prioritize recovery outreach and escalation for high-risk/high-value clients",
            "Data Inputs": "Service case history, client value, delay duration, sentiment proxy, product category, contact history",
            "Validation Rules": "Review false positives, fairness by client tier, and manual override outcomes",
            "Monitoring Needs": "Resolution time, satisfaction proxy, repeat case rate, escalation accuracy",
            "Central Handoff Notes": "Partner with client care, CRM, and data privacy teams for sensitive case handling."
        }
    ]

    df = pd.DataFrame(data)

    df["Priority Score"] = (
        df["Business Value"] * 0.30
        + df["Feasibility"] * 0.20
        + df["Data Readiness"] * 0.15
        + df["Global Scalability"] * 0.20
        + (100 - df["Risk Complexity"]) * 0.15
    ).round(1)

    df["ROI per Week"] = (
        df["Estimated ROI ($K)"] / df["Time to Implement (Weeks)"]
    ).round(1)

    return df


roadmap = load_roadmap_data()

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("Luxury Analytics & AI Roadmap Prioritization Platform")
st.caption(
    "Portfolio project demonstrating analytics product strategy, AI use-case prioritization, "
    "value realization, KPI governance, and scalable central-team handoff for luxury retail."
)

# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.header("Roadmap Filters")

selected_phase = st.sidebar.multiselect(
    "Phase",
    options=sorted(roadmap["Phase"].unique()),
    default=sorted(roadmap["Phase"].unique())
)

selected_function = st.sidebar.multiselect(
    "Business Function",
    options=sorted(roadmap["Business Function"].unique()),
    default=sorted(roadmap["Business Function"].unique())
)

min_priority = st.sidebar.slider(
    "Minimum Priority Score",
    0,
    100,
    0
)

filtered = roadmap[
    roadmap["Phase"].isin(selected_phase)
    & roadmap["Business Function"].isin(selected_function)
    & (roadmap["Priority Score"] >= min_priority)
].copy()

# ------------------------------------------------------------
# Executive KPI cards
# ------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='small-label'>Total Use Cases</div>
            <div class='big-number'>{len(filtered)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='small-label'>Avg Priority Score</div>
            <div class='big-number'>{filtered['Priority Score'].mean():.1f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='small-label'>Estimated ROI</div>
            <div class='big-number'>${filtered['Estimated ROI ($K)'].sum():,.0f}K</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='small-label'>Avg Time to Implement</div>
            <div class='big-number'>{filtered['Time to Implement (Weeks)'].mean():.1f} wks</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Roadmap",
    "Prioritization Matrix",
    "Value Realization",
    "Handoff Documentation",
    "Data Table"
])

with tab1:
    st.subheader("AI Roadmap by Phase")

    st.markdown(
        """
        <div class='insight-box'>
        <b>Executive readout:</b> Prioritize quick wins with high feasibility while moving high-value, scalable AI use cases into pilot and central industrialization phases.
        </div>
        """,
        unsafe_allow_html=True
    )

    phase_summary = filtered.groupby("Phase", as_index=False).agg(
        Use_Cases=("Use Case", "count"),
        Avg_Priority=("Priority Score", "mean"),
        Estimated_ROI_K=("Estimated ROI ($K)", "sum"),
        Avg_Time_Weeks=("Time to Implement (Weeks)", "mean")
    )

    phase_summary["Avg_Priority"] = phase_summary["Avg_Priority"].round(1)
    phase_summary["Avg_Time_Weeks"] = phase_summary["Avg_Time_Weeks"].round(1)

    fig_phase = px.bar(
        phase_summary,
        x="Phase",
        y="Estimated_ROI_K",
        text="Estimated_ROI_K",
        title="Estimated ROI by Roadmap Phase",
        labels={"Estimated_ROI_K": "Estimated ROI ($K)"},
        color="Phase",
        color_discrete_map=PHASE_COLORS
    )

    fig_phase.update_traces(
        texttemplate="$%{text:,.0f}K",
        textposition="outside"
    )
    fig_phase.update_layout(
        height=420,
        showlegend=False
    )
    fig_phase = apply_luxury_theme(fig_phase)

    st.plotly_chart(fig_phase, use_container_width=True)

    st.dataframe(
        phase_summary.rename(
            columns={
                "Use_Cases": "Use Cases",
                "Avg_Priority": "Avg Priority Score",
                "Estimated_ROI_K": "Estimated ROI ($K)",
                "Avg_Time_Weeks": "Avg Time to Implement (Weeks)"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.subheader("Use-Case Prioritization Matrix")

    st.markdown(
        """
        <div class='insight-box'>
        <b>Scoring logic:</b> Priority Score weighs business value, feasibility, data readiness, global scalability, and lower implementation risk.
        </div>
        """,
        unsafe_allow_html=True
    )

    fig_matrix = px.scatter(
        filtered,
        x="Feasibility",
        y="Business Value",
        size="Estimated ROI ($K)",
        color="Phase",
        hover_name="Use Case",
        hover_data=[
            "Priority Score",
            "Data Readiness",
            "Risk Complexity",
            "Global Scalability",
            "Time to Implement (Weeks)"
        ],
        title="Business Value vs. Feasibility",
        size_max=45,
        color_discrete_map=PHASE_COLORS
    )

    fig_matrix.add_hline(
        y=80,
        line_dash="dash",
        line_color="#CBB8A0",
        opacity=0.7
    )
    fig_matrix.add_vline(
        x=80,
        line_dash="dash",
        line_color="#CBB8A0",
        opacity=0.7
    )
    fig_matrix.update_layout(height=560)
    fig_matrix = apply_luxury_theme(fig_matrix)

    st.plotly_chart(fig_matrix, use_container_width=True)

    top_priority = filtered.sort_values("Priority Score", ascending=False).head(5)

    st.markdown("### Top Priority Use Cases")

    st.dataframe(
        top_priority[
            [
                "Use Case",
                "Business Function",
                "Phase",
                "Priority Score",
                "Estimated ROI ($K)",
                "Time to Implement (Weeks)",
                "ROI per Week"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

with tab3:
    st.subheader("Value Realization & KPI Ownership")

    selected_use_case = st.selectbox(
        "Select a use case",
        filtered["Use Case"].tolist()
    )

    row = filtered[filtered["Use Case"] == selected_use_case].iloc[0]

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Priority Score", f"{row['Priority Score']:.1f}")

    with col_b:
        st.metric("Estimated ROI", f"${row['Estimated ROI ($K)']:,.0f}K")

    with col_c:
        st.metric("Implementation Timeline", f"{row['Time to Implement (Weeks)']} weeks")

    st.markdown("### KPI Definition")

    kpi_col1, kpi_col2 = st.columns(2)

    with kpi_col1:
        st.markdown(
            f"""
            <div class='section-box'>
            <b>Success Metric</b><br>{row['Success Metric']}<br><br>
            <b>Baseline KPI</b><br>{row['Baseline KPI']}<br><br>
            <b>Target KPI</b><br>{row['Target KPI']}
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class='section-box'>
            <b>Business Owner</b><br>{row['Owner']}<br><br>
            <b>Model Output</b><br>{row['Model Output']}<br><br>
            <b>Business Action</b><br>{row['Business Action']}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### KPI Radar")

    radar_categories = [
        "Business Value",
        "Feasibility",
        "Data Readiness",
        "Global Scalability",
        "Risk Adjusted"
    ]

    radar_values = [
        row["Business Value"],
        row["Feasibility"],
        row["Data Readiness"],
        row["Global Scalability"],
        100 - row["Risk Complexity"]
    ]

    fig_radar = go.Figure()

    fig_radar.add_trace(
        go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_categories + [radar_categories[0]],
            fill="toself",
            name=selected_use_case,
            line=dict(color=ACCENT, width=2),
            fillcolor="rgba(183,155,122,0.35)"
        )
    )

    fig_radar.update_layout(
        polar=dict(
            bgcolor=CARD,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=GRID,
                linecolor="#D9D0C5",
                tickfont=dict(color="#4A4A4A")
            ),
            angularaxis=dict(
                tickfont=dict(color="#4A4A4A")
            )
        ),
        paper_bgcolor=BG,
        font=dict(
            family="Helvetica Neue, Arial, sans-serif",
            color=TEXT
        ),
        showlegend=False,
        height=460,
        margin=dict(l=30, r=30, t=40, b=30)
    )

    st.plotly_chart(fig_radar, use_container_width=True)

with tab4:
    st.subheader("Production-Style Handoff Documentation")

    doc_case = st.selectbox(
        "Select use case for handoff documentation",
        filtered["Use Case"].tolist(),
        key="doc_case"
    )

    doc = filtered[filtered["Use Case"] == doc_case].iloc[0]

    st.markdown(
        f"""
        <div class='section-box'>
        <h3>{doc['Use Case']}</h3>
        <p><b>Business Function:</b> {doc['Business Function']}</p>
        <p><b>Roadmap Phase:</b> {doc['Phase']}</p>
        <p><b>Business Owner:</b> {doc['Owner']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f"""
            <div class='section-box'>
            <h4>Problem Statement</h4>
            <p>Develop a scalable analytics or AI solution to support <b>{doc['Business Function']}</b> decisions through measurable, reusable data products.</p>

            <h4>Data Inputs</h4>
            <p>{doc['Data Inputs']}</p>

            <h4>Model / Analytical Output</h4>
            <p>{doc['Model Output']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class='section-box'>
            <h4>Validation Rules</h4>
            <p>{doc['Validation Rules']}</p>

            <h4>Monitoring Requirements</h4>
            <p>{doc['Monitoring Needs']}</p>

            <h4>Central-Team Handoff Notes</h4>
            <p>{doc['Central Handoff Notes']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.download_button(
        label="Download Handoff Summary as CSV",
        data=filtered[filtered["Use Case"] == doc_case].to_csv(index=False),
        file_name=f"{doc_case.lower().replace(' ', '_').replace('/', '_')}_handoff_summary.csv",
        mime="text/csv"
    )

with tab5:
    st.subheader("Roadmap Data Table")

    st.dataframe(
        filtered.sort_values("Priority Score", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Download Full Roadmap Data",
        data=filtered.to_csv(index=False),
        file_name="luxury_ai_roadmap_prioritization_data.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.divider()

st.caption(
    "Portfolio note: This dashboard uses synthetic data and is designed to demonstrate analytics "
    "product strategy, AI prioritization logic, KPI governance, value-realization tracking, and "
    "scalable handoff documentation for luxury retail analytics use cases."
)

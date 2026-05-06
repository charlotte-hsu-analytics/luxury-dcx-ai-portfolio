import streamlit as st

st.set_page_config(
    page_title="Charlotte Hsu | Luxury Analytics Portfolio",
    page_icon="✨",
    layout="wide"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F7F4EF;
        color: #111111;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        color: #111111;
        font-family: "Helvetica Neue", Arial, sans-serif;
        font-weight: 600;
    }

    p, div, span {
        font-family: "Helvetica Neue", Arial, sans-serif;
    }

    .hero-card {
        background: #FFFFFF;
        padding: 2.5rem;
        border-radius: 24px;
        border: 1px solid #E5DDD2;
        box-shadow: 0 4px 18px rgba(17,17,17,0.05);
    }

    .tag {
        display: inline-block;
        background: #F3EEE7;
        color: #7C6F63;
        border: 1px solid #D8C3A5;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    .project-card {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid #E5DDD2;
        box-shadow: 0 2px 12px rgba(17,17,17,0.04);
        min-height: 300px;
    }

    .small-label {
        color: #7C6F63;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        font-weight: 700;
    }

    .button-link {
        display: inline-block;
        background: #111111;
        color: #FFFFFF !important;
        padding: 0.7rem 1.1rem;
        border-radius: 12px;
        text-decoration: none;
        font-weight: 600;
        margin-top: 1rem;
    }

    .button-link-light {
        display: inline-block;
        background: #FFFFFF;
        color: #111111 !important;
        border: 1px solid #D8C3A5;
        padding: 0.7rem 1.1rem;
        border-radius: 12px;
        text-decoration: none;
        font-weight: 600;
        margin-top: 1rem;
        margin-left: 0.5rem;
    }

    .section-box {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid #E5DDD2;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Hero
# -----------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="small-label">Luxury Analytics · AI Data Products · Digital Client Experience</div>
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">Charlotte Hsu</h1>
        <h3 style="color:#4A4A4A; font-weight:400; line-height:1.5;">
            Analytics & AI Data Product Leader building predictive models, executive dashboards,
            KPI frameworks, and luxury retail analytics portfolios that translate data into business action.
        </h3>
        <div style="margin-top:1.5rem;">
            <span class="tag">Python</span>
            <span class="tag">SQL</span>
            <span class="tag">Streamlit</span>
            <span class="tag">Machine Learning</span>
            <span class="tag">Power BI</span>
            <span class="tag">Tableau</span>
            <span class="tag">Luxury Retail Analytics</span>
        </div>
        <a class="button-link" href="https://github.com/hsucharlotte" target="_blank">GitHub</a>
        <a class="button-link-light" href="mailto:Hsinchuen_hsu@fitnyc.edu">Contact Me</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# -----------------------------
# About
# -----------------------------
st.markdown("## About")
st.markdown(
    """
I bring **10+ years of analytics experience** across enterprise dashboards, predictive modeling,
KPI governance, forecasting, and stakeholder-facing decision support.

My portfolio focuses on **luxury retail analytics** use cases including client segmentation,
repeat-purchase prediction, next-best-action, GenAI clienteling, experimentation,
MLOps monitoring, and AI roadmap prioritization.
"""
)

# -----------------------------
# Featured Projects
# -----------------------------
st.markdown("## Featured Projects")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="project-card">
            <div class="small-label">Data Science · Client Intelligence · GenAI</div>
            <h3>Luxury DCX AI Portfolio</h3>
            <p>
            End-to-end luxury Digital Client Experience analytics portfolio using synthetic client data
            to demonstrate segmentation, repeat-purchase prediction, next-best-action recommendations,
            brand-safe GenAI clienteling, experimentation, and MLOps monitoring.
            </p>
            <span class="tag">Python</span>
            <span class="tag">Streamlit</span>
            <span class="tag">scikit-learn</span>
            <span class="tag">MLOps</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="project-card">
            <div class="small-label">BI Strategy · Retail Performance</div>
            <h3>Owned Retail Performance Excellence Dashboard</h3>
            <p>
            Simulated owned retail dashboard modeling boutique-level performance across Fashion Boutiques,
            Fragrance & Beauty, and Watches & Fine Jewelry, with KPI views for sales, clienteling conversion,
            advisor productivity, and payout exception risk.
            </p>
            <span class="tag">Retail KPIs</span>
            <span class="tag">Executive BI</span>
            <span class="tag">Plotly</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="project-card">
            <div class="small-label">AI Product Strategy · Roadmap</div>
            <h3>Luxury Analytics & AI Roadmap Platform</h3>
            <p>
            Portfolio platform demonstrating AI use-case prioritization, value-realization tracking,
            KPI ownership, and production-style handoff documentation for scalable luxury retail analytics initiatives.
            </p>
            <span class="tag">AI Roadmap</span>
            <span class="tag">KPI Governance</span>
            <span class="tag">Value Realization</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Capabilities
# -----------------------------
st.markdown("## Capabilities")

cap1, cap2 = st.columns(2)

with cap1:
    st.markdown(
        """
        <div class="section-box">
        <h3>Data Science & AI</h3>
        <p>Predictive modeling, segmentation, scoring frameworks, feature engineering, model validation, GenAI use cases.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with cap2:
    st.markdown(
        """
        <div class="section-box">
        <h3>BI & Executive Analytics</h3>
        <p>Power BI, Tableau, KPI frameworks, dashboard adoption, semantic layers, executive storytelling.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

cap3, cap4 = st.columns(2)

with cap3:
    st.markdown(
        """
        <div class="section-box">
        <h3>Data Product & Governance</h3>
        <p>SQL, Python, Databricks, reusable data assets, validation logic, methodology documentation.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with cap4:
    st.markdown(
        """
        <div class="section-box">
        <h3>Luxury Retail Analytics</h3>
        <p>Clienteling, CRM analytics, retail KPIs, assortment planning, pricing strategy, fine jewelry knowledge.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Experience
# -----------------------------
st.markdown("## Experience Highlights")

st.markdown(
    """
- **NYU Langone Health — Data & Analytics Manager:** Lead enterprise analytics, forecasting, KPI governance, and decision-support tools supporting a $500M portfolio.
- **NYU Langone Health — Senior Data Analyst:** Built pipeline analytics, scoring dashboards, segmentation views, and reporting automation.
- **NYU University Development & Alumni Relations — BI Developer / Data Analyst:** Built 50+ dashboards, migrated 150+ legacy reports, and supported a $200M+ portfolio.
- **Inteplast Group — Business / Data Analyst:** Supported cost, procurement, supplier, and inventory analytics across six manufacturing facilities.
"""
)

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.markdown(
    """
**Contact:** Hsinchuen_hsu@fitnyc.edu  
**GitHub:** github.com/hsucharlotte
"""
)

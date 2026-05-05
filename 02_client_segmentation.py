"""
Luxury DCX Client Intelligence
Step 2: Client Segmentation

Run:
    python3 02_client_segmentation.py

Input:
    data/dcx_client_data.csv

Output:
    data/dcx_client_segmented.csv

This script creates:
- RFM scores
- engagement score
- digital intent score
- campaign engagement score
- service risk score
- luxury value band
- client segment
- recommended action
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


INPUT_FILE = "data/dcx_client_data.csv"
OUTPUT_FILE = "data/dcx_client_segmented.csv"


def minmax_scale(series: pd.Series) -> pd.Series:
    """Scale a numeric series to 0-1."""
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(0, index=series.index)

    return (series - min_value) / (max_value - min_value)


def assign_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Create RFM scores for purchase relationship strength."""

    df = df.copy()

    df["recency_score"] = pd.qcut(
        df["last_purchase_days"].rank(method="first", ascending=False),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    df["frequency_score"] = pd.qcut(
        df["purchase_frequency_12m"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    df["monetary_score"] = pd.qcut(
        df["total_spend_12m"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    df["rfm_score"] = (
        df["recency_score"] + df["frequency_score"] + df["monetary_score"]
    )

    return df


def add_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """Create a combined engagement score using boutique, advisor, event, and digital behavior."""

    df = df.copy()

    df["engagement_score"] = (
        0.12 * minmax_scale(df["boutique_visits_12m"])
        + 0.10 * minmax_scale(df["online_visits_90d"])
        + 0.12 * minmax_scale(df["appointment_count_12m"])
        + 0.14 * minmax_scale(df["clienteling_touchpoints_90d"])
        + 0.08 * minmax_scale(df["email_open_rate"])
        + 0.08 * minmax_scale(df["event_attendance_12m"])
        + 0.06 * minmax_scale(df["advisor_follow_up_flag"])
        + 0.06 * minmax_scale(df["app_engagement_score"])
        + 0.06 * minmax_scale(df["wishlist_additions_90d"])
        + 0.06 * minmax_scale(df["campaign_click_30d"])
        + 0.06 * minmax_scale(df["appointment_completed_flag"])
        + 0.06 * minmax_scale(df["client_contactable_flag"])
    )

    df["engagement_score"] = np.round(df["engagement_score"] * 100, 1)

    return df


def add_digital_intent_score(df: pd.DataFrame) -> pd.DataFrame:
    """Create a digital intent score using browsing, wishlist, campaign, and app behavior."""

    df = df.copy()

    df["digital_intent_score"] = (
        0.18 * minmax_scale(df["product_page_views_90d"])
        + 0.14 * minmax_scale(df["wishlist_additions_90d"])
        + 0.12 * minmax_scale(df["high_intent_browsing_flag"])
        + 0.14 * minmax_scale(df["app_engagement_score"])
        + 0.12 * minmax_scale(df["digital_campaign_click_rate"])
        + 0.10 * minmax_scale(df["campaign_click_30d"])
        + 0.08 * minmax_scale(df["online_visits_90d"])
        + 0.06 * minmax_scale(df["fashion_page_views_90d"])
        + 0.06 * minmax_scale(df["beauty_page_views_90d"])
    )

    # Add extra WFJ intent weight.
    df["digital_intent_score"] = df["digital_intent_score"] + (
        0.10 * minmax_scale(df["wfj_page_views_90d"])
    )

    df["digital_intent_score"] = np.round(
        np.clip(df["digital_intent_score"], 0, 1) * 100,
        1,
    )

    return df


def add_campaign_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """Create campaign response score."""

    df = df.copy()

    df["campaign_engagement_score"] = (
        0.25 * df["campaign_sent_30d"]
        + 0.25 * df["campaign_open_30d"]
        + 0.30 * df["campaign_click_30d"]
        + 0.20 * df["campaign_conversion_30d"]
    )

    df["campaign_engagement_score"] = np.round(
        df["campaign_engagement_score"] * 100,
        1,
    )

    return df


def add_service_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Create service risk score to flag clients who may need relationship recovery."""

    df = df.copy()

    satisfaction_risk = 1 - minmax_scale(df["client_satisfaction_score"])

    df["service_risk_score"] = (
        0.35 * minmax_scale(df["return_or_exchange_12m"])
        + 0.25 * minmax_scale(df["estimated_return_rate"])
        + 0.25 * satisfaction_risk
        + 0.15 * minmax_scale(df["service_recovery_flag"])
    )

    df["service_risk_score"] = np.round(df["service_risk_score"] * 100, 1)

    return df


def add_luxury_value_band(df: pd.DataFrame) -> pd.DataFrame:
    """Create business-friendly luxury value band."""

    df = df.copy()

    def assign_value_band(row: pd.Series) -> str:
        if row["client_tier"] == "Top Client" or row["lifetime_spend"] >= 150000:
            return "Ultra High Value"
        if row["client_tier"] == "VIP" or row["lifetime_spend"] >= 75000:
            return "High Value"
        if row["client_tier"] == "High Potential" or row["lifetime_spend"] >= 30000:
            return "Growth Value"
        if row["lifetime_spend"] >= 10000:
            return "Core Value"
        return "Entry Value"

    df["luxury_value_band"] = df.apply(assign_value_band, axis=1)

    return df


def assign_client_segment(row: pd.Series) -> str:
    """Assign luxury DCX client segment."""

    client_tier = row["client_tier"]
    primary_division = row["primary_division"]
    primary_category = row["primary_category"]
    secondary_interest = row["secondary_category_interest"]
    high_value = row["high_value_client_flag"]
    lifetime_spend = row["lifetime_spend"]
    last_purchase_days = row["last_purchase_days"]
    purchase_frequency = row["purchase_frequency_12m"]
    engagement = row["engagement_score"]
    digital_intent = row["digital_intent_score"]
    service_risk = row["service_risk_score"]
    contactable = row["client_contactable_flag"]
    campaign_conversion = row["campaign_conversion_30d"]
    appointment_conversion = row["appointment_conversion_flag"]
    wfj_focus = row["boutique_wfj_focus_flag"]

    if (
        client_tier in ["VIP", "Top Client"]
        and service_risk >= 45
        and lifetime_spend >= 50000
    ):
        return "VIP Service Recovery"

    if (
        client_tier in ["VIP", "Top Client"]
        and last_purchase_days <= 90
        and purchase_frequency >= 4
        and high_value == 1
    ):
        return "VIP Loyalist"

    if (
        client_tier in ["VIP", "Top Client"]
        and last_purchase_days > 120
        and lifetime_spend >= 40000
    ):
        return "At-Risk VIP"

    if (
        primary_division == "Watches & Fine Jewelry"
        and lifetime_spend >= 25000
        and (row["appointment_count_12m"] >= 1 or wfj_focus == 1)
    ):
        return "WFJ Opportunity Client"

    if (
        secondary_interest in ["Fine Jewelry Interest", "Watches"]
        and lifetime_spend >= 20000
        and digital_intent >= 35
    ):
        return "Cross-Category WFJ Prospect"

    if (
        client_tier == "High Potential"
        and engagement >= 35
        and last_purchase_days <= 120
    ):
        return "High-Potential Client"

    if (
        primary_division == "Fashion"
        and primary_category in ["Handbags", "Ready-to-Wear"]
        and (row["total_spend_12m"] >= 3000 or lifetime_spend >= 15000)
    ):
        return "Fashion-First Client"

    if (
        primary_division == "Fragrance & Beauty"
        and digital_intent >= 40
        and row["campaign_click_30d"] == 1
    ):
        return "Beauty-First Digital Client"

    if (
        primary_division == "Fragrance & Beauty"
        and purchase_frequency >= 3
        and engagement >= 25
    ):
        return "Beauty-First Client"

    if (
        row["cross_category_flag"] == 1
        and engagement >= 35
        and purchase_frequency >= 2
    ):
        return "Cross-Category Growth Client"

    if (
        row["preferred_channel"] in ["Online", "Omnichannel"]
        and digital_intent >= 45
        and row["wishlist_additions_90d"] >= 2
    ):
        return "Digital High-Intent Client"

    if (
        campaign_conversion == 1
        or appointment_conversion == 1
    ):
        return "Recently Converted Client"

    if (
        contactable == 0
    ):
        return "Non-Contactable Client"

    if (
        client_tier == "New"
        and purchase_frequency <= 1
        and last_purchase_days <= 180
    ):
        return "New Client"

    if (
        engagement < 15
        and last_purchase_days > 150
        and row["advisor_follow_up_flag"] == 0
    ):
        return "Low-Engagement Client"

    return "Core Client"


def add_recommended_action(df: pd.DataFrame) -> pd.DataFrame:
    """Map each segment to a recommended business action."""

    df = df.copy()

    action_map = {
        "VIP Service Recovery": "Prioritize service recovery before any sales outreach.",
        "VIP Loyalist": "Invite to private preview, exclusive appointment, or high-touch boutique experience.",
        "At-Risk VIP": "Prioritize advisor outreach with personalized re-engagement message.",
        "WFJ Opportunity Client": "Recommend Watches & Fine Jewelry advisor follow-up and appointment consultation.",
        "Cross-Category WFJ Prospect": "Introduce Watches & Fine Jewelry discovery journey based on browsing and cross-category signals.",
        "High-Potential Client": "Invite to boutique appointment or curated cross-category discovery journey.",
        "Fashion-First Client": "Recommend styling appointment, new collection preview, or category expansion outreach.",
        "Beauty-First Digital Client": "Target with personalized beauty launch journey and digital-to-boutique conversion path.",
        "Beauty-First Client": "Target with fragrance, skincare, or beauty replenishment campaign.",
        "Cross-Category Growth Client": "Recommend cross-category storytelling and personalized product discovery.",
        "Digital High-Intent Client": "Use digital retargeting, wishlist follow-up, and appointment conversion prompt.",
        "Recently Converted Client": "Move to post-purchase nurture journey and monitor for next best offer.",
        "Non-Contactable Client": "Do not contact directly; monitor behavior and rely on owned-channel experience.",
        "New Client": "Enroll in welcome journey with brand story, service introduction, and first follow-up touchpoint.",
        "Low-Engagement Client": "Use low-pressure reactivation before assigning high-touch advisor effort.",
        "Core Client": "Maintain regular lifecycle communication and monitor for upgrade or churn signals.",
    }

    df["recommended_action"] = df["client_segment"].map(action_map)

    return df


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Run 01_generate_dcx_client_data.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    df = assign_rfm_scores(df)
    df = add_engagement_score(df)
    df = add_digital_intent_score(df)
    df = add_campaign_engagement_score(df)
    df = add_service_risk_score(df)
    df = add_luxury_value_band(df)
    df["client_segment"] = df.apply(assign_client_segment, axis=1)
    df = add_recommended_action(df)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Created segmented dataset: {OUTPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nClient segment distribution:")
    print(df["client_segment"].value_counts(normalize=True).round(3))

    print("\nLuxury value band distribution:")
    print(df["luxury_value_band"].value_counts(normalize=True).round(3))

    print("\nAverage lifetime spend by segment:")
    print(
        df.groupby("client_segment")["lifetime_spend"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )

    print("\nAverage digital intent score by segment:")
    print(
        df.groupby("client_segment")["digital_intent_score"]
        .mean()
        .sort_values(ascending=False)
        .round(1)
    )

    print("\nAverage service risk score by segment:")
    print(
        df.groupby("client_segment")["service_risk_score"]
        .mean()
        .sort_values(ascending=False)
        .round(1)
    )


if __name__ == "__main__":
    main()

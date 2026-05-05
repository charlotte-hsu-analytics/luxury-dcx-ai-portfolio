"""
Luxury DCX Client Intelligence
Step 4: Next-Best-Action Recommendations

Run:
    python3 04_next_best_action.py

Input:
    data/dcx_client_model_scored.csv

Output:
    data/dcx_client_next_best_action.csv

Purpose:
    Convert predictive scores, client segments, contactability, service risk,
    and business priority into action-oriented clienteling recommendations.
"""

from __future__ import annotations

import os
import pandas as pd


INPUT_FILE = "data/dcx_client_model_scored.csv"
OUTPUT_FILE = "data/dcx_client_next_best_action.csv"


def assign_next_best_action(row: pd.Series) -> str:
    """Assign a business-ready next-best-action."""

    segment = row["client_segment"]
    priority = row["model_priority_group"]
    probability = row["predicted_repeat_purchase_probability"]
    division = row["primary_division"]
    category = row["primary_category"]
    contactable = row["client_contactable_flag"]
    service_recovery = row["service_recovery_flag"]
    service_risk = row["service_risk_score"]
    high_value = row["high_value_client_flag"]
    campaign_conversion = row["campaign_conversion_30d"]
    appointment_conversion = row["appointment_conversion_flag"]
    high_intent = row["high_intent_browsing_flag"]
    wfj_views = row["wfj_page_views_90d"]
    preferred_channel = row["preferred_channel"]

    # Contactability first: do not recommend outbound outreach when not contactable.
    if contactable == 0:
        return "Do Not Contact / Monitor"

    # Service recovery should override commercial outreach.
    if segment == "VIP Service Recovery" or (
        service_recovery == 1 and high_value == 1 and service_risk >= 45
    ):
        return "At-Risk VIP Service Recovery"

    if segment == "At-Risk VIP":
        return "At-Risk VIP Re-Engagement"

    if segment == "VIP Loyalist" and priority == "Top Priority":
        return "Private Preview Invitation"

    if segment == "WFJ Opportunity Client":
        return "WFJ Advisor Follow-Up"

    if segment == "Cross-Category WFJ Prospect":
        return "WFJ Discovery Journey"

    if (
        division == "Watches & Fine Jewelry"
        and probability >= 0.55
        and (wfj_views >= 6 or high_intent == 1)
    ):
        return "WFJ Appointment Outreach"

    if segment == "Fashion-First Client":
        return "Fashion Styling Appointment"

    if segment == "Beauty-First Digital Client":
        return "Beauty Digital-to-Boutique Journey"

    if segment == "Beauty-First Client":
        return "Beauty Replenishment Campaign"

    if segment == "Digital High-Intent Client":
        return "Digital High-Intent Retargeting"

    if segment == "High-Potential Client":
        return "Boutique Appointment Outreach"

    if segment == "Cross-Category Growth Client":
        return "Cross-Category Product Discovery"

    if segment == "Recently Converted Client" or campaign_conversion == 1 or appointment_conversion == 1:
        return "Post-Purchase Nurture Journey"

    if segment == "New Client":
        return "Welcome Journey"

    if segment == "Low-Engagement Client":
        return "Reactivation Campaign"

    if priority == "Top Priority":
        return "High-Touch Clienteling Outreach"

    if priority == "High Priority":
        if preferred_channel in ["Boutique", "Omnichannel"]:
            return "Boutique Appointment Outreach"
        return "Personalized Digital Product Discovery"

    if priority == "Medium Priority":
        return "Lifecycle Nurture Campaign"

    return "Low-Touch Brand Engagement"


def assign_recommended_outreach_channel(row: pd.Series) -> str:
    """Recommend operational outreach channel."""

    action = row["next_best_action"]
    contactable = row["client_contactable_flag"]
    preferred_contact = row["preferred_contact_method"]
    preferred_channel = row["preferred_channel"]

    if contactable == 0 or action == "Do Not Contact / Monitor":
        return "No Direct Outreach"

    boutique_actions = [
        "At-Risk VIP Service Recovery",
        "At-Risk VIP Re-Engagement",
        "Private Preview Invitation",
        "WFJ Advisor Follow-Up",
        "WFJ Discovery Journey",
        "WFJ Appointment Outreach",
        "Fashion Styling Appointment",
        "Boutique Appointment Outreach",
        "High-Touch Clienteling Outreach",
    ]

    digital_actions = [
        "Beauty Digital-to-Boutique Journey",
        "Beauty Replenishment Campaign",
        "Digital High-Intent Retargeting",
        "Personalized Digital Product Discovery",
        "Lifecycle Nurture Campaign",
        "Welcome Journey",
        "Reactivation Campaign",
        "Low-Touch Brand Engagement",
        "Post-Purchase Nurture Journey",
    ]

    if action in boutique_actions:
        return "Client Advisor / Boutique"

    if action in digital_actions:
        if preferred_contact == "Email + SMS":
            return "Email + SMS"
        if preferred_contact in ["Email", "SMS"]:
            return preferred_contact
        return "Email / Digital"

    if preferred_channel == "Omnichannel":
        return "Omnichannel"

    return preferred_channel


def assign_business_priority(row: pd.Series) -> str:
    """Translate model priority and action type into operational timing."""

    action = row["next_best_action"]
    model_priority = row["model_priority_group"]

    immediate_actions = [
        "At-Risk VIP Service Recovery",
        "At-Risk VIP Re-Engagement",
        "Private Preview Invitation",
        "WFJ Advisor Follow-Up",
        "WFJ Appointment Outreach",
        "High-Touch Clienteling Outreach",
    ]

    this_week_actions = [
        "WFJ Discovery Journey",
        "Fashion Styling Appointment",
        "Boutique Appointment Outreach",
        "Beauty Digital-to-Boutique Journey",
        "Digital High-Intent Retargeting",
        "Cross-Category Product Discovery",
        "Personalized Digital Product Discovery",
    ]

    monitor_actions = [
        "Do Not Contact / Monitor",
        "Low-Touch Brand Engagement",
    ]

    if action in immediate_actions:
        return "Immediate"

    if action in this_week_actions:
        return "This Week"

    if action in monitor_actions:
        return "Monitor"

    if model_priority == "Top Priority":
        return "Immediate"

    if model_priority == "High Priority":
        return "This Week"

    if model_priority == "Medium Priority":
        return "This Month"

    return "Monitor"


def assign_action_rationale(row: pd.Series) -> str:
    """Create a plain-English rationale for the recommended action."""

    return (
        f"{row['client_segment']} in {row['primary_division']} / {row['primary_category']} "
        f"with {row['model_priority_group'].lower()} model priority, "
        f"{row['predicted_repeat_purchase_probability']:.1%} predicted repeat-purchase probability, "
        f"${row['lifetime_spend']:,.0f} lifetime spend, "
        f"{row['digital_intent_score']:.1f} digital intent score, "
        f"{row['service_risk_score']:.1f} service risk score, "
        f"and contactability status: {row['preferred_contact_method']}."
    )


def estimate_action_revenue(row: pd.Series) -> float:
    """
    Estimate revenue opportunity for the recommended action.

    This is a portfolio simulation metric:
    predicted probability × average order value × action multiplier × contactability/service adjustment.
    """

    base = row["predicted_repeat_purchase_probability"] * row["average_order_value"]

    action = row["next_best_action"]

    action_multiplier = {
        "Private Preview Invitation": 1.45,
        "At-Risk VIP Re-Engagement": 1.20,
        "At-Risk VIP Service Recovery": 0.85,
        "WFJ Advisor Follow-Up": 1.55,
        "WFJ Discovery Journey": 1.35,
        "WFJ Appointment Outreach": 1.50,
        "Fashion Styling Appointment": 1.25,
        "Boutique Appointment Outreach": 1.20,
        "Beauty Digital-to-Boutique Journey": 1.10,
        "Beauty Replenishment Campaign": 0.85,
        "Digital High-Intent Retargeting": 1.05,
        "Cross-Category Product Discovery": 1.20,
        "Post-Purchase Nurture Journey": 0.75,
        "Welcome Journey": 0.65,
        "Reactivation Campaign": 0.55,
        "Lifecycle Nurture Campaign": 0.60,
        "Personalized Digital Product Discovery": 0.75,
        "Low-Touch Brand Engagement": 0.35,
        "Do Not Contact / Monitor": 0.10,
        "High-Touch Clienteling Outreach": 1.25,
    }.get(action, 0.70)

    contactability_multiplier = 1.0 if row["client_contactable_flag"] == 1 else 0.25
    service_multiplier = 0.70 if row["service_recovery_flag"] == 1 else 1.0

    return round(base * action_multiplier * contactability_multiplier * service_multiplier, 2)


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Run 03_predictive_modeling.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    df["next_best_action"] = df.apply(assign_next_best_action, axis=1)
    df["recommended_outreach_channel"] = df.apply(assign_recommended_outreach_channel, axis=1)
    df["business_priority"] = df.apply(assign_business_priority, axis=1)
    df["action_rationale"] = df.apply(assign_action_rationale, axis=1)
    df["estimated_action_revenue"] = df.apply(estimate_action_revenue, axis=1)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Created next-best-action dataset: {OUTPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nNext-best-action distribution:")
    print(df["next_best_action"].value_counts(normalize=True).round(3))

    print("\nBusiness priority distribution:")
    print(df["business_priority"].value_counts(normalize=True).round(3))

    print("\nRecommended outreach channel distribution:")
    print(df["recommended_outreach_channel"].value_counts(normalize=True).round(3))

    print("\nAverage estimated action revenue by next-best-action:")
    print(
        df.groupby("next_best_action")["estimated_action_revenue"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )

    print("\nTotal estimated action revenue by business priority:")
    print(
        df.groupby("business_priority")["estimated_action_revenue"]
        .sum()
        .sort_values(ascending=False)
        .round(2)
    )


if __name__ == "__main__":
    main()

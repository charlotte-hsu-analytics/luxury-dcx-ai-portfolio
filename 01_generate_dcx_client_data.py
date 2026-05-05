"""
Luxury DCX Client Intelligence
Step 1: Generate Simulated DCX Client Data

Run:
    python3 01_generate_dcx_client_data.py

Output:
    data/dcx_client_data.csv

Dataset grain:
    One row = one simulated luxury client in a U.S. DCX client database,
    linked to boutique, advisor, product interest, campaign response,
    appointment conversion, consent/contactability, service experience,
    browsing intent, purchase behavior, and repeat-purchase outcome.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_CLIENTS = 5000
OUTPUT_DIR = "data"
OUTPUT_FILE = "dcx_client_data.csv"

np.random.seed(RANDOM_SEED)


def weighted_choice(options: list[str], probabilities: list[float], size: int) -> np.ndarray:
    """Return weighted random choices."""
    return np.random.choice(options, size=size, p=probabilities)


def build_boutique_reference() -> pd.DataFrame:
    """Create boutique/channel reference data."""

    boutique_data = [
        {
            "boutique_id": "B001",
            "boutique_name": "NYC Flagship",
            "boutique_city": "New York",
            "region": "Northeast",
            "boutique_format": "Flagship",
            "boutique_sales_tier": "Tier 1",
            "boutique_clienteling_maturity": "Advanced",
            "boutique_wfj_focus_flag": 1,
            "boutique_digital_adoption_score": 88,
            "selection_weight": 0.18,
        },
        {
            "boutique_id": "B002",
            "boutique_name": "SoHo Boutique",
            "boutique_city": "New York",
            "region": "Northeast",
            "boutique_format": "Fashion Boutique",
            "boutique_sales_tier": "Tier 1",
            "boutique_clienteling_maturity": "Advanced",
            "boutique_wfj_focus_flag": 0,
            "boutique_digital_adoption_score": 82,
            "selection_weight": 0.10,
        },
        {
            "boutique_id": "B003",
            "boutique_name": "Beverly Hills Boutique",
            "boutique_city": "Los Angeles",
            "region": "West",
            "boutique_format": "Flagship",
            "boutique_sales_tier": "Tier 1",
            "boutique_clienteling_maturity": "Advanced",
            "boutique_wfj_focus_flag": 1,
            "boutique_digital_adoption_score": 86,
            "selection_weight": 0.15,
        },
        {
            "boutique_id": "B004",
            "boutique_name": "Miami Design District",
            "boutique_city": "Miami",
            "region": "Southeast",
            "boutique_format": "Mixed Boutique",
            "boutique_sales_tier": "Tier 1",
            "boutique_clienteling_maturity": "Advanced",
            "boutique_wfj_focus_flag": 1,
            "boutique_digital_adoption_score": 80,
            "selection_weight": 0.11,
        },
        {
            "boutique_id": "B005",
            "boutique_name": "Chicago Boutique",
            "boutique_city": "Chicago",
            "region": "Midwest",
            "boutique_format": "Mixed Boutique",
            "boutique_sales_tier": "Tier 2",
            "boutique_clienteling_maturity": "Developing",
            "boutique_wfj_focus_flag": 0,
            "boutique_digital_adoption_score": 70,
            "selection_weight": 0.08,
        },
        {
            "boutique_id": "B006",
            "boutique_name": "Dallas Boutique",
            "boutique_city": "Dallas",
            "region": "South",
            "boutique_format": "Mixed Boutique",
            "boutique_sales_tier": "Tier 2",
            "boutique_clienteling_maturity": "Developing",
            "boutique_wfj_focus_flag": 1,
            "boutique_digital_adoption_score": 72,
            "selection_weight": 0.08,
        },
        {
            "boutique_id": "B007",
            "boutique_name": "San Francisco Boutique",
            "boutique_city": "San Francisco",
            "region": "West",
            "boutique_format": "Fashion Boutique",
            "boutique_sales_tier": "Tier 2",
            "boutique_clienteling_maturity": "Developing",
            "boutique_wfj_focus_flag": 0,
            "boutique_digital_adoption_score": 78,
            "selection_weight": 0.07,
        },
        {
            "boutique_id": "B008",
            "boutique_name": "Boston Boutique",
            "boutique_city": "Boston",
            "region": "Northeast",
            "boutique_format": "Fragrance & Beauty Boutique",
            "boutique_sales_tier": "Tier 3",
            "boutique_clienteling_maturity": "Emerging",
            "boutique_wfj_focus_flag": 0,
            "boutique_digital_adoption_score": 68,
            "selection_weight": 0.06,
        },
        {
            "boutique_id": "B009",
            "boutique_name": "Washington DC Boutique",
            "boutique_city": "Washington DC",
            "region": "Northeast",
            "boutique_format": "Mixed Boutique",
            "boutique_sales_tier": "Tier 3",
            "boutique_clienteling_maturity": "Emerging",
            "boutique_wfj_focus_flag": 0,
            "boutique_digital_adoption_score": 66,
            "selection_weight": 0.05,
        },
        {
            "boutique_id": "B010",
            "boutique_name": "Digital / E-Commerce",
            "boutique_city": "Digital",
            "region": "Digital",
            "boutique_format": "Digital",
            "boutique_sales_tier": "Digital",
            "boutique_clienteling_maturity": "Digital",
            "boutique_wfj_focus_flag": 0,
            "boutique_digital_adoption_score": 95,
            "selection_weight": 0.12,
        },
    ]

    return pd.DataFrame(boutique_data)


def assign_primary_category(primary_division: np.ndarray) -> np.ndarray:
    """Assign product category based on primary division."""

    categories = []

    for division in primary_division:
        if division == "Fashion":
            categories.append(
                np.random.choice(
                    ["Handbags", "Ready-to-Wear", "Shoes", "Small Leather Goods", "Accessories"],
                    p=[0.34, 0.24, 0.16, 0.16, 0.10],
                )
            )
        elif division == "Fragrance & Beauty":
            categories.append(
                np.random.choice(
                    ["Fragrance", "Skincare", "Makeup", "Beauty Services"],
                    p=[0.42, 0.24, 0.26, 0.08],
                )
            )
        else:
            categories.append(
                np.random.choice(
                    ["Fine Jewelry", "Watches", "High Jewelry Interest", "Bridal / Occasion Jewelry"],
                    p=[0.45, 0.25, 0.12, 0.18],
                )
            )

    return np.array(categories)


def assign_secondary_category_interest(primary_division: np.ndarray) -> np.ndarray:
    """Assign secondary interest, including cross-category signals."""

    interests = []

    for division in primary_division:
        if division == "Fashion":
            interests.append(
                np.random.choice(
                    ["Fragrance Discovery", "Fine Jewelry Interest", "Shoes", "Accessories", "None"],
                    p=[0.26, 0.18, 0.20, 0.22, 0.14],
                )
            )
        elif division == "Fragrance & Beauty":
            interests.append(
                np.random.choice(
                    ["Fashion Entry", "Handbags", "Fine Jewelry Interest", "Skincare", "None"],
                    p=[0.28, 0.18, 0.12, 0.24, 0.18],
                )
            )
        else:
            interests.append(
                np.random.choice(
                    ["Fashion Styling", "Handbags", "Fragrance Discovery", "Watches", "None"],
                    p=[0.24, 0.22, 0.14, 0.20, 0.20],
                )
            )

    return np.array(interests)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    boutiques = build_boutique_reference()

    selected_boutiques = boutiques.sample(
        n=N_CLIENTS,
        replace=True,
        weights=boutiques["selection_weight"],
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    client_ids = [f"C{str(i).zfill(6)}" for i in range(1, N_CLIENTS + 1)]

    age_group = weighted_choice(
        ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        [0.08, 0.24, 0.28, 0.20, 0.14, 0.06],
        N_CLIENTS,
    )

    client_tier = weighted_choice(
        ["New", "Core", "High Potential", "VIP", "Top Client"],
        [0.28, 0.38, 0.20, 0.11, 0.03],
        N_CLIENTS,
    )

    preferred_channel = weighted_choice(
        ["Boutique", "Online", "Omnichannel"],
        [0.42, 0.32, 0.26],
        N_CLIENTS,
    )

    digital_mask = selected_boutiques["boutique_format"].eq("Digital").to_numpy()
    preferred_channel = np.where(digital_mask, "Online", preferred_channel)

    acquisition_channel = weighted_choice(
        ["Boutique Walk-In", "Online Search", "Social Media", "Event", "Referral", "Email Campaign"],
        [0.28, 0.20, 0.16, 0.12, 0.14, 0.10],
        N_CLIENTS,
    )

    primary_division = []
    for boutique_format in selected_boutiques["boutique_format"]:
        if boutique_format == "Fragrance & Beauty Boutique":
            probs = [0.18, 0.72, 0.10]
        elif boutique_format == "Fashion Boutique":
            probs = [0.58, 0.30, 0.12]
        elif boutique_format == "Flagship":
            probs = [0.38, 0.38, 0.24]
        elif boutique_format == "Digital":
            probs = [0.26, 0.62, 0.12]
        else:
            probs = [0.34, 0.46, 0.20]

        primary_division.append(
            np.random.choice(
                ["Fashion", "Fragrance & Beauty", "Watches & Fine Jewelry"],
                p=probs,
            )
        )

    primary_division = np.array(primary_division)
    primary_category = assign_primary_category(primary_division)
    secondary_category_interest = assign_secondary_category_interest(primary_division)
    cross_category_flag = np.where(secondary_category_interest == "None", 0, 1)

    tier_spend_multiplier = {
        "New": 0.55,
        "Core": 1.00,
        "High Potential": 1.55,
        "VIP": 2.80,
        "Top Client": 5.00,
    }

    boutique_sales_multiplier = {
        "Tier 1": 1.25,
        "Tier 2": 1.00,
        "Tier 3": 0.82,
        "Digital": 0.70,
    }

    division_aov_base = {
        "Fragrance & Beauty": 260,
        "Fashion": 4200,
        "Watches & Fine Jewelry": 8500,
    }

    base_aov = np.array([division_aov_base[d] for d in primary_division])
    tier_multiplier = np.array([tier_spend_multiplier[t] for t in client_tier])
    boutique_multiplier = np.array(
        [boutique_sales_multiplier[t] for t in selected_boutiques["boutique_sales_tier"]]
    )

    average_order_value = np.random.lognormal(
        mean=np.log(base_aov * tier_multiplier * boutique_multiplier),
        sigma=0.45,
    )
    average_order_value = np.round(average_order_value, 2)

    purchase_frequency_12m = np.random.poisson(
        lam=np.select(
            [
                client_tier == "New",
                client_tier == "Core",
                client_tier == "High Potential",
                client_tier == "VIP",
                client_tier == "Top Client",
            ],
            [1.2, 2.2, 3.1, 4.8, 7.2],
            default=2.0,
        )
    )
    purchase_frequency_12m = np.maximum(purchase_frequency_12m, 0)

    transaction_count_12m = purchase_frequency_12m
    units_per_transaction = np.random.normal(
        loc=np.select(
            [
                primary_division == "Fragrance & Beauty",
                primary_division == "Fashion",
                primary_division == "Watches & Fine Jewelry",
            ],
            [2.1, 1.4, 1.1],
            default=1.5,
        ),
        scale=0.35,
    )
    units_per_transaction = np.clip(units_per_transaction, 1.0, 4.0)
    total_units_12m = np.round(transaction_count_12m * units_per_transaction).astype(int)

    total_spend_12m = np.round(average_order_value * purchase_frequency_12m, 2)

    relationship_tenure_years = np.random.poisson(
        lam=np.select(
            [
                client_tier == "New",
                client_tier == "Core",
                client_tier == "High Potential",
                client_tier == "VIP",
                client_tier == "Top Client",
            ],
            [0.4, 2.0, 3.2, 5.5, 7.5],
            default=2.0,
        )
    )
    relationship_tenure_years = np.clip(relationship_tenure_years, 0, 15)

    lifetime_spend = total_spend_12m * np.select(
        [
            client_tier == "New",
            client_tier == "Core",
            client_tier == "High Potential",
            client_tier == "VIP",
            client_tier == "Top Client",
        ],
        [1.1, 2.5, 3.5, 5.5, 8.0],
        default=2.5,
    )
    lifetime_spend = lifetime_spend + np.random.normal(
        loc=0,
        scale=np.maximum(total_spend_12m * 0.20, 100),
        size=N_CLIENTS,
    )
    lifetime_spend = np.maximum(lifetime_spend, total_spend_12m)
    lifetime_spend = np.round(lifetime_spend, 2)

    high_value_client_flag = np.where(
        (client_tier == "VIP")
        | (client_tier == "Top Client")
        | (lifetime_spend >= 50000),
        1,
        0,
    )

    strategic_division_opportunity = np.where(
        primary_division == "Watches & Fine Jewelry",
        "High Strategic Value",
        np.where(primary_division == "Fashion", "Medium Strategic Value", "Scale Opportunity"),
    )

    client_value_score = (
        0.35 * np.log1p(lifetime_spend)
        + 0.25 * np.log1p(total_spend_12m)
        + 0.20 * relationship_tenure_years
        + 0.20 * high_value_client_flag * 10
        + 0.50 * cross_category_flag
    )
    client_value_score = np.round(client_value_score, 2)

    last_purchase_days = np.random.gamma(
        shape=2.2,
        scale=np.select(
            [
                client_tier == "New",
                client_tier == "Core",
                client_tier == "High Potential",
                client_tier == "VIP",
                client_tier == "Top Client",
            ],
            [65, 55, 43, 35, 28],
            default=55,
        ),
    )
    last_purchase_days = np.clip(last_purchase_days, 1, 365).round().astype(int)

    categories_purchased = np.random.poisson(
        lam=np.select(
            [
                client_tier == "New",
                client_tier == "Core",
                client_tier == "High Potential",
                client_tier == "VIP",
                client_tier == "Top Client",
            ],
            [1.0, 1.5, 2.1, 3.0, 4.2],
            default=1.5,
        )
    )
    categories_purchased = np.clip(categories_purchased, 1, 6)

    boutique_visits_12m = np.random.poisson(
        lam=np.select(
            [
                preferred_channel == "Online",
                preferred_channel == "Boutique",
                preferred_channel == "Omnichannel",
            ],
            [0.5, 3.0, 4.2],
            default=2.0,
        )
        * tier_multiplier
    )
    boutique_visits_12m = np.where(digital_mask, np.minimum(boutique_visits_12m, 2), boutique_visits_12m)
    boutique_visits_12m = np.clip(boutique_visits_12m, 0, 30)

    online_visits_90d = np.random.poisson(
        lam=np.select(
            [
                preferred_channel == "Boutique",
                preferred_channel == "Online",
                preferred_channel == "Omnichannel",
            ],
            [3.0, 9.0, 12.0],
            default=6.0,
        )
    )
    online_visits_90d = np.where(
        digital_mask,
        online_visits_90d + np.random.poisson(5, N_CLIENTS),
        online_visits_90d,
    )
    online_visits_90d = np.clip(online_visits_90d, 0, 80)

    # Category browsing intent
    product_page_views_90d = online_visits_90d * np.random.randint(2, 8, N_CLIENTS)
    fashion_page_views_90d = np.random.poisson(
        lam=np.where(primary_division == "Fashion", 10, 3) + np.where(secondary_category_interest == "Fashion Entry", 5, 0),
        size=N_CLIENTS,
    )
    beauty_page_views_90d = np.random.poisson(
        lam=np.where(primary_division == "Fragrance & Beauty", 12, 4) + np.where(secondary_category_interest == "Fragrance Discovery", 5, 0),
        size=N_CLIENTS,
    )
    wfj_page_views_90d = np.random.poisson(
        lam=np.where(primary_division == "Watches & Fine Jewelry", 8, 2) + np.where(secondary_category_interest == "Fine Jewelry Interest", 6, 0),
        size=N_CLIENTS,
    )
    wishlist_additions_90d = np.random.poisson(
        lam=np.select(
            [
                preferred_channel == "Boutique",
                preferred_channel == "Online",
                preferred_channel == "Omnichannel",
            ],
            [0.4, 1.6, 2.2],
            default=1.0,
        )
    )
    wishlist_additions_90d = np.clip(wishlist_additions_90d, 0, 15)

    high_intent_browsing_flag = np.where(
        (wishlist_additions_90d >= 2)
        | (wfj_page_views_90d >= 8)
        | (product_page_views_90d >= 60),
        1,
        0,
    )

    appointment_count_12m = np.random.poisson(
        lam=np.select(
            [
                client_tier == "New",
                client_tier == "Core",
                client_tier == "High Potential",
                client_tier == "VIP",
                client_tier == "Top Client",
            ],
            [0.2, 0.6, 1.2, 2.4, 4.0],
            default=0.6,
        )
    )
    appointment_count_12m = np.clip(appointment_count_12m, 0, 12)

    appointment_scheduled_flag = np.random.binomial(
        1,
        p=np.clip(0.08 + 0.04 * appointment_count_12m + 0.10 * high_intent_browsing_flag, 0, 0.85),
    )
    appointment_completed_flag = np.random.binomial(
        1,
        p=np.where(appointment_scheduled_flag == 1, 0.72, 0.03),
    )
    appointment_conversion_flag = np.random.binomial(
        1,
        p=np.where(appointment_completed_flag == 1, 0.34 + 0.12 * high_value_client_flag, 0.02),
    )
    days_since_last_appointment = np.where(
        appointment_scheduled_flag == 1,
        np.random.randint(1, 180, N_CLIENTS),
        999,
    )

    clienteling_touchpoints_90d = np.random.poisson(
        lam=np.select(
            [
                client_tier == "New",
                client_tier == "Core",
                client_tier == "High Potential",
                client_tier == "VIP",
                client_tier == "Top Client",
            ],
            [0.5, 1.2, 2.2, 4.0, 6.5],
            default=1.0,
        )
    )
    maturity_boost = selected_boutiques["boutique_clienteling_maturity"].map(
        {"Advanced": 1.25, "Developing": 1.00, "Emerging": 0.80, "Digital": 0.75}
    ).to_numpy()
    clienteling_touchpoints_90d = np.round(clienteling_touchpoints_90d * maturity_boost).astype(int)
    clienteling_touchpoints_90d = np.clip(clienteling_touchpoints_90d, 0, 20)

    email_open_rate = np.random.beta(
        a=np.select(
            [
                client_tier == "New",
                client_tier == "Core",
                client_tier == "High Potential",
                client_tier == "VIP",
                client_tier == "Top Client",
            ],
            [2.0, 2.4, 3.0, 3.8, 4.2],
            default=2.4,
        ),
        b=4.2,
    )
    email_open_rate = np.round(email_open_rate, 3)

    email_opt_in_flag = np.random.binomial(
        1,
        p=np.select(
            [
                client_tier == "New",
                client_tier == "Core",
                client_tier == "High Potential",
                client_tier == "VIP",
                client_tier == "Top Client",
            ],
            [0.68, 0.74, 0.80, 0.86, 0.90],
            default=0.75,
        ),
    )

    sms_opt_in_flag = np.random.binomial(
        1,
        p=np.select(
            [
                age_group == "18-24",
                age_group == "25-34",
                age_group == "35-44",
                age_group == "45-54",
                age_group == "55-64",
                age_group == "65+",
            ],
            [0.58, 0.55, 0.48, 0.40, 0.34, 0.28],
            default=0.45,
        ),
    )

    client_contactable_flag = np.where((email_opt_in_flag == 1) | (sms_opt_in_flag == 1), 1, 0)

    preferred_contact_method = np.select(
        [
            (email_opt_in_flag == 1) & (sms_opt_in_flag == 1),
            (email_opt_in_flag == 1) & (sms_opt_in_flag == 0),
            (email_opt_in_flag == 0) & (sms_opt_in_flag == 1),
        ],
        ["Email + SMS", "Email", "SMS"],
        default="Do Not Contact",
    )

    app_engagement_score = np.random.beta(
        a=np.where(preferred_channel == "Online", 3.5, np.where(preferred_channel == "Omnichannel", 4.0, 2.0)),
        b=4.0,
    )
    app_engagement_score = np.round(app_engagement_score * 100, 1)

    digital_campaign_click_rate = np.random.beta(
        a=2.0 + (email_open_rate * 3.0),
        b=8.0,
    )
    digital_campaign_click_rate = np.round(digital_campaign_click_rate, 3)

    # Campaign history
    last_campaign_type = weighted_choice(
        [
            "Fragrance Launch",
            "Fashion Preview",
            "WFJ Appointment Invite",
            "Beauty Replenishment",
            "Private Event",
            "Welcome Journey",
            "Reactivation",
            "No Recent Campaign",
        ],
        [0.18, 0.14, 0.10, 0.16, 0.10, 0.12, 0.12, 0.08],
        N_CLIENTS,
    )

    campaign_sent_30d = np.where(last_campaign_type == "No Recent Campaign", 0, 1)
    campaign_open_30d = np.random.binomial(
        1,
        p=np.where(campaign_sent_30d == 1, np.clip(email_open_rate + 0.10 * high_value_client_flag, 0, 0.95), 0),
    )
    campaign_click_30d = np.random.binomial(
        1,
        p=np.where(campaign_open_30d == 1, np.clip(digital_campaign_click_rate + 0.08 * high_intent_browsing_flag, 0, 0.80), 0),
    )
    campaign_conversion_30d = np.random.binomial(
        1,
        p=np.where(campaign_click_30d == 1, 0.18 + 0.10 * appointment_completed_flag + 0.08 * high_value_client_flag, 0.03),
    )
    days_since_last_campaign = np.where(
        campaign_sent_30d == 1,
        np.random.randint(1, 31, N_CLIENTS),
        999,
    )

    event_attendance_12m = np.random.poisson(
        lam=np.select(
            [
                client_tier == "New",
                client_tier == "Core",
                client_tier == "High Potential",
                client_tier == "VIP",
                client_tier == "Top Client",
            ],
            [0.05, 0.15, 0.35, 0.90, 1.80],
            default=0.20,
        )
    )
    event_attendance_12m = np.clip(event_attendance_12m, 0, 8)

    advisor_follow_up_flag = np.random.binomial(
        1,
        p=np.select(
            [
                client_tier == "New",
                client_tier == "Core",
                client_tier == "High Potential",
                client_tier == "VIP",
                client_tier == "Top Client",
            ],
            [0.18, 0.28, 0.45, 0.68, 0.82],
            default=0.30,
        ),
    )

    advisor_id = []
    advisor_tenure_years = []
    advisor_client_book_size = []
    advisor_productivity_tier = []

    productivity_options = ["Emerging", "Core", "High Performer", "Top Performer"]
    productivity_probs = [0.18, 0.45, 0.27, 0.10]

    for boutique_id in selected_boutiques["boutique_id"]:
        if boutique_id == "B010":
            advisor_id.append("DIGITAL_POOL")
            advisor_tenure_years.append(0)
            advisor_client_book_size.append(0)
            advisor_productivity_tier.append("Digital")
        else:
            advisor_num = np.random.randint(1, 16)
            advisor_id.append(f"{boutique_id}_A{advisor_num:02d}")
            advisor_tenure_years.append(int(np.clip(np.random.poisson(3.5), 0, 18)))
            advisor_client_book_size.append(int(np.clip(np.random.normal(180, 55), 50, 400)))
            advisor_productivity_tier.append(np.random.choice(productivity_options, p=productivity_probs))

    advisor_tenure_years = np.array(advisor_tenure_years)
    advisor_client_book_size = np.array(advisor_client_book_size)
    advisor_productivity_tier = np.array(advisor_productivity_tier)

    # Return/service behavior
    service_interaction_12m = np.random.poisson(
        lam=np.where(high_value_client_flag == 1, 1.6, 0.6) + np.where(primary_division == "Fashion", 0.3, 0),
        size=N_CLIENTS,
    )
    service_interaction_12m = np.clip(service_interaction_12m, 0, 8)

    return_or_exchange_12m = np.random.binomial(
        n=np.maximum(transaction_count_12m, 1),
        p=np.select(
            [
                primary_division == "Fragrance & Beauty",
                primary_division == "Fashion",
                primary_division == "Watches & Fine Jewelry",
            ],
            [0.04, 0.10, 0.03],
            default=0.06,
        ),
    )
    return_or_exchange_12m = np.clip(return_or_exchange_12m, 0, 8)

    client_satisfaction_score = np.random.normal(
        loc=82 + 5 * high_value_client_flag + 3 * appointment_completed_flag - 8 * (return_or_exchange_12m > 0),
        scale=9,
        size=N_CLIENTS,
    )
    client_satisfaction_score = np.clip(client_satisfaction_score, 40, 100).round(1)

    service_recovery_flag = np.where(
        (client_satisfaction_score < 70) | ((return_or_exchange_12m > 0) & (high_value_client_flag == 1)),
        1,
        0,
    )

    estimated_return_rate = np.where(
        np.maximum(transaction_count_12m, 1) > 0,
        return_or_exchange_12m / np.maximum(transaction_count_12m, 1),
        0,
    )
    estimated_return_rate = np.round(np.clip(estimated_return_rate, 0, 1), 3)

    atv = np.where(transaction_count_12m > 0, total_spend_12m / np.maximum(transaction_count_12m, 1), 0)
    upt = np.where(transaction_count_12m > 0, total_units_12m / np.maximum(transaction_count_12m, 1), 0)

    # Hidden business reality used to generate repeat-purchase target.
    log_spend = np.log1p(total_spend_12m)

    score = (
        -2.55
        - 0.010 * last_purchase_days
        + 0.230 * purchase_frequency_12m
        + 0.090 * categories_purchased
        + 0.065 * boutique_visits_12m
        + 0.025 * online_visits_90d
        + 0.150 * appointment_count_12m
        + 0.200 * appointment_conversion_flag
        + 0.130 * clienteling_touchpoints_90d
        + 0.900 * email_open_rate
        + 0.170 * event_attendance_12m
        + 0.260 * advisor_follow_up_flag
        + 0.100 * campaign_open_30d
        + 0.170 * campaign_click_30d
        + 0.260 * campaign_conversion_30d
        + 0.080 * wishlist_additions_90d
        + 0.006 * app_engagement_score
        + 0.100 * high_intent_browsing_flag
        + 0.080 * client_contactable_flag
        - 0.180 * service_recovery_flag
        - 0.120 * return_or_exchange_12m
        + 0.006 * client_satisfaction_score
        + 0.085 * log_spend
        + 0.025 * relationship_tenure_years
        + 0.016 * client_value_score
        + 0.160 * cross_category_flag
        + np.where(client_tier == "VIP", 0.25, 0)
        + np.where(client_tier == "Top Client", 0.50, 0)
        + np.where(preferred_channel == "Omnichannel", 0.20, 0)
        + np.where(selected_boutiques["boutique_clienteling_maturity"].to_numpy() == "Advanced", 0.14, 0)
        + np.where(advisor_productivity_tier == "Top Performer", 0.16, 0)
        + np.where(advisor_productivity_tier == "High Performer", 0.09, 0)
    )

    repeat_purchase_probability_true = 1 / (1 + np.exp(-score))
    repeat_purchase_90d = np.random.binomial(1, repeat_purchase_probability_true)

    estimated_clienteling_revenue = (
        repeat_purchase_probability_true
        * average_order_value
        * np.where(client_contactable_flag == 1, 1, 0.25)
        * np.where(service_recovery_flag == 1, 0.65, 1.0)
    )
    estimated_clienteling_revenue = np.round(estimated_clienteling_revenue, 2)

    df = pd.DataFrame(
        {
            "client_id": client_ids,
            "age_group": age_group,
            "client_tier": client_tier,
            "preferred_channel": preferred_channel,
            "acquisition_channel": acquisition_channel,

            "boutique_id": selected_boutiques["boutique_id"],
            "boutique_name": selected_boutiques["boutique_name"],
            "boutique_city": selected_boutiques["boutique_city"],
            "region": selected_boutiques["region"],
            "boutique_format": selected_boutiques["boutique_format"],
            "boutique_sales_tier": selected_boutiques["boutique_sales_tier"],
            "boutique_clienteling_maturity": selected_boutiques["boutique_clienteling_maturity"],
            "boutique_wfj_focus_flag": selected_boutiques["boutique_wfj_focus_flag"],
            "boutique_digital_adoption_score": selected_boutiques["boutique_digital_adoption_score"],

            "advisor_id": advisor_id,
            "advisor_tenure_years": advisor_tenure_years,
            "advisor_client_book_size": advisor_client_book_size,
            "advisor_productivity_tier": advisor_productivity_tier,

            "primary_division": primary_division,
            "primary_category": primary_category,
            "secondary_category_interest": secondary_category_interest,
            "cross_category_flag": cross_category_flag,
            "strategic_division_opportunity": strategic_division_opportunity,

            "last_purchase_days": last_purchase_days,
            "purchase_frequency_12m": purchase_frequency_12m,
            "transaction_count_12m": transaction_count_12m,
            "total_units_12m": total_units_12m,
            "atv": np.round(atv, 2),
            "upt": np.round(upt, 2),
            "total_spend_12m": total_spend_12m,
            "lifetime_spend": lifetime_spend,
            "average_order_value": average_order_value,
            "categories_purchased": categories_purchased,
            "relationship_tenure_years": relationship_tenure_years,
            "high_value_client_flag": high_value_client_flag,
            "client_value_score": client_value_score,

            "online_visits_90d": online_visits_90d,
            "product_page_views_90d": product_page_views_90d,
            "fashion_page_views_90d": fashion_page_views_90d,
            "beauty_page_views_90d": beauty_page_views_90d,
            "wfj_page_views_90d": wfj_page_views_90d,
            "wishlist_additions_90d": wishlist_additions_90d,
            "high_intent_browsing_flag": high_intent_browsing_flag,
            "app_engagement_score": app_engagement_score,

            "last_campaign_type": last_campaign_type,
            "campaign_sent_30d": campaign_sent_30d,
            "campaign_open_30d": campaign_open_30d,
            "campaign_click_30d": campaign_click_30d,
            "campaign_conversion_30d": campaign_conversion_30d,
            "days_since_last_campaign": days_since_last_campaign,
            "digital_campaign_click_rate": digital_campaign_click_rate,

            "appointment_scheduled_flag": appointment_scheduled_flag,
            "appointment_completed_flag": appointment_completed_flag,
            "appointment_conversion_flag": appointment_conversion_flag,
            "days_since_last_appointment": days_since_last_appointment,
            "appointment_count_12m": appointment_count_12m,
            "boutique_visits_12m": boutique_visits_12m,

            "email_open_rate": email_open_rate,
            "email_opt_in_flag": email_opt_in_flag,
            "sms_opt_in_flag": sms_opt_in_flag,
            "client_contactable_flag": client_contactable_flag,
            "preferred_contact_method": preferred_contact_method,

            "service_interaction_12m": service_interaction_12m,
            "return_or_exchange_12m": return_or_exchange_12m,
            "estimated_return_rate": estimated_return_rate,
            "client_satisfaction_score": client_satisfaction_score,
            "service_recovery_flag": service_recovery_flag,

            "clienteling_touchpoints_90d": clienteling_touchpoints_90d,
            "event_attendance_12m": event_attendance_12m,
            "advisor_follow_up_flag": advisor_follow_up_flag,

            "repeat_purchase_probability_true": np.round(repeat_purchase_probability_true, 4),
            "repeat_purchase_90d": repeat_purchase_90d,
            "estimated_clienteling_revenue": estimated_clienteling_revenue,
        }
    )

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(output_path, index=False)

    print(f"Created dataset: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nRepeat purchase rate:")
    print(round(df["repeat_purchase_90d"].mean(), 3))

    print("\nBoutique distribution:")
    print(df["boutique_name"].value_counts(normalize=True).round(3))

    print("\nPrimary division distribution:")
    print(df["primary_division"].value_counts(normalize=True).round(3))

    print("\nClient tier distribution:")
    print(df["client_tier"].value_counts(normalize=True).round(3))

    print("\nContactable client rate:")
    print(round(df["client_contactable_flag"].mean(), 3))

    print("\nHigh-value client rate:")
    print(round(df["high_value_client_flag"].mean(), 3))

    print("\nAverage estimated clienteling revenue:")
    print(round(df["estimated_clienteling_revenue"].mean(), 2))


if __name__ == "__main__":
    main()

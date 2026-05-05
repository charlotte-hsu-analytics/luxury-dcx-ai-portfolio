"""
Luxury DCX Client Intelligence
Step 3: Predictive Modeling

Run:
    python3 03_predictive_modeling.py

Input:
    data/dcx_client_segmented.csv

Outputs:
    data/dcx_client_model_scored.csv
    data/model_feature_importance.csv

Purpose:
    Train models to predict repeat_purchase_90d and create business-priority groups.
"""

from __future__ import annotations

import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_FILE = "data/dcx_client_segmented.csv"
SCORED_OUTPUT_FILE = "data/dcx_client_model_scored.csv"
FEATURE_IMPORTANCE_FILE = "data/model_feature_importance.csv"

TARGET = "repeat_purchase_90d"
RANDOM_SEED = 42


def load_data() -> pd.DataFrame:
    """Load segmented DCX client dataset."""

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Run 02_client_segmentation.py first."
        )

    return pd.read_csv(INPUT_FILE)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Prepare model features and target."""

    numeric_features = [
        # Purchase behavior
        "last_purchase_days",
        "purchase_frequency_12m",
        "transaction_count_12m",
        "total_units_12m",
        "atv",
        "upt",
        "total_spend_12m",
        "lifetime_spend",
        "average_order_value",
        "categories_purchased",
        "relationship_tenure_years",
        "high_value_client_flag",
        "client_value_score",

        # Digital browsing intent
        "online_visits_90d",
        "product_page_views_90d",
        "fashion_page_views_90d",
        "beauty_page_views_90d",
        "wfj_page_views_90d",
        "wishlist_additions_90d",
        "high_intent_browsing_flag",
        "app_engagement_score",

        # Campaign behavior
        "campaign_sent_30d",
        "campaign_open_30d",
        "campaign_click_30d",
        "campaign_conversion_30d",
        "days_since_last_campaign",
        "digital_campaign_click_rate",

        # Appointment / boutique conversion
        "appointment_scheduled_flag",
        "appointment_completed_flag",
        "appointment_conversion_flag",
        "days_since_last_appointment",
        "appointment_count_12m",
        "boutique_visits_12m",

        # Consent / contactability
        "email_open_rate",
        "email_opt_in_flag",
        "sms_opt_in_flag",
        "client_contactable_flag",

        # Return / service behavior
        "service_interaction_12m",
        "return_or_exchange_12m",
        "estimated_return_rate",
        "client_satisfaction_score",
        "service_recovery_flag",

        # Advisor / clienteling
        "advisor_tenure_years",
        "advisor_client_book_size",
        "clienteling_touchpoints_90d",
        "event_attendance_12m",
        "advisor_follow_up_flag",

        # Boutique context
        "boutique_wfj_focus_flag",
        "boutique_digital_adoption_score",

        # Engineered scores
        "cross_category_flag",
        "recency_score",
        "frequency_score",
        "monetary_score",
        "rfm_score",
        "engagement_score",
        "digital_intent_score",
        "campaign_engagement_score",
        "service_risk_score",
    ]

    categorical_features = [
        # Client profile
        "age_group",
        "client_tier",
        "preferred_channel",
        "acquisition_channel",
        "preferred_contact_method",

        # Boutique / advisor
        "boutique_name",
        "boutique_city",
        "region",
        "boutique_format",
        "boutique_sales_tier",
        "boutique_clienteling_maturity",
        "advisor_productivity_tier",

        # Product / category
        "primary_division",
        "primary_category",
        "secondary_category_interest",
        "strategic_division_opportunity",

        # Campaign / segment
        "last_campaign_type",
        "luxury_value_band",
        "client_segment",
    ]

    feature_columns = numeric_features + categorical_features

    X = df[feature_columns].copy()
    y = df[TARGET].copy()

    return X, y, numeric_features, categorical_features


def create_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create preprocessing pipeline."""

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def get_feature_names(
    model_pipeline: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    """Get readable feature names after preprocessing."""

    onehot = (
        model_pipeline.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["onehot"]
    )

    categorical_feature_names = list(
        onehot.get_feature_names_out(categorical_features)
    )

    return numeric_features + categorical_feature_names


def assign_model_priority(row: pd.Series) -> str:
    """
    Assign business-friendly priority groups.

    The model predicts repeat-purchase probability.
    Business priority combines model score with:
    - client value
    - VIP status
    - WFJ opportunity
    - contactability
    - service risk
    - appointment/campaign conversion
    """

    probability = row["predicted_repeat_purchase_probability"]
    client_tier = row["client_tier"]
    lifetime_spend = row["lifetime_spend"]
    high_value_flag = row["high_value_client_flag"]
    segment = row["client_segment"]
    primary_division = row["primary_division"]
    contactable = row["client_contactable_flag"]
    service_risk = row["service_risk_score"]
    strategic_opportunity = row["strategic_division_opportunity"]
    luxury_value_band = row["luxury_value_band"]
    appointment_conversion = row["appointment_conversion_flag"]
    campaign_conversion = row["campaign_conversion_30d"]

    # High-value service recovery should be handled urgently, even if probability is lower.
    if (
        segment == "VIP Service Recovery"
        and lifetime_spend >= 50000
    ):
        return "Top Priority"

    # True luxury high-touch priority.
    if (
        client_tier in ["VIP", "Top Client"]
        and high_value_flag == 1
        and contactable == 1
        and probability >= 0.70
    ):
        return "Top Priority"

    # Ultra high value clients with strong predicted likelihood.
    if (
        luxury_value_band == "Ultra High Value"
        and contactable == 1
        and probability >= 0.68
    ):
        return "Top Priority"

    # At-risk VIPs should be prioritized even with moderate probability.
    if (
        segment == "At-Risk VIP"
        and lifetime_spend >= 40000
        and service_risk < 70
    ):
        return "Top Priority"

    # Strong opportunity groups.
    if (
        client_tier in ["High Potential", "VIP", "Top Client"]
        and contactable == 1
        and probability >= 0.60
    ):
        return "High Priority"

    # WFJ opportunity should be elevated due to strategic value.
    if (
        primary_division == "Watches & Fine Jewelry"
        and strategic_opportunity == "High Strategic Value"
        and lifetime_spend >= 25000
        and probability >= 0.55
    ):
        return "High Priority"

    # Recently converted clients are valuable but usually move into nurture, not immediate sales.
    if (
        appointment_conversion == 1
        or campaign_conversion == 1
    ) and probability >= 0.50:
        return "High Priority"

    # Non-contactable clients should usually not be high-touch outbound priority.
    if contactable == 0:
        if probability >= 0.65 and lifetime_spend >= 50000:
            return "Medium Priority"
        return "Low Priority"

    # Service risk lowers sales-priority unless it is VIP recovery.
    if service_risk >= 60:
        return "Medium Priority"

    if probability >= 0.40:
        return "Medium Priority"

    return "Low Priority"


def main() -> None:
    df = load_data()

    X, y, numeric_features, categorical_features = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    logistic_model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    random_forest_model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=350,
                    max_depth=9,
                    min_samples_leaf=20,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
    }

    results = {}

    print("Model Evaluation")
    print("=" * 70)

    for model_name, model_pipeline in models.items():
        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)
        y_prob = model_pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)

        results[model_name] = {
            "pipeline": model_pipeline,
            "auc": auc,
            "accuracy": accuracy,
        }

        print(f"\n{model_name}")
        print("-" * 70)
        print(f"AUC: {auc:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    best_model_name = max(results, key=lambda name: results[name]["auc"])
    best_model = results[best_model_name]["pipeline"]

    print("\nBest Model")
    print("=" * 70)
    print(f"{best_model_name} selected based on highest AUC.")

    # Score the full dataset.
    df["predicted_repeat_purchase_probability"] = best_model.predict_proba(X)[:, 1]
    df["predicted_repeat_purchase_probability"] = (
        df["predicted_repeat_purchase_probability"].round(4)
    )

    # Business priority is intentionally separate from model score.
    df["model_priority_group"] = df.apply(assign_model_priority, axis=1)

    df.to_csv(SCORED_OUTPUT_FILE, index=False)

    print(f"\nCreated scored dataset: {SCORED_OUTPUT_FILE}")

    # Use Random Forest for business-friendly feature importance.
    rf_model = results["Random Forest"]["pipeline"]
    feature_names = get_feature_names(rf_model, numeric_features, categorical_features)
    importances = rf_model.named_steps["model"].feature_importances_

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    feature_importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    print(f"Created feature importance file: {FEATURE_IMPORTANCE_FILE}")

    print("\nTop 25 model drivers:")
    print(feature_importance.head(25).to_string(index=False))

    print("\nPriority group distribution:")
    print(df["model_priority_group"].value_counts(normalize=True).round(3))

    print("\nPriority group count:")
    print(df["model_priority_group"].value_counts())

    print("\nAverage lifetime spend by priority group:")
    print(
        df.groupby("model_priority_group")["lifetime_spend"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )

    print("\nAverage predicted repeat probability by priority group:")
    print(
        df.groupby("model_priority_group")["predicted_repeat_purchase_probability"]
        .mean()
        .sort_values(ascending=False)
        .round(3)
    )

    print("\nPriority group by client tier:")
    print(
        pd.crosstab(
            df["client_tier"],
            df["model_priority_group"],
            normalize="index",
        ).round(3)
    )


if __name__ == "__main__":
    main()

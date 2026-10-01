"""
Titanic Data Cleaning and Feature Engineering Module
Week 2: Clean Data and Engineer Features

Provides reusable, tested functions to clean raw Titanic passenger data,
impute missing values with justifiable domain strategies, engineer new predictive
features (family_size, is_alone, has_cabin, fare_per_person, age_group),
and encode categorical features into model-ready numeric format.
"""

import os
import argparse
import pandas as pd
import numpy as np


def clean_data(
    df: pd.DataFrame,
    drop_first: bool = True,
    encode_categoricals: bool = True,
    encode_age_group: bool = False,
    drop_unmodeled_text: bool = True
) -> pd.DataFrame:
    """
    Cleans raw Titanic DataFrame and engineers informative features.
    Guarantees raw data remains unmodified by working on a deep copy.

    Cleaning Steps:
    1. Impute missing 'Embarked' using the modal port ('S' - Southampton).
    2. Impute missing 'Age' using the median age conditioned on 'Pclass' and 'Sex'.
    3. Impute missing 'Fare' (if any) using the median fare conditioned on 'Pclass'.
    4. Feature Engineering:
       - 'family_size': SibSp + Parch + 1 (total travel party size).
       - 'is_alone': 1 if family_size == 1, else 0 (binary solo travel indicator).
       - 'has_cabin': 1 if Cabin was recorded, else 0.
       - 'fare_per_person': Fare / family_size (effective ticket price per passenger).
       - 'age_group': Binned age categories ('child', 'teen', 'adult', 'senior').
       - 'age_group_code': Numeric code (0, 1, 2, 3) representing age category.
    5. Categorical Encoding:
       - 'Sex' and 'Embarked' encoded via pd.get_dummies() with dtype=int.
       - If encode_age_group=True, one-hot encodes 'age_group'.
    6. Drops raw high-cardinality/unmodeled text columns ('Cabin', 'Ticket', 'Name').

    Parameters
    ----------
    df : pd.DataFrame
        Raw Titanic dataset containing standard Kaggle columns.
    drop_first : bool, default True
        Whether to drop the first dummy category to avoid the dummy variable trap.
    encode_categoricals : bool, default True
        Whether to convert 'Sex' and 'Embarked' to numeric dummy columns.
    encode_age_group : bool, default False
        Whether to one-hot encode the 'age_group' column.
    drop_unmodeled_text : bool, default True
        Whether to drop 'Cabin', 'Ticket', and 'Name'.

    Returns
    -------
    pd.DataFrame
        The cleaned, transformed, and feature-engineered DataFrame.
    """
    # 0. Always work on a fresh copy to guarantee raw data immutability
    df_clean = df.copy()

    # 1. Impute Embarked with the mode ('S')
    # Southampton accounts for ~72% of all boardings; imputing 2 missing values preserves distribution.
    if 'Embarked' in df_clean.columns:
        embarked_mode = df_clean['Embarked'].mode()[0] if not df_clean['Embarked'].dropna().empty else 'S'
        df_clean['Embarked'] = df_clean['Embarked'].fillna(embarked_mode)

    # 2. Impute Age with median grouped by Pclass and Sex
    # Age is right-skewed; median avoids outlier distortion, and grouping by Class & Gender captures strong demographics.
    if 'Age' in df_clean.columns:
        if 'Pclass' in df_clean.columns and 'Sex' in df_clean.columns:
            median_ages = df_clean.groupby(['Pclass', 'Sex'])['Age'].transform('median')
            df_clean['Age'] = df_clean['Age'].fillna(median_ages)
        # Fallback to overall median if any edge-case NaN remains
        overall_median_age = df_clean['Age'].median() if not df_clean['Age'].dropna().empty else 28.0
        df_clean['Age'] = df_clean['Age'].fillna(overall_median_age)

    # 3. Impute Fare (if missing, e.g. in test set passenger 1044) with median of Pclass
    if 'Fare' in df_clean.columns:
        if 'Pclass' in df_clean.columns:
            median_fares = df_clean.groupby('Pclass')['Fare'].transform('median')
            df_clean['Fare'] = df_clean['Fare'].fillna(median_fares)
        overall_median_fare = df_clean['Fare'].median() if not df_clean['Fare'].dropna().empty else 14.45
        df_clean['Fare'] = df_clean['Fare'].fillna(overall_median_fare)

    # 4. Feature Engineering
    # family_size and is_alone
    if 'SibSp' in df_clean.columns and 'Parch' in df_clean.columns:
        df_clean['family_size'] = df_clean['SibSp'] + df_clean['Parch'] + 1
        df_clean['is_alone'] = (df_clean['family_size'] == 1).astype(int)

    # has_cabin: extract binary survival signal before dropping raw Cabin string
    if 'Cabin' in df_clean.columns:
        df_clean['has_cabin'] = df_clean['Cabin'].notnull().astype(int)
    elif 'has_cabin' not in df_clean.columns:
        df_clean['has_cabin'] = 0

    # fare_per_person
    if 'Fare' in df_clean.columns and 'family_size' in df_clean.columns:
        df_clean['fare_per_person'] = (df_clean['Fare'] / df_clean['family_size']).round(2)

    # Stretch goal: age_group bucketing using pd.cut
    if 'Age' in df_clean.columns:
        age_bins = [0, 12, 18, 60, 120]
        age_labels = ['child', 'teen', 'adult', 'senior']
        df_clean['age_group'] = pd.cut(
            df_clean['Age'],
            bins=age_bins,
            labels=age_labels,
            right=False
        )
        df_clean['age_group_code'] = df_clean['age_group'].cat.codes.astype(int)

    # 5. Categorical Encoding
    if encode_categoricals:
        cat_cols = [c for c in ['Sex', 'Embarked'] if c in df_clean.columns]
        if encode_age_group and 'age_group' in df_clean.columns:
            cat_cols.append('age_group')
        if cat_cols:
            df_clean = pd.get_dummies(
                df_clean,
                columns=cat_cols,
                drop_first=drop_first,
                dtype=int
            )

    # 6. Drop unmodeled high-cardinality/messy raw columns
    if drop_unmodeled_text:
        cols_to_drop = ['Cabin', 'Ticket', 'Name']
        df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])

    return df_clean


def main():
    parser = argparse.ArgumentParser(description="Clean Titanic dataset and engineer features.")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(base_dir, 'data', 'titanic_raw.csv')
    default_output = os.path.join(base_dir, 'data', 'titanic_clean.csv')

    parser.add_argument('--input', default=default_input, help="Path to input raw CSV.")
    parser.add_argument('--output', default=default_output, help="Path to save cleaned CSV.")
    args = parser.parse_args()

    print(f"Loading raw data from: {args.input}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Raw data file not found at {args.input}")

    df_raw = pd.read_csv(args.input)
    print(f"Raw dataset shape: {df_raw.shape}")
    print("Raw missing values summary:")
    print(df_raw.isnull().sum()[df_raw.isnull().sum() > 0])

    print("\nExecuting clean_data pipeline...")
    df_cleaned = clean_data(df_raw)

    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    print("Cleaned missing values check:")
    nulls = df_cleaned.isnull().sum()
    print(nulls)
    assert nulls.sum() == 0, f"Error: Remaining null values found: {nulls[nulls > 0]}"

    # Verify features
    assert 'family_size' in df_cleaned.columns, "Missing family_size"
    assert 'is_alone' in df_cleaned.columns, "Missing is_alone"
    assert df_cleaned['family_size'].min() >= 1, "family_size cannot be less than 1"
    assert set(df_cleaned['is_alone'].unique()).issubset({0, 1}), "is_alone must be 0 or 1"

    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_cleaned.to_csv(args.output, index=False)
    print(f"\n[OK] Cleaned dataset successfully saved to: {args.output}")

    # Verify reload
    reloaded = pd.read_csv(args.output)
    print(f"[OK] Verified reload: shape={reloaded.shape}, columns={len(reloaded.columns)}")


if __name__ == '__main__':
    main()

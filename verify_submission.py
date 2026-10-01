"""
Automated Verification Suite for Week 2: Clean Data & Engineer Features
Validates all acceptance criteria in the 'Done When' checklist.
"""

import os
import unittest
import pandas as pd
import numpy as np
from clean_data import clean_data


class TestTitanicWeek2Submission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.raw_path = os.path.join(cls.base_dir, 'data', 'titanic_raw.csv')
        cls.clean_path = os.path.join(cls.base_dir, 'data', 'titanic_clean.csv')

        assert os.path.exists(cls.raw_path), f"Raw data not found at {cls.raw_path}"
        cls.raw_df = pd.read_csv(cls.raw_path)

        # Generate fresh clean DataFrame
        cls.cleaned_df = clean_data(cls.raw_df)

    def test_criterion_1_zero_missing_values(self):
        """
        Done When Criterion 1:
        clean_data(df).isnull().sum() shows zero missing values in the columns you chose to clean
        """
        null_counts = self.cleaned_df.isnull().sum()
        self.assertEqual(
            null_counts.sum(), 0,
            f"Expected 0 missing values across all cleaned columns, but found:\n{null_counts[null_counts > 0]}"
        )
        # Ensure Age and Embarked specifically have 0 nulls
        self.assertEqual(self.cleaned_df['Age'].isnull().sum(), 0)
        self.assertTrue('Embarked' not in self.cleaned_df.columns or self.cleaned_df['Embarked'].isnull().sum() == 0)

    def test_criterion_2_feature_engineering_sanity(self):
        """
        Done When Criterion 2:
        The new feature columns (family_size, is_alone) contain correct, sensible values
        """
        # 1. family_size must equal SibSp + Parch + 1
        expected_family_size = self.raw_df['SibSp'] + self.raw_df['Parch'] + 1
        pd.testing.assert_series_equal(
            self.cleaned_df['family_size'],
            expected_family_size,
            check_names=False
        )

        # 2. Minimum family size must be 1 (the passenger themselves)
        self.assertGreaterEqual(self.cleaned_df['family_size'].min(), 1)
        self.assertLessEqual(self.cleaned_df['family_size'].max(), 20)

        # 3. is_alone must be 1 (or True) when family_size == 1, and 0 (or False) when family_size > 1
        expected_is_alone = (self.cleaned_df['family_size'] == 1).astype(int)
        pd.testing.assert_series_equal(
            self.cleaned_df['is_alone'],
            expected_is_alone,
            check_names=False
        )

        # Check distribution: 537 solo passengers, 354 with family
        self.assertEqual((self.cleaned_df['is_alone'] == 1).sum(), 537)
        self.assertEqual((self.cleaned_df['is_alone'] == 0).sum(), 354)

    def test_criterion_3_categorical_columns_numeric_after_encoding(self):
        """
        Done When Criterion 3:
        The categorical columns are numeric after encoding
        """
        # Categorical columns Sex and Embarked should be converted to numeric dummy columns
        self.assertIn('Sex_male', self.cleaned_df.columns)
        self.assertIn('Embarked_Q', self.cleaned_df.columns)
        self.assertIn('Embarked_S', self.cleaned_df.columns)

        # Verify numeric dtypes for encoded columns
        self.assertTrue(np.issubdtype(self.cleaned_df['Sex_male'].dtype, np.number))
        self.assertTrue(np.issubdtype(self.cleaned_df['Embarked_Q'].dtype, np.number))
        self.assertTrue(np.issubdtype(self.cleaned_df['Embarked_S'].dtype, np.number))

        # Check values are strictly binary 0 or 1
        self.assertTrue(set(self.cleaned_df['Sex_male'].unique()).issubset({0, 1}))
        self.assertTrue(set(self.cleaned_df['Embarked_Q'].unique()).issubset({0, 1}))
        self.assertTrue(set(self.cleaned_df['Embarked_S'].unique()).issubset({0, 1}))

    def test_criterion_4_titanic_clean_csv_reload(self):
        """
        Done When Criterion 4:
        titanic_clean.csv loads back into pandas with the expected shape and columns
        """
        self.assertTrue(os.path.exists(self.clean_path), f"File {self.clean_path} does not exist.")

        loaded_df = pd.read_csv(self.clean_path)
        # Expected rows: exactly 891
        self.assertEqual(len(loaded_df), 891, f"Expected 891 rows, got {len(loaded_df)}")

        # Expected required columns
        required_cols = [
            'PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
            'family_size', 'is_alone', 'has_cabin', 'fare_per_person',
            'Sex_male', 'Embarked_Q', 'Embarked_S'
        ]
        for col in required_cols:
            self.assertIn(col, loaded_df.columns, f"Required column '{col}' missing from clean CSV.")

        # Zero missing values in loaded CSV
        self.assertEqual(loaded_df.isnull().sum().sum(), 0, "Loaded clean CSV has unexpected missing values.")

    def test_immutability_of_raw_data(self):
        """
        Verify clean_data() does not mutate the original raw DataFrame.
        """
        raw_copy = self.raw_df.copy()
        _ = clean_data(self.raw_df)
        pd.testing.assert_frame_equal(self.raw_df, raw_copy)

    def test_stretch_goal_age_buckets(self):
        """
        Verify stretch goal: age_group categories and survival differences.
        """
        self.assertIn('age_group', self.cleaned_df.columns)
        age_surv = self.cleaned_df.groupby('age_group', observed=False)['Survived'].mean()
        # Child survival rate should be noticeably higher than senior survival rate
        self.assertGreater(age_surv['child'], age_surv['senior'])


if __name__ == '__main__':
    unittest.main()

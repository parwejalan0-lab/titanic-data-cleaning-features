"""
Builds and executes titanic_clean_and_engineer.ipynb with all required
sections, markdown justifications, code cells, and inline visualizations.
"""

import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

nb = nbf.v4.new_notebook()

cells = []

# Title cell
cells.append(nbf.v4.new_markdown_cell("""# Week 2: Clean Data and Engineer Features
**Dataset:** Titanic Passenger Survival  
**Level:** Beginner+ | **Author:** Data Science Engineering Project  

---

### 🎯 This Week's Objectives
1. **Handle Missing Values:** Implement domain-justified strategies for `Age`, `Embarked`, and `Cabin` using `.fillna()` and feature extraction.
2. **Pure Data Cleaning Pipeline:** Write a modular, immutable `clean_data(df)` function that preserves raw data integrity.
3. **Feature Engineering:** Create high-signal predictive features:
   - `family_size`: Combining siblings, spouses, parents, children, and self.
   - `is_alone`: Binary indicator for solo travelers.
   - `has_cabin`: Capturing the socioeconomic survival advantage of cabin assignment.
   - `fare_per_person`: Effective per-ticket passenger cost.
4. **Categorical Encoding:** Transform `Sex` and `Embarked` into model-ready numeric dummy columns using `pd.get_dummies()`.
5. **Outlier Detection & Decision:** Audit `Fare` outliers using boxplots and the Interquartile Range (IQR) rule, documenting whether to cap or keep them.
6. **Groupby Analysis Re-run:** Re-examine baseline survival patterns to confirm consistency and evaluate new engineered features.
7. **🚀 Stretch Goal:** Bucket `Age` into discrete generational life stages (`child`, `teen`, `adult`, `senior`) using `pd.cut` and analyze survival gradients.
8. **Export & Verify:** Save cleaned dataset to `data/titanic_clean.csv` and validate zero nulls and expected schema.
"""))

# Imports & Setup
cells.append(nbf.v4.new_code_cell("""# Core scientific computing and visualization libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Visual formatting settings
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

print("Libraries imported successfully!")
print(f"Pandas version: {pd.__version__}")
"""))

# Section 1: Loading Raw Data & Missing Value Audit
cells.append(nbf.v4.new_markdown_cell("""---
## 1. Loading Raw Data & Missing Value Audit

Before designing imputation strategies, we load the raw dataset and systematically audit missing values, percentages, and data types.
"""))

cells.append(nbf.v4.new_code_cell("""# Load the raw Titanic dataset
data_path = os.path.join("data", "titanic_raw.csv")
df_raw = pd.read_csv(data_path)

print(f"Raw Dataset Dimensions: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns\\n")
display(df_raw.head())
"""))

cells.append(nbf.v4.new_code_cell("""# Missing value summary table
missing_count = df_raw.isnull().sum()
missing_pct = (missing_count / len(df_raw)) * 100

missing_summary = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing Percentage (%)': missing_pct.round(2),
    'Data Type': df_raw.dtypes
})

missing_summary = missing_summary[missing_summary['Missing Count'] > 0].sort_values(
    by='Missing Count', ascending=False
)
print("=== Columns with Missing Values in Raw Data ===")
display(missing_summary)
"""))

# Section 2: Strategy Justification
cells.append(nbf.v4.new_markdown_cell("""---
## 2. Missing Data Handling: Strategies & Justifications

Here we document and justify our treatment strategy for each column with missing values:

### 1. `Age` (177 missing values, 19.87%)
* **Distribution Profile:** `Age` is continuous, positive, and moderately right-skewed with extreme values on the upper end (up to 80 years old).
* **Strategy:** Impute missing ages using the **median age grouped by passenger class (`Pclass`) and gender (`Sex`)**.
* **Justification:**
  - The mean is heavily pulled by older passengers, whereas the median reflects the robust 50th percentile.
  - Demographic distribution varied dramatically across travel classes: 1st class passengers were substantially older (median ~38 for males, ~35 for females) than 3rd class passengers (median ~25 for males, ~21.5 for females). Grouping preserves these real socioeconomic age distributions rather than flattening everyone into a single artificial average of 28.

### 2. `Embarked` (2 missing values, 0.22%)
* **Distribution Profile:** Categorical port of embarkation (`S` = Southampton, `C` = Cherbourg, `Q` = Queenstown).
* **Strategy:** Impute the 2 missing records with the **mode (`'S'`)**.
* **Justification:**
  - Over 72% of all passengers boarded at Southampton (`S`).
  - With only 2 out of 891 records missing (0.22%), modal imputation minimizes distortion without necessitating the permanent removal of two rows.

### 3. `Cabin` (687 missing values, 77.10%)
* **Distribution Profile:** Alphanumeric cabin designation (e.g., `C85`, `B51 B53 B55`).
* **Strategy:** Do **not** impute the text directly; instead, create a binary indicator **`has_cabin = Cabin.notnull().astype(int)`**, and drop the raw `Cabin` string column.
* **Justification:**
  - Over 77% of data is missing, making string or deck-level imputation speculative and prone to severe noise.
  - However, the *fact of missingness itself* is highly predictive: passengers with recorded cabins were overwhelmingly in 1st class and survived at ~66.7%, whereas those without a recorded cabin survived at only ~29.98%. Capturing this in `has_cabin` preserves 100% of the survival signal while eliminating high-cardinality missingness.

### 4. `Fare` (0 missing in train, but potential in test/real-world)
* **Strategy:** Fallback imputation using the **median fare grouped by `Pclass`**.
* **Justification:** Guarantees production pipeline resilience against missing test records (such as Kaggle test passenger 1044).
"""))

# Inspection of group medians
cells.append(nbf.v4.new_code_cell("""# Empirical proof of Class & Sex age variation
age_medians_table = df_raw.groupby(['Pclass', 'Sex'])['Age'].agg(['count', 'median', 'mean']).round(1)
print("=== Empirical Age Distributions by Pclass and Sex ===")
display(age_medians_table)
"""))

# Section 3: Clean Data Pipeline Function
cells.append(nbf.v4.new_markdown_cell("""---
## 3. Writing the `clean_data(df)` Pipeline Function

We now package all cleaning rules and feature transformations into a pure function `clean_data(df)`.  
Following production best practices:
- It creates a deep copy (`df.copy()`) so the original raw data remains untouched.
- It returns a clean, transformed DataFrame ready for modeling.
"""))

cells.append(nbf.v4.new_code_cell("""def clean_data(
    data: pd.DataFrame,
    drop_first: bool = True,
    encode_categoricals: bool = True,
    drop_unmodeled: bool = True
) -> pd.DataFrame:
    \"\"\"
    Cleans raw Titanic DataFrame and engineers model-ready features.
    Guarantees raw data immutability by operating on a deep copy.
    \"\"\"
    df_clean = data.copy()

    # 1. Impute Embarked with Mode ('S')
    embarked_mode = df_clean['Embarked'].mode()[0] if not df_clean['Embarked'].dropna().empty else 'S'
    df_clean['Embarked'] = df_clean['Embarked'].fillna(embarked_mode)

    # 2. Impute Age with conditional median by Pclass and Sex
    if 'Pclass' in df_clean.columns and 'Sex' in df_clean.columns:
        median_ages = df_clean.groupby(['Pclass', 'Sex'])['Age'].transform('median')
        df_clean['Age'] = df_clean['Age'].fillna(median_ages)
    df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())

    # 3. Impute Fare (if missing) with conditional median by Pclass
    if 'Fare' in df_clean.columns:
        if 'Pclass' in df_clean.columns:
            median_fares = df_clean.groupby('Pclass')['Fare'].transform('median')
            df_clean['Fare'] = df_clean['Fare'].fillna(median_fares)
        df_clean['Fare'] = df_clean['Fare'].fillna(df_clean['Fare'].median())

    # 4. Feature Engineering
    # Total family size (siblings/spouses + parents/children + self)
    df_clean['family_size'] = df_clean['SibSp'] + df_clean['Parch'] + 1
    
    # Binary solo traveler indicator
    df_clean['is_alone'] = (df_clean['family_size'] == 1).astype(int)

    # Cabin presence indicator
    df_clean['has_cabin'] = df_clean['Cabin'].notnull().astype(int)

    # Ticket fare per family member
    df_clean['fare_per_person'] = (df_clean['Fare'] / df_clean['family_size']).round(2)

    # Stretch Goal: Life-stage age bucketing
    age_bins = [0, 12, 18, 60, 120]
    age_labels = ['child', 'teen', 'adult', 'senior']
    df_clean['age_group'] = pd.cut(df_clean['Age'], bins=age_bins, labels=age_labels, right=False)
    df_clean['age_group_code'] = df_clean['age_group'].cat.codes.astype(int)

    # 5. Categorical Encoding via pd.get_dummies
    if encode_categoricals:
        cat_cols = [c for c in ['Sex', 'Embarked'] if c in df_clean.columns]
        df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=drop_first, dtype=int)

    # 6. Drop unmodeled high-cardinality/messy raw text columns
    if drop_unmodeled:
        cols_to_drop = ['Cabin', 'Ticket', 'Name']
        df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])

    return df_clean

# Apply cleaning function
df_cleaned = clean_data(df_raw)
print(f"Cleaned DataFrame Shape: {df_cleaned.shape}")
print(f"Missing Values Across Cleaned Columns: {df_cleaned.isnull().sum().sum()}")
display(df_cleaned.head())
"""))

# Verification of raw immutability
cells.append(nbf.v4.new_code_cell("""# Verify that raw data was NOT modified
print(f"Raw data Age nulls: {df_raw['Age'].isnull().sum()} (Must be 177)")
print(f"Raw data Embarked nulls: {df_raw['Embarked'].isnull().sum()} (Must be 2)")
print(f"Raw data columns count: {len(df_raw.columns)} (Must be 12)")
assert df_raw['Age'].isnull().sum() == 177, "Raw data was accidentally mutated!"
assert len(df_raw.columns) == 12, "Raw data columns were altered!"
print("\\n[PASSED] Raw data remains completely untouched and immutable.")
"""))

# Section 4: Feature Engineering Deep-Dive
cells.append(nbf.v4.new_markdown_cell("""---
## 4. Feature Engineering Deep-Dive

We engineered several key features:
1. `family_size = SibSp + Parch + 1`
2. `is_alone = (family_size == 1)`
3. `has_cabin = Cabin.notnull()`
4. `fare_per_person = Fare / family_size`
5. `age_group`: Generational buckets via `pd.cut`

Let us examine the distribution and survival dynamics of these engineered variables.
"""))

cells.append(nbf.v4.new_code_cell("""# Summary statistics of new engineered features
display(df_cleaned[['family_size', 'is_alone', 'has_cabin', 'fare_per_person']].describe().round(2))
"""))

cells.append(nbf.v4.new_code_cell("""# Visualizing Survival by Family Size and Travel Status
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
baseline_rate = df_cleaned['Survived'].mean()

# Left: Survival by Family Size
fam_summary = df_cleaned.groupby('family_size')['Survived'].agg(['mean', 'count']).reset_index()
sns.barplot(data=fam_summary, x='family_size', y='mean', ax=axes[0], hue='family_size', palette='viridis', legend=False)
axes[0].axhline(baseline_rate, color='red', linestyle='--', label=f'Baseline ({baseline_rate:.1%})')
axes[0].set_title("Survival Rate by Family Size (Sweet Spot: 2 to 4)", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Family Size (SibSp + Parch + 1)")
axes[0].set_ylabel("Survival Rate")
axes[0].set_ylim(0, 0.85)
axes[0].legend()

# Right: Solo vs Family Travelers
alone_summary = df_cleaned.groupby('is_alone')['Survived'].agg(['mean', 'count']).reset_index()
alone_summary['status'] = alone_summary['is_alone'].map({1: 'Alone (n=537)', 0: 'With Family (n=354)'})
sns.barplot(data=alone_summary, x='status', y='mean', ax=axes[1], hue='status', palette=['#e74c3c', '#2ecc71'], legend=False)
axes[1].axhline(baseline_rate, color='blue', linestyle='--', label=f'Baseline ({baseline_rate:.1%})')
axes[1].set_title("Solo Travelers vs. Family Groups (+20.2% Survival Boost)", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Travel Status")
axes[1].set_ylabel("Survival Rate")
axes[1].set_ylim(0, 0.7)
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

# Section 5: Categorical Encoding
cells.append(nbf.v4.new_markdown_cell("""---
## 5. Categorical Encoding with `pandas.get_dummies()`

Machine learning models require numerical feature inputs. We transformed the categorical features `Sex` and `Embarked` using one-hot encoding with `drop_first=True`:

- **`Sex`** (`female`, `male`) $\\rightarrow$ `Sex_male` (1 if male, 0 if female).
- **`Embarked`** (`C`, `Q`, `S`) $\\rightarrow$ `Embarked_Q` and `Embarked_S` (with Cherbourg `C` as reference baseline when both are 0).
- **Why `drop_first=True`?** Dropping the reference column avoids the **dummy variable trap** (perfect multicollinearity where $x_1 + x_2 = 1$).
"""))

cells.append(nbf.v4.new_code_cell("""# Inspect encoded categorical columns
encoded_cols = ['Sex_male', 'Embarked_Q', 'Embarked_S']
print("=== Encoded Categorical Columns Sample ===")
display(df_cleaned[encoded_cols].head())

print("\\nData types of encoded columns:")
print(df_cleaned[encoded_cols].dtypes)
"""))

# Section 6: Outlier Detection on Fare
cells.append(nbf.v4.new_markdown_cell("""---
## 6. Outlier Detection on `Fare` & Architectural Decision

We now inspect the `Fare` column using standard boxplots and the Interquartile Range (IQR) rule:
$$\\text{IQR} = Q_3 - Q_1$$
$$\\text{Upper Outlier Fence} = Q_3 + 1.5 \\times \\text{IQR}$$
"""))

cells.append(nbf.v4.new_code_cell("""# Calculate IQR and fences
q1 = df_raw['Fare'].quantile(0.25)
q3 = df_raw['Fare'].quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
fare_outliers = df_raw[df_raw['Fare'] > upper_fence]

print(f"25th Percentile (Q1): ${q1:.2f}")
print(f"75th Percentile (Q3): ${q3:.2f}")
print(f"Interquartile Range (IQR): ${iqr:.2f}")
print(f"Upper Fence Threshold (Q3 + 1.5*IQR): ${upper_fence:.2f}")
print(f"Number of Fare Outliers: {len(fare_outliers)} ({len(fare_outliers)/len(df_raw)*100:.1f}% of all passengers)")
print(f"Maximum Ticket Fare: ${df_raw['Fare'].max():.2f}")
print(f"99th Percentile Fare: ${df_raw['Fare'].quantile(0.99):.2f}")
"""))

cells.append(nbf.v4.new_code_cell("""# Visualizing Fare distribution and Outliers
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Overall Fare Boxplot
sns.boxplot(y=df_raw['Fare'], ax=axes[0], color='#3498db', width=0.35,
            flierprops=dict(marker='o', markersize=6, markerfacecolor='#e74c3c', alpha=0.6))
axes[0].axhline(upper_fence, color='darkred', linestyle='--', linewidth=2, label=f'1.5*IQR Fence (${upper_fence:.2f})')
axes[0].set_title(f"Overall Fare Distribution\\n({len(fare_outliers)} Outliers > ${upper_fence:.2f})", fontsize=13, fontweight='bold')
axes[0].set_ylabel("Ticket Fare ($)")
axes[0].legend(loc='upper right')

# Fare by Passenger Class Boxplot
df_plot = df_raw.copy()
df_plot['Class_Label'] = df_plot['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
sns.boxplot(x='Class_Label', y='Fare', data=df_plot, ax=axes[1], hue='Class_Label',
            palette=['#2980b9', '#3498db', '#85c1e9'], legend=False, width=0.45,
            flierprops=dict(marker='o', markersize=5, markerfacecolor='#e74c3c', alpha=0.6))
axes[1].set_title("Fare Distribution by Class\\n(104 of 116 Outliers Belong to 1st Class)", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Passenger Class")
axes[1].set_ylabel("Ticket Fare ($)")

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### 💡 Outlier Decision: Cap vs. Keep

#### Who are the outliers?
- 104 out of the 116 statistical outliers (89.7%) were **1st Class passengers**.
- The top 3 tickets (\$512.33) were purchased by the Cardeza family party (Thomas Drake Martinez Cardeza, Miss Anna Ward, and Gustave Lesurer), who occupied the luxurious three-room First Class Parlor Suite (B51-53-55).

#### Decision: **KEEP the authentic raw fares, with optional log-transformation or 99th-percentile capping for sensitive models.**
1. **Valid Data, Not Entry Errors:** Unlike a faulty sensor or clerical typo (e.g. recording 9999), these fares represent verified historical purchases reflecting actual economic status.
2. **Avoiding Information Loss:** Truncating or dropping these 116 rows would discard 13% of the dataset and obscure the strongest signal of passenger survival (wealth and suite deck proximity).
3. **Model Choice Alignment:**
   - **Tree-based algorithms (Decision Trees, Random Forests, XGBoost):** Trees rely on rank-based threshold splits. Extreme values do **not** distort decision boundaries. Keeping raw fares is strictly optimal.
   - **Linear & Distance-based models (Logistic Regression, SVM, KNN):** Extreme fares can exert undue leverage. For these algorithms, applying a **log-transform** (`np.log1p(df['Fare'])`) or **capping at the 99th percentile (\$249.01)** stabilizes gradient descent without discarding observations.
"""))

# Section 7: Re-running Week 1 Groupby Analysis
cells.append(nbf.v4.new_markdown_cell("""---
## 7. Re-running Week 1 Groupby Analysis on Cleaned Data

We re-run our core exploratory groupbys on the cleaned data to verify that imputation and encoding preserved baseline distributions and to benchmark the new features.
"""))

cells.append(nbf.v4.new_code_cell("""# Groupby 1: Survival by Gender (using encoded Sex_male: 0 = Female, 1 = Male)
gender_surv = df_cleaned.groupby('Sex_male')['Survived'].agg(
    Passengers='count',
    Survivors='sum',
    Survival_Rate='mean'
).round(4)
gender_surv.index = ['Female (0)', 'Male (1)']
print("=== Survival by Gender ===")
display(gender_surv)

# Groupby 2: Survival by Passenger Class
class_surv = df_cleaned.groupby('Pclass')['Survived'].agg(
    Passengers='count',
    Survivors='sum',
    Survival_Rate='mean'
).round(4)
class_surv.index = ['1st Class', '2nd Class', '3rd Class']
print("\\n=== Survival by Passenger Class ===")
display(class_surv)

# Groupby 3: Survival by Solo Travel Status (is_alone)
alone_surv = df_cleaned.groupby('is_alone')['Survived'].agg(
    Passengers='count',
    Survivors='sum',
    Survival_Rate='mean'
).round(4)
alone_surv.index = ['With Family (0)', 'Alone (1)']
print("\\n=== Survival by Solo Travel Status ===")
display(alone_surv)

# Groupby 4: Survival by Cabin Assignment (has_cabin)
cabin_surv = df_cleaned.groupby('has_cabin')['Survived'].agg(
    Passengers='count',
    Survivors='sum',
    Survival_Rate='mean'
).round(4)
cabin_surv.index = ['No Recorded Cabin (0)', 'Recorded Cabin (1)']
print("\\n=== Survival by Cabin Assignment ===")
display(cabin_surv)
"""))

# Section 8: Stretch Goal
cells.append(nbf.v4.new_markdown_cell("""---
## 8. 🚀 Stretch Goal: Age Buckets & Survival Rates

Using `pd.cut`, we bucketed passenger ages into 4 life stages:
- **Child:** 0 to 11.9 years
- **Teen:** 12 to 17.9 years
- **Adult:** 18 to 59.9 years
- **Senior:** 60+ years

Let us evaluate whether age grouping uncovers non-linear survival dynamics.
"""))

cells.append(nbf.v4.new_code_cell("""# Groupby analysis across age buckets
age_group_surv = df_cleaned.groupby('age_group', observed=False)['Survived'].agg(
    Passengers='count',
    Survivors='sum',
    Survival_Rate='mean'
).round(4)

print("=== Survival Rates Across Age Buckets ===")
display(age_group_surv)
"""))

cells.append(nbf.v4.new_code_cell("""# Visualizing Age Bucket Survival
plt.figure(figsize=(9, 5))
bars = sns.barplot(
    data=age_group_surv.reset_index(),
    x='age_group',
    y='Survival_Rate',
    hue='age_group',
    palette='flare',
    legend=False
)
plt.axhline(baseline_rate, color='black', linestyle='--', linewidth=1.5, label=f'Overall Baseline ({baseline_rate:.1%})')
plt.title("Survival Rate Across Age Buckets\\n('Women and Children First' Protocol)", fontsize=13, fontweight='bold')
plt.xlabel("Age Group Category", fontsize=11)
plt.ylabel("Survival Rate", fontsize=11)
plt.ylim(0, 0.75)
plt.legend(loc='upper right')

for p in bars.patches:
    h = p.get_height()
    plt.annotate(f"{h:.1%}", (p.get_x() + p.get_width() / 2., h + 0.02),
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### 💡 Key Takeaway from Age Bucketing:
- **Children (<12)** had the highest survival rate (**57.4%**), +19 percentage points above the overall passenger baseline (38.4%), proving the historical adherence to the "women and children first" maritime evacuation protocol.
- **Teens (12-17)** followed with a **48.9%** survival rate.
- **Adults (18-59)** had a **36.4%** survival rate.
- **Seniors (60+)** had the lowest survival rate (**26.9%**), facing mobility challenges and lower lifeboat placement priority.
"""))

# Section 9: Saving Cleaned Dataset & Done When Verification
cells.append(nbf.v4.new_markdown_cell("""---
## 9. Saving the Cleaned Dataset & Verification Checklist

We save the final cleaned DataFrame to `data/titanic_clean.csv` so subsequent weeks can load it directly.
"""))

cells.append(nbf.v4.new_code_cell("""# Export cleaned dataset
output_path = os.path.join("data", "titanic_clean.csv")
df_cleaned.to_csv(output_path, index=False)
print(f"[OK] Cleaned dataset successfully exported to: {output_path}")

# Reload into pandas to verify integrity
df_reloaded = pd.read_csv(output_path)
print(f"[OK] Reloaded shape: {df_reloaded.shape[0]} rows, {df_reloaded.shape[1]} columns")
display(df_reloaded.head(3))
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Final Acceptance Verification Checklist ('Done When')
# ==============================================================================
checks = []

# Criterion 1: Zero missing values in cleaned columns
zero_nulls = df_cleaned.isnull().sum().sum() == 0
checks.append(("Criterion 1: Zero missing values in cleaned columns", zero_nulls))

# Criterion 2: family_size and is_alone contain correct, sensible values
fam_valid = (
    (df_cleaned['family_size'] == df_raw['SibSp'] + df_raw['Parch'] + 1).all() and
    (df_cleaned['is_alone'] == (df_cleaned['family_size'] == 1).astype(int)).all()
)
checks.append(("Criterion 2: family_size and is_alone logic & values", fam_valid))

# Criterion 3: Categorical columns are numeric after encoding
cats_numeric = (
    np.issubdtype(df_cleaned['Sex_male'].dtype, np.number) and
    np.issubdtype(df_cleaned['Embarked_Q'].dtype, np.number) and
    np.issubdtype(df_cleaned['Embarked_S'].dtype, np.number)
)
checks.append(("Criterion 3: Categorical columns are numeric after encoding", cats_numeric))

# Criterion 4: titanic_clean.csv loads back with expected shape and columns
csv_valid = (
    df_reloaded.shape == df_cleaned.shape and
    df_reloaded.isnull().sum().sum() == 0 and
    'family_size' in df_reloaded.columns and
    'is_alone' in df_reloaded.columns and
    'Sex_male' in df_reloaded.columns
)
checks.append(("Criterion 4: titanic_clean.csv reload shape & columns match", csv_valid))

# Criterion 5 (Stretch Goal): Age bucket feature present and valid
age_valid = 'age_group' in df_cleaned.columns and df_cleaned['age_group'].isnull().sum() == 0
checks.append(("Criterion 5 (Stretch): Age bucket categories present and valid", age_valid))

print("=== Done When Acceptance Checklist Summary ===")
all_passed = True
for name, passed in checks:
    status = "[PASSED]" if passed else "[FAILED]"
    print(f"{status} - {name}")
    if not passed:
        all_passed = False

assert all_passed, "One or more verification checks failed!"
print("\\n🎉 ALL WEEK 2 REQUIREMENTS & CRITERIA SUCCESSFULLY VERIFIED!")
"""))

nb.cells = cells

# Save notebook file
notebook_path = os.path.join(r"C:\Users\sattar\.gemini\antigravity\scratch\titanic_data_cleaning_features", "titanic_clean_and_engineer.ipynb")
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created notebook at: {notebook_path}")

# Execute the notebook to embed all outputs and figures
print("Executing notebook with nbconvert ExecutePreprocessor...")
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_to_run = nbf.read(f, as_version=4)

ep.preprocess(nb_to_run, {'metadata': {'path': os.path.dirname(notebook_path)}})

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb_to_run, f)

print("Notebook successfully executed and saved with all outputs embedded!")

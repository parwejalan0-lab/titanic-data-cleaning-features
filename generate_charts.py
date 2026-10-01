"""
Chart Generation Script for Titanic Week 2: Clean Data & Engineer Features
Generates publication-quality charts for outliers, feature engineering, and age bucketing.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
project_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_dir, 'data', 'titanic_raw.csv')
charts_dir = os.path.join(project_dir, 'charts')
os.makedirs(charts_dir, exist_ok=True)

df = pd.read_csv(data_path)

# Set global aesthetic styling
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
baseline_survival = df['Survived'].mean()

# ==============================================================================
# Chart 1: Fare Distribution & Outliers (Overall & by Pclass)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

q1 = df['Fare'].quantile(0.25)
q3 = df['Fare'].quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr

# Overall Boxplot
sns.boxplot(
    y=df['Fare'],
    ax=axes[0],
    color='#3498db',
    width=0.4,
    flierprops=dict(marker='o', markersize=6, markerfacecolor='#e74c3c', alpha=0.6)
)
axes[0].axhline(upper_fence, color='#c0392b', linestyle='--', linewidth=2, label=f'1.5*IQR Fence (${upper_fence:.2f})')
axes[0].set_title(f'Titanic Fare Distribution & Outliers\n(116 passengers > ${upper_fence:.2f})', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Ticket Fare ($)', fontsize=12)
axes[0].legend(loc='upper right')

# Boxplot by Pclass
df_chart1 = df.copy()
df_chart1['Pclass_label'] = df_chart1['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
sns.boxplot(
    x='Pclass_label',
    y='Fare',
    data=df_chart1,
    ax=axes[1],
    hue='Pclass_label',
    palette=['#2980b9', '#3498db', '#85c1e9'],
    legend=False,
    width=0.5,
    flierprops=dict(marker='o', markersize=5, markerfacecolor='#e74c3c', alpha=0.6)
)
axes[1].set_title('Fare Outliers by Passenger Class\n(104 of 116 Outliers Belong to 1st Class)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Passenger Class', fontsize=12)
axes[1].set_ylabel('Ticket Fare ($)', fontsize=12)

plt.tight_layout()
fare_chart_path = os.path.join(charts_dir, 'fare_boxplot_outliers.png')
plt.savefig(fare_chart_path, dpi=300)
plt.close()
print(f'[OK] Saved: {fare_chart_path}')


# ==============================================================================
# Chart 2: Survival by Family Size & Is Alone
# ==============================================================================
df['family_size'] = df['SibSp'] + df['Parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# By Family Size
fam_stats = df.groupby('family_size')['Survived'].agg(['mean', 'count']).reset_index()
bars1 = sns.barplot(
    x='family_size',
    y='mean',
    data=fam_stats,
    ax=axes[0],
    hue='family_size',
    palette='crest',
    legend=False
)
axes[0].axhline(baseline_survival, color='#e74c3c', linestyle='--', linewidth=1.8, label=f'Baseline ({baseline_survival:.1%})')
axes[0].set_title('Survival Rate by Family Size\n(Sweet Spot: Small Families of 2-4)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Family Size (SibSp + Parch + 1)', fontsize=12)
axes[0].set_ylabel('Survival Rate', fontsize=12)
axes[0].set_ylim(0, 0.85)
axes[0].legend(loc='upper right')

for p in bars1.patches:
    h = p.get_height()
    if h > 0:
        axes[0].annotate(f'{h:.1%}', (p.get_x() + p.get_width() / 2., h + 0.02),
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

# By Is Alone
alone_stats = df.groupby('is_alone')['Survived'].agg(['mean', 'count']).reset_index()
alone_stats['label'] = alone_stats['is_alone'].map({1: 'Alone (n=537)', 0: 'With Family (n=354)'})
bars2 = sns.barplot(
    x='label',
    y='mean',
    data=alone_stats,
    ax=axes[1],
    hue='label',
    palette=['#e74c3c', '#27ae60'],
    legend=False
)
axes[1].axhline(baseline_survival, color='#2980b9', linestyle='--', linewidth=1.8, label=f'Baseline ({baseline_survival:.1%})')
axes[1].set_title('Solo Travelers vs. Family Groups\n(Family Travelers Had a +20.2% Survival Boost)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Travel Status', fontsize=12)
axes[1].set_ylabel('Survival Rate', fontsize=12)
axes[1].set_ylim(0, 0.7)
axes[1].legend(loc='upper right')

for p in bars2.patches:
    h = p.get_height()
    axes[1].annotate(f'{h:.1%}', (p.get_x() + p.get_width() / 2., h + 0.02),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
fam_chart_path = os.path.join(charts_dir, 'survival_by_family_and_alone.png')
plt.savefig(fam_chart_path, dpi=300)
plt.close()
print(f'[OK] Saved: {fam_chart_path}')


# ==============================================================================
# Chart 3: Survival by Age Group (Stretch Goal)
# ==============================================================================
median_ages = df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
df['Age_imputed'] = df['Age'].fillna(median_ages)
age_bins = [0, 12, 18, 60, 120]
age_labels = ['Child (<12)', 'Teen (12-17)', 'Adult (18-59)', 'Senior (60+)']
df['age_group'] = pd.cut(df['Age_imputed'], bins=age_bins, labels=age_labels, right=False)

age_stats = df.groupby('age_group', observed=False)['Survived'].agg(['mean', 'count']).reset_index()

plt.figure(figsize=(10, 6))
bars3 = sns.barplot(
    x='age_group',
    y='mean',
    data=age_stats,
    hue='age_group',
    palette='flare',
    legend=False
)
plt.axhline(baseline_survival, color='#2c3e50', linestyle='--', linewidth=2, label=f'Overall Baseline ({baseline_survival:.1%})')
plt.title('Stretch Goal: Survival Rate by Age Bucket\n("Women & Children First" Evacuation Priority)', fontsize=14, fontweight='bold')
plt.xlabel('Age Group Category', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.ylim(0, 0.75)
plt.legend(loc='upper right')

for p in bars3.patches:
    h = p.get_height()
    plt.annotate(f'{h:.1%}', (p.get_x() + p.get_width() / 2., h + 0.02),
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
age_chart_path = os.path.join(charts_dir, 'survival_by_age_bucket.png')
plt.savefig(age_chart_path, dpi=300)
plt.close()
print(f'[OK] Saved: {age_chart_path}')

print('All charts generated successfully!')

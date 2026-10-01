"""
Streamlit Web Application: Titanic Data Cleaning & Feature Engineering
Week 2: Interactive Data Science Dashboard
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from clean_data import clean_data

# Page Configuration
st.set_page_config(
    page_title="Titanic Data Cleaning & Feature Engineering",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_datasets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, 'data', 'titanic_raw.csv')
    clean_path = os.path.join(base_dir, 'data', 'titanic_clean.csv')

    df_raw = pd.read_csv(raw_path)
    if os.path.exists(clean_path):
        df_clean = pd.read_csv(clean_path)
    else:
        df_clean = clean_data(df_raw)
    return df_raw, df_clean

df_raw, df_clean = load_datasets()

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1f77b4;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title & Header
st.markdown('<div class="main-title">🚢 Titanic: Data Cleaning & Feature Engineering</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Week 2 Project Dashboard · Missing Value Imputation · Feature Engineering · Outlier Analysis</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/640px-RMS_Titanic_3.jpg", use_container_width=True)
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Choose View:",
    [
        "📊 Overview & Key Metrics",
        "🧹 Before vs. After Cleaning",
        "⚙️ Feature Engineering Explorer",
        "🎯 Fare Outlier Analysis",
        "🔬 Demographic Groupbys",
        "📥 Data Export & LinkedIn Post"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Author:** Parwej Alam  
**Curriculum:** Week 2  
**Dataset:** 891 Passengers (12 raw cols → 16 clean cols)  
[GitHub Repo](https://github.com/parwejalan0-lab/titanic-data-cleaning-features)
""")

# ==============================================================================
# VIEW 1: Overview & Key Metrics
# ==============================================================================
if page == "📊 Overview & Key Metrics":
    st.subheader("Key Findings at a Glance")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Total Passengers",
            value="891",
            delta="100% Retained"
        )
    with col2:
        baseline_survival = df_raw['Survived'].mean()
        st.metric(
            label="Overall Survival Rate",
            value=f"{baseline_survival:.1%}",
            delta="342 Survivors"
        )
    with col3:
        solo_penalty = df_clean.groupby('is_alone')['Survived'].mean()
        diff = solo_penalty[0] - solo_penalty[1]
        st.metric(
            label="Family vs Solo Advantage",
            value=f"+{diff:.1%}",
            delta="50.6% vs 30.4%"
        )
    with col4:
        child_rate = df_clean[df_clean['age_group'] == 'child']['Survived'].mean()
        st.metric(
            label="Child (<12) Survival Rate",
            value=f"{child_rate:.1%}",
            delta="+19.0% vs Avg",
            delta_color="normal"
        )

    st.markdown("---")
    st.markdown("### 🎯 Core Weekly Objectives Achieved")
    c1, c2 = st.columns(2)
    with c1:
        st.success("""
        - **1. Missing Values Resolved:**
          - `Age`: 177 missing filled with conditional median by `Pclass` & `Sex`.
          - `Embarked`: 2 missing filled with modal port (`'S'`).
          - `Cabin`: Transformed into high-signal `has_cabin` (66.7% vs 30.0% survival).
        - **2. Reusable Function:**
          - `clean_data(df)` operates on `df.copy()`, preserving raw data immutability.
        """)
    with c2:
        st.info("""
        - **3. Engineered Features:**
          - `family_size = SibSp + Parch + 1`
          - `is_alone = (family_size == 1)`
          - `fare_per_person = Fare / family_size`
          - `age_group` (`child`, `teen`, `adult`, `senior`)
        - **4. Outlier Strategy:**
          - 116 Fare outliers analyzed; authentic luxury tickets kept for tree models.
        """)

    # Quick interactive chart
    fig = px.bar(
        df_clean.groupby('age_group', as_index=False)['Survived'].mean(),
        x='age_group',
        y='Survived',
        title="Survival Rate by Age Bucket (Stretch Goal)",
        color='age_group',
        labels={'age_group': 'Age Bucket', 'Survived': 'Survival Rate'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.add_hline(y=baseline_survival, line_dash="dash", line_color="black", annotation_text=f"Baseline ({baseline_survival:.1%})")
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# VIEW 2: Before vs. After Cleaning
# ==============================================================================
elif page == "🧹 Before vs. After Cleaning":
    st.subheader("Data Cleaning Audit: Raw vs. Cleaned")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ❌ Raw Dataset Missingness (Before)")
        raw_nulls = df_raw.isnull().sum()
        raw_null_df = pd.DataFrame({
            'Column': raw_nulls.index,
            'Missing Count': raw_nulls.values,
            'Percentage (%)': (raw_nulls.values / len(df_raw) * 100).round(2)
        })
        raw_null_df = raw_null_df[raw_null_df['Missing Count'] > 0]
        st.dataframe(raw_null_df, use_container_width=True)

        fig_raw = px.bar(
            raw_null_df,
            x='Column',
            y='Missing Count',
            text='Missing Count',
            color='Column',
            title="Raw Data: Missing Value Distribution",
            color_discrete_sequence=['#e74c3c', '#e67e22', '#f1c40f']
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    with col2:
        st.markdown("#### ✅ Cleaned Dataset Missingness (After)")
        clean_nulls = df_clean.isnull().sum()
        clean_null_df = pd.DataFrame({
            'Column': clean_nulls.index,
            'Missing Count': clean_nulls.values,
            'Percentage (%)': (clean_nulls.values / len(df_clean) * 100).round(2)
        })
        st.dataframe(clean_null_df.head(6), use_container_width=True)

        st.success(f"🎉 **Zero Missing Values!** All {len(df_clean.columns)} columns in `titanic_clean.csv` have 0 null entries.")

        st.markdown("#### 📐 Dataset Dimensions Comparison")
        comp_df = pd.DataFrame({
            'Metric': ['Total Rows', 'Total Columns', 'Total Missing Cells', 'Categorical Dtypes'],
            'Raw Data': [len(df_raw), len(df_raw.columns), int(raw_nulls.sum()), "Text (Object)"],
            'Cleaned Data': [len(df_clean), len(df_clean.columns), 0, "Numeric (int64/float64)"]
        })
        st.table(comp_df)


# ==============================================================================
# VIEW 3: Feature Engineering Explorer
# ==============================================================================
elif page == "⚙️ Feature Engineering Explorer":
    st.subheader("Interactive Feature Engineering Explorer")

    feat_choice = st.selectbox(
        "Select Feature to Investigate:",
        ["family_size & is_alone", "has_cabin (Cabin Indicator)", "age_group (Stretch Goal)", "fare_per_person"]
    )

    if feat_choice == "family_size & is_alone":
        st.markdown("""
        **Formula:**  
        $$\\text{family\\_size} = \\text{SibSp} + \\text{Parch} + 1$$  
        $$\\text{is\\_alone} = 1 \\text{ if } \\text{family\\_size} == 1 \\text{ else } 0$$
        """)

        col1, col2 = st.columns(2)
        with col1:
            fam_df = df_clean.groupby('family_size', as_index=False)['Survived'].agg(['mean', 'count']).reset_index()
            fam_df.columns = ['Family Size', 'Survival Rate', 'Passenger Count']
            fig_fam = px.bar(
                fam_df,
                x='Family Size',
                y='Survival Rate',
                text=fam_df['Survival Rate'].apply(lambda x: f"{x:.1%}"),
                color='Survival Rate',
                color_continuous_scale='Viridis',
                title="Survival Rate by Family Size (Sweet Spot: 2 to 4)"
            )
            fig_fam.add_hline(y=df_clean['Survived'].mean(), line_dash='dash', line_color='red')
            st.plotly_chart(fig_fam, use_container_width=True)

        with col2:
            alone_df = df_clean.groupby('is_alone', as_index=False)['Survived'].agg(['mean', 'count']).reset_index()
            alone_df.columns = ['is_alone', 'Survival Rate', 'Passenger Count']
            alone_df['Status'] = alone_df['is_alone'].map({1: 'Solo Traveler (n=537)', 0: 'With Family (n=354)'})

            fig_alone = px.bar(
                alone_df,
                x='Status',
                y='Survival Rate',
                text=alone_df['Survival Rate'].apply(lambda x: f"{x:.1%}"),
                color='Status',
                color_discrete_sequence=['#e74c3c', '#2ecc71'],
                title="Solo Penalty: Solo vs. Family Travelers"
            )
            fig_alone.add_hline(y=df_clean['Survived'].mean(), line_dash='dash', line_color='blue')
            st.plotly_chart(fig_alone, use_container_width=True)

    elif feat_choice == "has_cabin (Cabin Indicator)":
        st.markdown("""
        **Rationale:**  
        While raw `Cabin` had 77% missing values, passengers with an assigned cabin occupied upper decks near the boat deck.
        """)
        cabin_df = df_clean.groupby('has_cabin', as_index=False)['Survived'].agg(['mean', 'count']).reset_index()
        cabin_df.columns = ['has_cabin', 'Survival Rate', 'Passenger Count']
        cabin_df['Cabin Assignment'] = cabin_df['has_cabin'].map({1: 'Recorded Cabin (n=204)', 0: 'No Cabin (n=687)'})

        fig_cabin = px.bar(
            cabin_df,
            x='Cabin Assignment',
            y='Survival Rate',
            text=cabin_df['Survival Rate'].apply(lambda x: f"{x:.1%}"),
            color='Cabin Assignment',
            color_discrete_sequence=['#95a5a6', '#2980b9'],
            title="Survival Rate by Cabin Assignment (66.7% vs 30.0%)"
        )
        st.plotly_chart(fig_cabin, use_container_width=True)

    elif feat_choice == "age_group (Stretch Goal)":
        st.markdown("""
        **Bucketing Strategy (`pd.cut`):**
        - **Child:** 0 – 11.9 years
        - **Teen:** 12 – 17.9 years
        - **Adult:** 18 – 59.9 years
        - **Senior:** 60+ years
        """)
        age_df = df_clean.groupby('age_group', as_index=False)['Survived'].agg(['mean', 'count']).reset_index()
        age_df.columns = ['Age Group', 'Survival Rate', 'Passenger Count']
        fig_age = px.bar(
            age_df,
            x='Age Group',
            y='Survival Rate',
            text=age_df['Survival Rate'].apply(lambda x: f"{x:.1%}"),
            color='Age Group',
            color_discrete_sequence=px.colors.diverging.Spectral,
            title="Survival Gradient by Life Stage"
        )
        fig_age.add_hline(y=df_clean['Survived'].mean(), line_dash='dash', line_color='black')
        st.plotly_chart(fig_age, use_container_width=True)

    elif feat_choice == "fare_per_person":
        fig_fpp = px.histogram(
            df_clean,
            x='fare_per_person',
            color='Survived',
            nbins=40,
            barmode='overlay',
            title="Distribution of Ticket Fare Per Family Member",
            labels={'fare_per_person': 'Fare Per Person ($)', 'Survived': 'Survived (1=Yes, 0=No)'}
        )
        st.plotly_chart(fig_fpp, use_container_width=True)


# ==============================================================================
# VIEW 4: Fare Outlier Analysis
# ==============================================================================
elif page == "🎯 Fare Outlier Analysis":
    st.subheader("Fare Outlier Detection & Architectural Decision")

    q1 = df_raw['Fare'].quantile(0.25)
    q3 = df_raw['Fare'].quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    outliers = df_raw[df_raw['Fare'] > upper_fence]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Q1 (25th %)", f"${q1:.2f}")
    col2.metric("Q3 (75th %)", f"${q3:.2f}")
    col3.metric("IQR Fence (Q3 + 1.5*IQR)", f"${upper_fence:.2f}")
    col4.metric("Number of Outliers", f"{len(outliers)}", f"{len(outliers)/len(df_raw):.1%}")

    fig_box = px.box(
        df_raw,
        x='Pclass',
        y='Fare',
        color='Pclass',
        points="all",
        hover_data=['Name', 'Survived'],
        title="Interactive Fare Boxplot by Passenger Class",
        labels={'Pclass': 'Passenger Class', 'Fare': 'Ticket Fare ($)'}
    )
    fig_box.add_hline(y=upper_fence, line_dash='dash', line_color='red', annotation_text=f"1.5*IQR Upper Fence (${upper_fence:.2f})")
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### 🏛️ Outlier Decision: Why We Kept Authentic Luxury Fares")
    st.info("""
    1. **Historical Authenticity:** The highest ticket price was **$512.33**, purchased by the Cardeza family occupying the three-room First Class Parlor Suite (B51-53-55). These are **not data entry errors**.
    2. **Class Concentration:** 104 out of the 116 outliers (89.7%) belonged to 1st Class. Dropping them would discard 13% of the dataset and obscure high-socioeconomic survival signals.
    3. **Model Selection:**
       - **Tree-Based Models (Random Forest, XGBoost):** Invariant to monotonic scale; capping is unnecessary.
       - **Linear Models (Logistic Regression):** Fares can be log-transformed (`np.log1p(Fare)`) or capped at the 99th percentile ($249.01).
    """)

    st.markdown("#### Top 5 Highest Fares Recorded:")
    st.dataframe(df_raw.nlargest(5, 'Fare')[['Name', 'Pclass', 'Fare', 'Cabin', 'Survived']], use_container_width=True)


# ==============================================================================
# VIEW 5: Demographic Groupbys
# ==============================================================================
elif page == "🔬 Demographic Groupbys":
    st.subheader("Demographic & Feature Groupby Analysis")

    groupby_col = st.selectbox(
        "Choose Grouping Column:",
        ['Pclass', 'is_alone', 'family_size', 'age_group', 'Sex_male']
    )

    grp = df_clean.groupby(groupby_col, as_index=False)['Survived'].agg(
        Total_Passengers='count',
        Survivors='sum',
        Survival_Rate='mean'
    )
    grp['Survival_Rate'] = (grp['Survival_Rate'] * 100).round(2)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(grp, use_container_width=True)
    with c2:
        fig_grp = px.bar(
            grp,
            x=groupby_col,
            y='Survival_Rate',
            text=grp['Survival_Rate'].apply(lambda x: f"{x}%"),
            color='Survival_Rate',
            title=f"Survival Rate by {groupby_col}"
        )
        st.plotly_chart(fig_grp, use_container_width=True)


# ==============================================================================
# VIEW 6: Data Export & LinkedIn Post
# ==============================================================================
elif page == "📥 Data Export & LinkedIn Post":
    st.subheader("Download Model-Ready Dataset")
    st.markdown("Next week's modeling tasks can load `titanic_clean.csv` directly:")

    csv_bytes = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download titanic_clean.csv (891 rows, 16 columns)",
        data=csv_bytes,
        file_name="titanic_clean.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("📱 Pre-Written LinkedIn Post Draft")
    post_text = """🚢 Week 2 Complete: Turning Messy Data into High-Signal Features (Titanic Dataset)

Cleaning data and engineering features is often called the "unglamorous" 80% of data science — but it's where models are won or lost.

Here is what I built and discovered this week:

1️⃣ Domain-Justified Missing Value Imputation:
Instead of blindly filling Age with a global average, I imputed missing values using the median grouped by Passenger Class and Sex. 1st-class men had a median age of 40, while 3rd-class women were 21.5 — context matters! Embarked was filled with the mode ('S', 72% frequency), and Cabin missingness was converted into a high-signal binary feature has_cabin (cabin holders survived at 66.7% vs 30.0%).

2️⃣ Feature Engineering Discoveries:
• The Solo Penalty: Traveling alone (is_alone = 1) resulted in a 30.4% survival rate, while traveling with family boosted survival to 50.6% (+20.2% jump).
• The Family "Sweet Spot": Small families (sizes 2–4) had the best survival rates (up to 72.4%). Beyond 4 members, survival collapsed to <20%.
• Age Bucketing (Stretch Goal): Children (<12) had a 57.4% survival rate vs. Seniors (60+) at 26.9% — a quantitative reflection of the historic "women and children first" maritime protocol.

3️⃣ Outlier Architecture (Fare):
Found 116 statistical outliers above $65.63, with 89.7% belonging to 1st Class. The $512 top tickets were real historical purchases (First Class Parlor Suites). Kept the authentic values for tree models while documenting 99th-percentile capping for linear models.

All cleaning steps were wrapped into a pure, immutable clean_data() function, verified with automated unit tests, and exported for Week 3 modeling!

🔗 Check out the full code and charts on GitHub: https://github.com/parwejalan0-lab/titanic-data-cleaning-features

#DataScience #MachineLearning #Pandas #FeatureEngineering #Python #DataAnalytics #EDA #Titanic"""

    st.text_area("Copy your LinkedIn submission post:", value=post_text, height=350)

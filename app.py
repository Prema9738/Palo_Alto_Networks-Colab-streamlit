import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Employee Engagement & Burnout Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #5f6368;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    possible_files = [
        "Palo Alto Networks.csv",
        "data/Palo Alto Networks.csv",
        "employee_data.csv",
        "data/employee_data.csv",
    ]

    for file_path in possible_files:
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            continue

    return None


st.sidebar.header("⚙️ Data & Filters")

uploaded_file = st.sidebar.file_uploader(
    "Upload employee CSV",
    type=["csv"],
    help="Upload the employee dataset if it is not already stored with the app.",
)

df = load_data(uploaded_file)

if df is None:
    st.title("📊 Employee Engagement & Burnout Intelligence")
    st.info(
        "Please upload your employee CSV using the sidebar. "
        "For Streamlit deployment, you can also place the CSV in the same "
        "repository as app.py."
    )
    st.markdown(
        """
        **Expected fields include:**
        `Department`, `JobRole`, `JobLevel`, `YearsAtCompany`,
        `YearsInCurrentRole`, `YearsSinceLastPromotion`, `OverTime`,
        `BusinessTravel`, `JobInvolvement`, `JobSatisfaction`,
        `EnvironmentSatisfaction`, `RelationshipSatisfaction`,
        `WorkLifeBalance` and `Attrition`.
        """
    )
    st.stop()

# ---------------------------------------------------------
# Data cleaning and feature engineering
# ---------------------------------------------------------
df = df.copy()
df = df.drop_duplicates()

required_columns = [
    "Department",
    "JobRole",
    "JobLevel",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "OverTime",
    "BusinessTravel",
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error("The uploaded CSV is missing required columns:")
    st.write(missing_columns)
    st.stop()

satisfaction_cols = [
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
]

for col in satisfaction_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

# Convert numeric fields safely
for col in [
    "JobLevel",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["Engagement_Index"] = df[
    [
        "JobInvolvement",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
    ]
].mean(axis=1)

df["Engagement_Score"] = (
    (df["Engagement_Index"] - 1) / 3 * 100
).clip(0, 100)


def engagement_category(score):
    if score < 50:
        return "Low"
    elif score < 75:
        return "Medium"
    return "High"


df["Engagement_Level"] = df["Engagement_Score"].apply(engagement_category)

# Burnout risk based on the original project logic:
# overtime + low work-life balance
df["Burnout_Risk_Score"] = 0
df.loc[df["OverTime"].astype(str).str.strip().str.lower() == "yes", "Burnout_Risk_Score"] += 1
df.loc[df["WorkLifeBalance"] <= 2, "Burnout_Risk_Score"] += 1


def burnout_level(score):
    if score == 2:
        return "High"
    elif score == 1:
        return "Medium"
    return "Low"


df["Burnout_Risk"] = df["Burnout_Risk_Score"].apply(burnout_level)

df["Overtime_Flag"] = np.where(
    df["OverTime"].astype(str).str.strip().str.lower() == "yes", 1, 0
)

travel_map = {
    "Non-Travel": 0,
    "Travel_Rarely": 1,
    "Travel_Frequently": 2,
}
df["Travel_Intensity"] = (
    df["BusinessTravel"].astype(str).map(travel_map).fillna(0)
)

df["Workload_Stress_Score"] = (
    df["Overtime_Flag"] + df["Travel_Intensity"]
)

df["Satisfaction_Stability"] = df[
    [
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
        "JobInvolvement",
    ]
].std(axis=1)

df["Satisfaction_Stability_Score"] = (
    100 - (df["Satisfaction_Stability"] / 1.5 * 100)
).clip(0, 100)

# ---------------------------------------------------------
# Sidebar filters - all requested user capabilities
# ---------------------------------------------------------
departments = sorted(df["Department"].dropna().astype(str).unique().tolist())
roles = sorted(df["JobRole"].dropna().astype(str).unique().tolist())

selected_departments = st.sidebar.multiselect(
    "🏢 Department",
    options=departments,
    default=departments,
)

selected_roles = st.sidebar.multiselect(
    "👤 Job Role",
    options=roles,
    default=roles,
)

overtime_only = st.sidebar.toggle(
    "⏰ Overtime employees only",
    value=False,
)

engagement_threshold = st.sidebar.slider(
    "📉 Engagement threshold",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help="Employees below this score are treated as low-engagement alerts.",
)

min_tenure = int(df["YearsAtCompany"].min())
max_tenure = int(df["YearsAtCompany"].max())

if min_tenure == max_tenure:
    tenure_range = (min_tenure, max_tenure)
else:
    tenure_range = st.sidebar.slider(
        "📅 Tenure range (years)",
        min_value=min_tenure,
        max_value=max_tenure,
        value=(min_tenure, max_tenure),
    )

filtered_df = df[
    df["Department"].astype(str).isin(selected_departments)
    & df["JobRole"].astype(str).isin(selected_roles)
    & df["YearsAtCompany"].between(tenure_range[0], tenure_range[1])
].copy()

if overtime_only:
    filtered_df = filtered_df[
        filtered_df["OverTime"].astype(str).str.strip().str.lower() == "yes"
    ].copy()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">📊 Employee Engagement & Burnout Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Preventive Employee Experience Analytics</div>',
    unsafe_allow_html=True,
)

st.write(
    "Interactive HR analytics dashboard for monitoring employee engagement, "
    "burnout risk, workload stress and career-stage patterns."
)

if filtered_df.empty:
    st.warning("No employees match the selected filters. Please adjust the filters.")
    st.stop()

# ---------------------------------------------------------
# KPI cards
# ---------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Organization Engagement Score",
    f"{filtered_df['Engagement_Score'].mean():.1f}/100",
)

col2.metric(
    "High Burnout Risk",
    f"{filtered_df['Burnout_Risk'].eq('High').mean() * 100:.1f}%",
)

col3.metric(
    "Avg Work-Life Balance",
    f"{filtered_df['WorkLifeBalance'].mean():.2f}/4",
)

col4.metric(
    "Low Engagement Alerts",
    f"{(filtered_df['Engagement_Score'] < engagement_threshold).sum():,}",
)

col5.metric(
    "Employees in View",
    f"{len(filtered_df):,}",
)

st.divider()

# ---------------------------------------------------------
# Dashboard modules
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "❤️ Engagement Health Overview",
        "🔥 Burnout Risk Dashboard",
        "📈 Role & Career Stage Analysis",
        "🚨 Manager Action Panel",
    ]
)

# =========================================================
# 1. Engagement Health Overview
# =========================================================
with tab1:
    st.subheader("Organization-wide Engagement Score")

    c1, c2 = st.columns(2)

    with c1:
        engagement_hist = px.histogram(
            filtered_df,
            x="Engagement_Score",
            nbins=20,
            title="Engagement Score Distribution",
            labels={"Engagement_Score": "Engagement Score"},
        )
        engagement_hist.add_vline(
            x=filtered_df["Engagement_Score"].mean(),
            line_dash="dash",
            annotation_text="Average",
        )
        st.plotly_chart(engagement_hist, use_container_width=True)

    with c2:
        level_order = ["Low", "Medium", "High"]
        level_counts = (
            filtered_df["Engagement_Level"]
            .value_counts()
            .reindex(level_order, fill_value=0)
            .reset_index()
        )
        level_counts.columns = ["Engagement Level", "Employees"]

        fig = px.bar(
            level_counts,
            x="Engagement Level",
            y="Employees",
            title="Engagement Level Distribution",
            text="Employees",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Satisfaction Distribution")

    satisfaction_labels = {
        "JobSatisfaction": "Job Satisfaction",
        "EnvironmentSatisfaction": "Environment Satisfaction",
        "RelationshipSatisfaction": "Relationship Satisfaction",
        "JobInvolvement": "Job Involvement",
        "WorkLifeBalance": "Work-Life Balance",
    }

    satisfaction_long = filtered_df[
        list(satisfaction_labels.keys())
    ].rename(columns=satisfaction_labels).melt(
        var_name="Measure",
        value_name="Rating",
    )

    satisfaction_fig = px.histogram(
        satisfaction_long,
        x="Rating",
        facet_col="Measure",
        facet_col_wrap=2,
        category_orders={"Rating": [1, 2, 3, 4]},
        title="Satisfaction Ratings Across Employees",
        labels={"Rating": "Rating (1–4)"},
    )
    satisfaction_fig.update_layout(height=650)
    satisfaction_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(satisfaction_fig, use_container_width=True)

# =========================================================
# 2. Burnout Risk Dashboard
# =========================================================
with tab2:
    st.subheader("High-Risk Employee Segments")

    risk_counts = (
        filtered_df["Burnout_Risk"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .reset_index()
    )
    risk_counts.columns = ["Risk", "Employees"]

    c1, c2 = st.columns(2)

    with c1:
        risk_fig = px.bar(
            risk_counts,
            x="Risk",
            y="Employees",
            title="Burnout Risk Distribution",
            text="Employees",
        )
        st.plotly_chart(risk_fig, use_container_width=True)

    with c2:
        overtime_fig = px.box(
            filtered_df,
            x="OverTime",
            y="Engagement_Score",
            title="Engagement: Overtime vs Non-Overtime",
            labels={
                "OverTime": "Overtime",
                "Engagement_Score": "Engagement Score",
            },
        )
        st.plotly_chart(overtime_fig, use_container_width=True)

    st.subheader("Overtime & Work-Life Imbalance")

    worklife_fig = px.box(
        filtered_df,
        x="WorkLifeBalance",
        y="Engagement_Score",
        color="OverTime",
        title="Engagement by Work-Life Balance and Overtime",
        labels={
            "WorkLifeBalance": "Work-Life Balance (1–4)",
            "Engagement_Score": "Engagement Score",
        },
    )
    st.plotly_chart(worklife_fig, use_container_width=True)

    high_risk = filtered_df[filtered_df["Burnout_Risk"] == "High"].copy()

    st.subheader("High-Risk Segment Details")
    if high_risk.empty:
        st.success("No high-burnout-risk employees are present in the current filter.")
    else:
        segment = (
            high_risk.groupby(["Department", "JobRole"])
            .agg(
                Employees=("Burnout_Risk", "size"),
                Avg_Engagement=("Engagement_Score", "mean"),
                Avg_WorkLifeBalance=("WorkLifeBalance", "mean"),
            )
            .reset_index()
            .sort_values(["Employees", "Avg_Engagement"], ascending=[False, True])
        )
        st.dataframe(
            segment.style.format(
                {
                    "Avg_Engagement": "{:.1f}",
                    "Avg_WorkLifeBalance": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

# =========================================================
# 3. Role & Career Stage Analysis
# =========================================================
with tab3:
    st.subheader("Engagement by Job Role and Level")

    role_summary = (
        filtered_df.groupby("JobRole")["Engagement_Score"]
        .mean()
        .reset_index()
        .sort_values("Engagement_Score")
    )

    role_fig = px.bar(
        role_summary,
        x="Engagement_Score",
        y="JobRole",
        orientation="h",
        title="Average Engagement by Job Role",
        text="Engagement_Score",
        labels={"Engagement_Score": "Average Engagement Score"},
    )
    st.plotly_chart(role_fig, use_container_width=True)

    level_summary = (
        filtered_df.groupby("JobLevel")["Engagement_Score"]
        .mean()
        .reset_index()
        .sort_values("JobLevel")
    )

    level_fig = px.bar(
        level_summary,
        x="JobLevel",
        y="Engagement_Score",
        title="Average Engagement by Job Level",
        text="Engagement_Score",
        labels={
            "JobLevel": "Job Level",
            "Engagement_Score": "Average Engagement Score",
        },
    )
    st.plotly_chart(level_fig, use_container_width=True)

    st.subheader("Tenure vs Engagement Trends")

    tenure_fig = px.scatter(
        filtered_df,
        x="YearsAtCompany",
        y="Engagement_Score",
        color="Engagement_Level",
        hover_data=["Department", "JobRole", "JobLevel"],
        title="Tenure vs Engagement",
        labels={
            "YearsAtCompany": "Years at Company",
            "Engagement_Score": "Engagement Score",
        },
    )
    st.plotly_chart(tenure_fig, use_container_width=True)

    career_summary = (
        filtered_df.groupby("YearsSinceLastPromotion")["Engagement_Score"]
        .mean()
        .reset_index()
    )

    career_fig = px.line(
        career_summary,
        x="YearsSinceLastPromotion",
        y="Engagement_Score",
        markers=True,
        title="Engagement vs Years Since Last Promotion",
        labels={
            "YearsSinceLastPromotion": "Years Since Last Promotion",
            "Engagement_Score": "Average Engagement Score",
        },
    )
    st.plotly_chart(career_fig, use_container_width=True)

# =========================================================
# 4. Manager Action Panel
# =========================================================
with tab4:
    st.subheader("Low-Engagement Alerts")

    low_engagement = filtered_df[
        filtered_df["Engagement_Score"] < engagement_threshold
    ].copy()

    if low_engagement.empty:
        st.success("No low-engagement alerts under the current threshold.")
    else:
        alert_col1, alert_col2, alert_col3 = st.columns(3)

        alert_col1.metric(
            "Employees Below Threshold",
            f"{len(low_engagement):,}"
        )

        avg_alert_score = (
            f"{low_engagement['Engagement_Score'].mean():.1f}"
        )

        alert_col2.metric(
            "Average Alert Score",
            avg_alert_score
        )

        high_risk_low_engagement = (
            (filtered_df["Burnout_Risk"] == "High") &
            (filtered_df["Engagement_Score"] < engagement_threshold)
        ).sum()

        alert_col3.metric(
            "High-Risk + Low Engagement",
            f"{high_risk_low_engagement:,}"
        )

        alert_summary = (
            low_engagement
            .groupby(["Department", "JobRole"])
            .agg(
                Employees=("Engagement_Score", "size"),
                Avg_Engagement=("Engagement_Score", "mean"),
                Avg_WorkLifeBalance=("WorkLifeBalance", "mean"),
                High_Burnout_Risk=(
                    "Burnout_Risk",
                    lambda x: (x == "High").sum(),
                ),
            )
            .reset_index()
            .sort_values(
                ["High_Burnout_Risk", "Avg_Engagement"],
                ascending=[False, True],
            )
        )

        st.dataframe(
            alert_summary.style.format(
                {
                    "Avg_Engagement": "{:.1f}",
                    "Avg_WorkLifeBalance": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Priority Intervention Areas")

    priority = filtered_df[
        (filtered_df["Burnout_Risk"] == "High")
        & (filtered_df["Engagement_Score"] < engagement_threshold)
    ].copy()

    if priority.empty:
        st.success("No priority intervention cases match the current filters.")
    else:
        priority_cols = [
            c
            for c in [
                "Department",
                "JobRole",
                "JobLevel",
                "Engagement_Score",
                "Engagement_Level",
                "Burnout_Risk",
                "OverTime",
                "WorkLifeBalance",
                "YearsAtCompany",
                "YearsSinceLastPromotion",
            ]
            if c in priority.columns
        ]

        st.dataframe(
            priority[priority_cols].sort_values("Engagement_Score"),
            use_container_width=True,
            hide_index=True,
        )

        st.warning(
            "Suggested manager focus: review workload/overtime, work-life balance, "
            "career progression and role-specific engagement for these priority cases."
        )

# ---------------------------------------------------------
# Download processed data
# ---------------------------------------------------------
st.divider()
st.subheader("📥 Export")

export_df = filtered_df.copy()

csv_data = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download filtered employee analysis (CSV)",
    data=csv_data,
    file_name="employee_engagement_analysis.csv",
    mime="text/csv",
)

st.caption(
    "Dashboard calculations follow the engagement and burnout logic defined "
    "in the supplied project code."
)

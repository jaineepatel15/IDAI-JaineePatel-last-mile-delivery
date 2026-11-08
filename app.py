"""
Last-Mile Delivery Analytics Dashboard  (FA-2)
Mathematics for AI-II | LogiSight Analytics Pvt. Ltd.

Stages covered:
  Stage 4 - Data cleaning & preparation
  Stage 5 - Five compulsory visualizations (+ 4 optional)
  Stage 6 - Streamlit interface with sidebar filters and live KPIs
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Last-Mile Delivery Analytics",
    page_icon="🚚",
    layout="wide",
)

DATA_FILE = "delivery_data.csv"          # must match your CSV file name
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FILE)

CAT_COLS = ["Weather", "Traffic", "Vehicle", "Area", "Category"]
TRAFFIC_ORDER = ["Low", "Medium", "High", "Jam"]
AGE_ORDER = ["<25", "25–40", "40+"]


def show(fig):
    """Display a plotly figure full-width (works on old and new Streamlit)."""
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# Stage 4: Data cleaning & preparation
# ----------------------------------------------------------------------------
@st.cache_data
def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates().copy()

    # 2. Tidy text columns: strip spaces, fix "NaN" strings, consistent case
    for col in CAT_COLS:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"NaN": pd.NA, "nan": pd.NA, "": pd.NA})
        df[col] = df[col].str.title()
    df["Area"] = df["Area"].replace({"Metropolitian": "Metropolitan"})  # typo in data

    # 3. Make numeric columns numeric
    for col in ["Delivery_Time", "Agent_Age", "Agent_Rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Treat impossible values as missing (rating must be 1-5, agents 18+)
    df.loc[~df["Agent_Rating"].between(1, 5), "Agent_Rating"] = np.nan
    df.loc[df["Agent_Age"] < 18, "Agent_Age"] = np.nan

    # 5. Handle missing values
    df = df.dropna(subset=["Delivery_Time"])                      # target column
    for col in CAT_COLS:                                          # categorical -> mode
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    for col in ["Agent_Age", "Agent_Rating"]:                     # numeric -> median
        df[col] = df[col].fillna(df[col].median())

    # 6. New calculated fields
    # Late delivery = Delivery_Time above (mean + 1 standard deviation)
    late_threshold = df["Delivery_Time"].mean() + df["Delivery_Time"].std()
    df["Is_Late"] = df["Delivery_Time"] > late_threshold

    df["Agent_Age_Group"] = pd.cut(
        df["Agent_Age"], bins=[0, 24, 40, 200], labels=AGE_ORDER
    ).astype(str)

    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

    return df.reset_index(drop=True), late_threshold


try:
    data, LATE_THRESHOLD = load_and_clean(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find **{DATA_FILE}**. Put it in the same folder as app.py.")
    st.stop()

# ----------------------------------------------------------------------------
# Sidebar filters (Stage 6)
# ----------------------------------------------------------------------------
st.sidebar.title("🚚 Last-Mile Delivery Analytics")
st.sidebar.header("Filters")


def sidebar_filter(label, column):
    options = sorted(data[column].unique())
    if column == "Traffic":
        options = [t for t in TRAFFIC_ORDER if t in options] + \
                  [t for t in options if t not in TRAFFIC_ORDER]
    return st.sidebar.multiselect(label, options, default=options)


sel_weather = sidebar_filter("Weather", "Weather")
sel_traffic = sidebar_filter("Traffic", "Traffic")
sel_vehicle = sidebar_filter("Vehicle Type", "Vehicle")
sel_area = sidebar_filter("Area", "Area")
sel_category = sidebar_filter("Product Category", "Category")

df = data[
    data["Weather"].isin(sel_weather)
    & data["Traffic"].isin(sel_traffic)
    & data["Vehicle"].isin(sel_vehicle)
    & data["Area"].isin(sel_area)
    & data["Category"].isin(sel_category)
]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing {len(df):,} of {len(data):,} deliveries")
st.sidebar.caption(
    f"Late delivery = longer than {LATE_THRESHOLD:.1f} min "
    "(overall mean + 1 std dev)"
)

# ----------------------------------------------------------------------------
# Header + key metrics
# ----------------------------------------------------------------------------
st.title("Last-Mile Delivery Performance Overview")
st.caption("Explore what drives delivery time. Use the sidebar filters to drill down.")

if df.empty:
    st.warning("No deliveries match the selected filters. Please widen your selection.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Deliveries", f"{len(df):,}")
k2.metric("Average Delivery Time", f"{df['Delivery_Time'].mean():.1f} min")
k3.metric("Late Deliveries", f"{df['Is_Late'].mean() * 100:.1f}%")
k4.metric("Average Agent Rating", f"{df['Agent_Rating'].mean():.2f} / 5")

st.markdown("---")

# ----------------------------------------------------------------------------
# Stage 5: Five compulsory visualizations
# ----------------------------------------------------------------------------
# Row 1 -- Delay Analyzer + Vehicle Comparison
c1, c2 = st.columns(2)

with c1:
    delay = df.groupby(["Weather", "Traffic"], as_index=False)["Delivery_Time"].mean()
    fig = px.bar(
        delay, x="Weather", y="Delivery_Time", color="Traffic", barmode="group",
        category_orders={"Traffic": TRAFFIC_ORDER},
        title="1. Delay Analyzer: Avg Delivery Time by Weather & Traffic",
        labels={"Delivery_Time": "Avg Delivery Time (min)"},
    )
    show(fig)

with c2:
    veh = (df.groupby("Vehicle", as_index=False)["Delivery_Time"].mean()
             .sort_values("Delivery_Time"))
    fig = px.bar(
        veh, x="Vehicle", y="Delivery_Time", color="Vehicle", text_auto=".1f",
        title="2. Vehicle Comparison: Avg Delivery Time by Vehicle Type",
        labels={"Delivery_Time": "Avg Delivery Time (min)"},
    )
    fig.update_layout(showlegend=False)
    show(fig)

# Row 2 -- Agent Performance + Area Heatmap
c3, c4 = st.columns(2)

with c3:
    # Sample for speed if there are many points (does not affect other charts)
    plot_df = df.sample(8000, random_state=1) if len(df) > 8000 else df
    fig = px.scatter(
        plot_df, x="Agent_Rating", y="Delivery_Time", color="Agent_Age_Group",
        opacity=0.5, category_orders={"Agent_Age_Group": AGE_ORDER},
        title="3. Agent Performance: Rating vs Delivery Time",
        labels={"Agent_Rating": "Agent Rating", "Delivery_Time": "Delivery Time (min)",
                "Agent_Age_Group": "Agent Age"},
    )
    show(fig)
    if len(df) > 8000:
        st.caption("Scatter shows a random sample of 8,000 deliveries for speed.")

with c4:
    compare = st.selectbox(
        "Heatmap: compare Area against",
        ["Traffic", "Weather", "Vehicle", "Category"],
        key="heat_dim",
    )
    pivot = df.pivot_table(index="Area", columns=compare,
                           values="Delivery_Time", aggfunc="mean")
    if compare == "Traffic":
        pivot = pivot[[t for t in TRAFFIC_ORDER if t in pivot.columns]]
    fig = px.imshow(
        pivot, text_auto=".1f", aspect="auto", color_continuous_scale="YlOrRd",
        title=f"4. Area Heatmap: Avg Delivery Time by Area & {compare}",
        labels=dict(color="Avg Time (min)"),
    )
    show(fig)

# Row 3 -- Category Visualizer
fig = px.box(
    df, x="Category", y="Delivery_Time", color="Category",
    title="5. Category Visualizer: Delivery Time Distribution by Product Category",
    labels={"Delivery_Time": "Delivery Time (min)"},
)
fig.update_layout(showlegend=False, xaxis_tickangle=-30)
show(fig)

# ----------------------------------------------------------------------------
# Optional visualizations
# ----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Additional Insights")
t1, t2, t3, t4 = st.tabs(
    ["% Late by Traffic", "% Late by Weather", "Time Distribution", "Monthly Trend"]
)

with t1:
    late = df.groupby("Traffic", as_index=False)["Is_Late"].mean()
    late["Late %"] = late["Is_Late"] * 100
    fig = px.bar(late, x="Traffic", y="Late %", color="Traffic", text_auto=".1f",
                 category_orders={"Traffic": TRAFFIC_ORDER},
                 title="Late Deliveries (%) by Traffic Condition")
    fig.update_layout(showlegend=False)
    show(fig)

with t2:
    late = df.groupby("Weather", as_index=False)["Is_Late"].mean()
    late["Late %"] = late["Is_Late"] * 100
    fig = px.bar(late, x="Weather", y="Late %", color="Weather", text_auto=".1f",
                 title="Late Deliveries (%) by Weather Condition")
    fig.update_layout(showlegend=False)
    show(fig)

with t3:
    fig = px.histogram(df, x="Delivery_Time", nbins=40,
                       title="Distribution of Delivery Time",
                       labels={"Delivery_Time": "Delivery Time (min)"})
    fig.add_vline(x=LATE_THRESHOLD, line_dash="dash", line_color="red",
                  annotation_text="Late threshold")
    show(fig)

with t4:
    trend = (df.dropna(subset=["Order_Date"])
               .groupby("Month", as_index=False)["Delivery_Time"].mean())
    if trend.empty:
        st.info("No valid order dates available for a trend chart.")
    else:
        fig = px.line(trend, x="Month", y="Delivery_Time", markers=True,
                      title="Average Delivery Time by Month",
                      labels={"Delivery_Time": "Avg Delivery Time (min)"})
        show(fig)

# ----------------------------------------------------------------------------
# Key insights (update with the filters)
# ----------------------------------------------------------------------------
st.markdown("---")
st.subheader("💡 Key Insights")

worst_weather = df.groupby("Weather")["Delivery_Time"].mean().idxmax()
worst_traffic = df.groupby("Traffic")["Delivery_Time"].mean().idxmax()
best_vehicle = df.groupby("Vehicle")["Delivery_Time"].mean().idxmin()
worst_area = df.groupby("Area")["Delivery_Time"].mean().idxmax()
worst_cat = df.groupby("Category")["Delivery_Time"].median().idxmax()

st.markdown(
    f"""
- **Weather:** *{worst_weather}* conditions have the longest average delivery time.
- **Traffic:** *{worst_traffic}* traffic is linked to the slowest deliveries.
- **Vehicles:** *{best_vehicle}* is the fastest vehicle type on average.
- **Areas:** *{worst_area}* areas see the highest average delivery time.
- **Categories:** *{worst_cat}* has the highest median delivery time.
"""
)
st.caption("Insights are calculated from the currently filtered data.")
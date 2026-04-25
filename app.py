import streamlit as st
import pandas as pd
import numpy as np
import pickle

# === Page Config ===
st.set_page_config(
    page_title="AI Marketing Decision Engine",
    page_icon="🎯",
    layout="wide"
)

# === Load Model Artifacts ===
@st.cache_resource
def load_model():
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return kmeans, scaler

@st.cache_data
def load_decisions():
    return pd.read_csv('decisions.csv')

kmeans, scaler = load_model()
decisions = load_decisions()

# === Cluster ID → Segment Name (from your training run) ===
SEGMENT_MAP = {
    0: 'New / Promising',
    1: 'Champions',
    2: 'Lost / Hibernating',
    3: 'At-Risk / Loyal Fading',
}

# === Business Rules (same dict from your Colab notebook) ===
DECISION_RULES = {
    'Champions': {
        'offer':    'Exclusive early access to new collection (no discount)',
        'priority': 'High — retain at all costs',
    },
    'At-Risk / Loyal Fading': {
        'offer':    'Win-back 20% discount on their favorite category',
        'priority': 'High — urgent reactivation',
    },
    'New / Promising': {
        'offer':    'Welcome bundle: 15% off second purchase + personalized recs',
        'priority': 'Medium — nurture to habit',
    },
    'Lost / Hibernating': {
        'offer':    'Final win-back: 40% off + free shipping (last-chance campaign)',
        'priority': 'Low — triage only',
    },
}

# === Helper 1: Predict segment for a hypothetical R/F/M ===
def predict_segment(recency, frequency, monetary):
    raw    = np.array([[recency, frequency, monetary]])
    logged = np.log1p(raw)              # same log transform as training
    scaled = scaler.transform(logged)   # same scaler as training
    cluster = int(kmeans.predict(scaled)[0])
    return SEGMENT_MAP[cluster]

# === Helper 2: Get the typical send time for a given segment ===
def get_segment_send_time(segment):
    seg = decisions[decisions['Segment'] == segment]
    return seg['BestDay'].mode()[0], int(seg['BestHour'].mode()[0])

# === Helper 3: Display the Recommendation Card ===
def show_recommendation(customer_id, recency, frequency, monetary,
                       segment, offer, priority, best_day, best_hour):
    segment_colors = {
        'Champions':              '#27ae60',
        'At-Risk / Loyal Fading': '#f39c12',
        'New / Promising':        '#3498db',
        'Lost / Hibernating':     '#e74c3c',
    }
    color = segment_colors.get(segment, '#7f8c8d')

    if customer_id is not None:
        st.markdown(f"### Customer #{int(customer_id)}")
    else:
        st.markdown("### 🆕 Hypothetical New Customer")

    col1, col2, col3 = st.columns(3)
    col1.metric("📅 Recency",   f"{int(recency)} days")
    col2.metric("🔁 Frequency", f"{int(frequency)} orders")
    col3.metric("💰 Monetary",  f"${monetary:,.0f}")

    st.markdown(
        f"#### 🏷️ Segment: <span style='color:{color}; font-weight:bold;'>{segment}</span>",
        unsafe_allow_html=True
    )
    st.info(f"🎁 **Offer:** {offer}")
    st.success(f"⏰ **Best Send Time:** {best_day} at {int(best_hour):02d}:00")
    st.warning(f"🚨 **Priority:** {priority}")

# === Main UI ===
st.title("🎯 AI Marketing Decision Engine")
st.markdown("Personalized campaign recommendations powered by K-Means clustering on RFM features.")
st.markdown(f"**Database:** {len(decisions):,} customers segmented into 4 groups.")

st.divider()

# === Two Tabs ===
tab1, tab2 = st.tabs(["🔍 Look Up Existing Customer", "🎯 Score New Customer"])

# --- Tab 1: Existing Customer Lookup ---
with tab1:
    st.subheader("Look Up a Customer")
    available_ids = sorted(decisions['Customer ID'].astype(int).unique())

    customer_id = st.selectbox(
        "Choose or search a Customer ID:",
        options=available_ids,
        index=0,
        help="Type to search.",
    )

    if st.button("Get Recommendation", type="primary", key="btn_lookup"):
        row = decisions[decisions['Customer ID'] == customer_id].iloc[0]
        show_recommendation(
            customer_id = row['Customer ID'],
            recency     = row['Recency'],
            frequency   = row['Frequency'],
            monetary    = row['Monetary'],
            segment     = row['Segment'],
            offer       = row['Offer'],
            priority    = row['Priority'],
            best_day    = row['BestDay'],
            best_hour   = row['BestHour'],
        )

# --- Tab 2: Score New Hypothetical Customer ---
with tab2:
    st.subheader("Score a Hypothetical New Customer")
    st.caption("Enter R/F/M values and the trained K-Means model will predict the segment in real time.")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.slider(
            "📅 Recency (days since last purchase)",
            min_value=0, max_value=400, value=45, step=1
        )
    with col2:
        frequency = st.slider(
            "🔁 Frequency (number of orders)",
            min_value=1, max_value=50, value=3, step=1
        )
    with col3:
        monetary = st.number_input(
            "💰 Monetary ($ total spend)",
            min_value=10.0, max_value=50000.0, value=1500.0, step=100.0
        )

    if st.button("Predict Segment", type="primary", key="btn_predict"):
        segment = predict_segment(recency, frequency, monetary)
        rules = DECISION_RULES[segment]
        best_day, best_hour = get_segment_send_time(segment)

        show_recommendation(
            customer_id = None,
            recency     = recency,
            frequency   = frequency,
            monetary    = monetary,
            segment     = segment,
            offer       = rules['offer'],
            priority    = rules['priority'],
            best_day    = best_day,
            best_hour   = best_hour,
        )
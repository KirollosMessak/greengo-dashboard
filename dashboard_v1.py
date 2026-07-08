"""
GreenGo Dashboard - Streamlit frontend for the GreenGo Store n8n backend.

Talks to two n8n webhooks (see "GreenGo Dashboard API (TEST)" workflow):
    GET  /webhook/greengo-dashboard-data  -> { sales_orders, order_lines, kpis }
    POST /webhook/greengo-chat            -> { reply }   (body: { message, session_id })

Setup:
    pip install streamlit pandas requests python-dotenv plotly

Optional .env overrides (defaults point at the TEST workflow):
    N8N_DASHBOARD_WEBHOOK_URL=https://askorg2026.app.n8n.cloud/webhook/greengo-dashboard-data
    N8N_CHAT_WEBHOOK_URL=https://askorg2026.app.n8n.cloud/webhook/greengo-chat

Run:
    streamlit run main.py

Note: the n8n workflow must be Active (or you're using the /webhook-test/ URLs
with "Listen for test event" clicked) for these calls to succeed.

Charts use Plotly (not matplotlib) - it renders interactive, hover-friendly
charts natively in Streamlit and is easier to theme with a custom color
palette, which matplotlib's static images don't do as cleanly.
"""

import os
import uuid

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

GET_DATA_URL = os.environ.get(
    "N8N_DASHBOARD_WEBHOOK_URL",
    "https://askorg2026.app.n8n.cloud/webhook/greengo-dashboard-data",
)
CHAT_URL = os.environ.get(
    "N8N_CHAT_WEBHOOK_URL",
    "https://askorg2026.app.n8n.cloud/webhook/greengo-chat",
)

# Fruity brand palette - pulled from produce colors (tomato, citrus, banana,
# grape, leafy green) rather than a single flat green, to match GreenGo's
# fruits & vegetables branding.
FRUIT_COLORS = ["#4c9a2a", "#f4a261", "#e63946", "#ffd166", "#8e7cc3", "#2a9d8f"]

st.set_page_config(page_title="GreenGo Dashboard", page_icon="🍉", layout="wide")

# GreenGo brand styling: warm, fruity gradient background (cream -> peach ->
# leafy green) instead of flat white, echoing the fresh-produce feel of
# greengo.it.com. Streamlit defaults to a dark theme, so we have to
# explicitly force light backgrounds AND dark text on every container -
# setting only .stApp's background while text stays theme-default (light)
# is what made things unreadable before.
st.markdown(
    """
    <style>
    :root, .stApp {
        color-scheme: light;
    }
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {
        background-image:
            url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPScyNDAnIGhlaWdodD0nMjQwJz4KICA8dGV4dCB4PSc1JyB5PSc0NScgZm9udC1zaXplPSczOCcgb3BhY2l0eT0nMC4xNic+JiMxMjc4MTc7PC90ZXh0PgogIDx0ZXh0IHg9JzEzMCcgeT0nOTUnIGZvbnQtc2l6ZT0nMzAnIG9wYWNpdHk9JzAuMTQnPiYjMTI3ODE4OzwvdGV4dD4KICA8dGV4dCB4PSc0MCcgeT0nMTUwJyBmb250LXNpemU9JzM0JyBvcGFjaXR5PScwLjE1Jz4mIzEyNzgxNTs8L3RleHQ+CiAgPHRleHQgeD0nMTYwJyB5PScyMDAnIGZvbnQtc2l6ZT0nMzAnIG9wYWNpdHk9JzAuMTQnPiYjMTI3ODExOzwvdGV4dD4KICA8dGV4dCB4PSc5NScgeT0nMzAnIGZvbnQtc2l6ZT0nMjYnIG9wYWNpdHk9JzAuMTMnPiYjMTI5NDk2OzwvdGV4dD4KICA8dGV4dCB4PScxMCcgeT0nMjE1JyBmb250LXNpemU9JzI4JyBvcGFjaXR5PScwLjE0Jz4mIzEyNzgxOTs8L3RleHQ+CiAgPHRleHQgeD0nMTkwJyB5PSc0MCcgZm9udC1zaXplPScyNCcgb3BhY2l0eT0nMC4xMyc+JiMxMjc4MjY7PC90ZXh0Pgo8L3N2Zz4K"),
            linear-gradient(135deg, #fffdf5 0%, #fff3e0 30%, #f4f8e8 65%, #eaf5e0 100%);
        background-repeat: repeat, no-repeat;
        background-size: 220px 220px, cover;
        background-attachment: fixed, fixed;
    }
    [data-testid="stSidebar"] {
        background-color: #f4f8f0;
    }
    [data-testid="stAppViewContainer"] * {
        color: #23391f;
    }
    h1, h2, h3 {
        color: #2f5233 !important;
    }
    [data-testid="stCaptionContainer"], .stCaption, small {
        color: #5a7a52 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffffcc;
        border: 1px solid #f0e0c0;
        border-left: 5px solid #f4a261;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 2px 6px rgba(47, 82, 51, 0.08);
    }
    div[data-testid="stMetric"]:nth-of-type(2n) {
        border-left-color: #4c9a2a;
    }
    div[data-testid="stMetric"]:nth-of-type(3n) {
        border-left-color: #e63946;
    }
    div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
        color: #23391f !important;
    }
    div[data-testid="stTextInput"] input, div[data-testid="stChatInput"] textarea {
        background-color: #ffffff;
        color: #23391f;
        border: 1px solid #bcd6b0;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #4c9a2a 0%, #2f5233 100%);
        color: #ffffff !important;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.6em 1.2em;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #f4a261 0%, #e63946 100%);
        color: #ffffff !important;
    }
    div.stButton > button p {
        color: #ffffff !important;
    }
    [data-testid="stChatMessage"] {
        background-color: #ffffffcc;
        border-radius: 10px;
    }
    div[data-testid="stExpander"] {
        background-color: #fffdf5cc;
        border: 1px solid #e3ede0;
        border-radius: 8px;
    }
    hr {
        border-color: #e0d0a8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig):
    """Apply the fruity theme with a near-opaque card background so axis labels
    and gridlines stay readable regardless of the page background behind them."""
    fig.update_layout(
        plot_bgcolor="rgba(255,255,255,0.88)",
        paper_bgcolor="rgba(255,255,255,0.72)",
        font=dict(color="#1b2e17", size=13),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#1b2e17")),
    )
    fig.update_xaxes(gridcolor="#cfe0c5", tickfont=dict(color="#1b2e17"), title_font=dict(color="#1b2e17"), linecolor="#8fae82")
    fig.update_yaxes(gridcolor="#cfe0c5", tickfont=dict(color="#1b2e17"), title_font=dict(color="#1b2e17"), linecolor="#8fae82")
    return fig


st.title("🍉 GreenGo Dashboard")
st.caption("Sales, products & KPIs for GreenGo Store - Alexandria")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dashboard_data" not in st.session_state:
    st.session_state.dashboard_data = None


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# ---- Load data button ----
col_btn, _ = st.columns([1, 4])
with col_btn:
    if st.button("Load / Refresh Data"):
        with st.spinner("Fetching latest data from Google Sheets via n8n..."):
            try:
                resp = requests.get(GET_DATA_URL, timeout=60)
                resp.raise_for_status()
                st.session_state.dashboard_data = resp.json()
            except Exception as e:
                st.error(f"Couldn't fetch dashboard data: {e}")

data = st.session_state.dashboard_data

if data:
    sales_orders = pd.DataFrame(data.get("sales_orders", []))
    order_lines = pd.DataFrame(data.get("order_lines", []))
    kpis = pd.DataFrame(data.get("kpis", []))

    # ---- KPI cards ----
    if not kpis.empty and "اسم المؤشر" in kpis.columns:
        st.subheader("Key Performance Indicators")
        cols = st.columns(min(len(kpis), 4) or 1)
        for i, (_, row) in enumerate(kpis.iterrows()):
            with cols[i % len(cols)]:
                name = row.get("اسم المؤشر", "KPI")
                current = row.get("القيمة الحالية", "N/A")
                target = row.get("الهدف الشهري", None)
                delta = None
                try:
                    if target not in (None, ""):
                        delta = f"{float(current) - float(target):+.0f} vs target"
                except (TypeError, ValueError):
                    delta = None
                st.metric(name, current, delta)

    # ---- Revenue over time (from order lines) ----
    if not order_lines.empty and {"تاريخ", "صافي إيراد السطر (جنيه)"} <= set(order_lines.columns):
        st.subheader("Revenue Over Time")
        ol = order_lines.copy()
        ol["صافي إيراد السطر (جنيه)"] = to_numeric(ol["صافي إيراد السطر (جنيه)"])
        revenue_by_date = (
            ol.groupby("تاريخ")["صافي إيراد السطر (جنيه)"].sum().sort_index().reset_index()
        )
        fig = px.area(
            revenue_by_date,
            x="تاريخ",
            y="صافي إيراد السطر (جنيه)",
            markers=True,
            color_discrete_sequence=["#4c9a2a"],
        )
        fig.update_traces(fill="tozeroy", fillcolor="rgba(76, 154, 42, 0.18)")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    # ---- Top products ----
    if not order_lines.empty and "كود المنتج" in order_lines.columns:
        st.subheader("Top Products by Revenue")
        ol = order_lines.copy()
        ol["صافي إيراد السطر (جنيه)"] = to_numeric(ol.get("صافي إيراد السطر (جنيه)", 0))
        top_products = (
            ol.groupby("كود المنتج")["صافي إيراد السطر (جنيه)"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig = px.bar(
            top_products,
            x="صافي إيراد السطر (جنيه)",
            y="كود المنتج",
            orientation="h",
            color="كود المنتج",
            color_discrete_sequence=FRUIT_COLORS,
            text="صافي إيراد السطر (جنيه)",
        )
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    # ---- Sales by channel ----
    if not sales_orders.empty and "قناة البيع" in sales_orders.columns:
        st.subheader("Sales by Channel")
        so = sales_orders.copy()
        so["صافي إيراد الأوردر"] = to_numeric(so.get("صافي إيراد الأوردر", 0))
        channel_rev = (
            so.groupby("قناة البيع")["صافي إيراد الأوردر"].sum().sort_values(ascending=False).reset_index()
        )
        fig = px.pie(
            channel_rev,
            names="قناة البيع",
            values="صافي إيراد الأوردر",
            hole=0.45,
            color_discrete_sequence=FRUIT_COLORS,
        )
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with st.expander("Raw data"):
        st.write("Sales Orders", sales_orders)
        st.write("Order Lines", order_lines)
        st.write("KPIs", kpis)
else:
    st.info("Click **Load / Refresh Data** above to fetch the latest numbers.")

st.divider()

# ---- Chat bar ----
st.subheader("Ask the GreenGo Assistant")
st.caption("Ask about sales, products, or KPIs - answers come from the same live sheets above.")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_msg = st.chat_input("e.g. What was our revenue this month?")
if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    CHAT_URL,
                    json={"message": user_msg, "session_id": st.session_state.session_id},
                    timeout=60,
                )
                resp.raise_for_status()
                reply = resp.json().get("reply", "No reply received.")
            except Exception as e:
                reply = f"Error contacting assistant: {e}"
            st.write(reply)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})

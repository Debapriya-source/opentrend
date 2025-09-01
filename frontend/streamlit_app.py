import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Minimal Streamlit frontend for OpenTrend AI (MVP)
BASE_URL = st.secrets.get("backend_url", "http://localhost:8000")

st.set_page_config(page_title="OpenTrend AI", layout="wide")

st.sidebar.title("OpenTrend AI")
page = st.sidebar.radio("Go to", ["Home", "Trends", "Data", "Analysis"])

if page == "Home":
    st.title("OpenTrend AI — Dashboard (MVP)")
    st.markdown(
        "This is a lightweight Streamlit dashboard that talks to the FastAPI backend."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Service Health")
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=3)
            st.json(r.json())
        except Exception as e:
            st.error(f"Backend unreachable: {e}")

    with col2:
        st.subheader("Quick Sample Chart")
        # sample synthetic data to show layout
        df = pd.DataFrame(
            {
                "date": pd.date_range(end=pd.Timestamp.today(), periods=30),
                "value": (pd.np.random.randn(30).cumsum() + 100),
            }
        )
        fig = px.line(df, x="date", y="value", title="Sample price series")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Trends":
    st.header("Identified Trends")
    st.write(
        "Fetches trend summaries from the backend endpoint /analysis/trends (if available)."
    )
    try:
        resp = requests.get(f"{BASE_URL}/analysis/trends", timeout=5)
        if resp.status_code == 200:
            trends = resp.json()
            st.json(trends)
            # If backend returns time-series, render a chart for the first trend
            if (
                isinstance(trends, list)
                and len(trends) > 0
                and "timeseries" in trends[0]
            ):
                ts = pd.DataFrame(trends[0]["timeseries"])
                st.line_chart(ts.set_index("date")["value"])
        else:
            st.warning(
                "No trends endpoint available on backend (HTTP %s)" % resp.status_code
            )
    except Exception as e:
        st.error(f"Failed to fetch trends: {e}")

elif page == "Data":
    st.header("Data Explorer")
    st.write("Upload a CSV or load sample market data for quick inspection.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df)
        st.download_button("Download (CSV)", df.to_csv(index=False), mime="text/csv")

    st.markdown("---")
    st.subheader("Sample OHLC data")
    # small synthetic ohlc
    dates = pd.date_range(end=pd.Timestamp.today(), periods=60)
    import numpy as np

    price = 100 + np.cumsum(np.random.randn(len(dates)))
    ohlc = pd.DataFrame(
        {
            "date": dates,
            "open": price + np.random.rand(len(dates)),
            "close": price,
            "high": price + 2 * np.random.rand(len(dates)),
            "low": price - 2 * np.random.rand(len(dates)),
        }
    )
    st.line_chart(ohlc.set_index("date")["close"])

elif page == "Analysis":
    st.header("Analysis & Forecast")
    st.write("Request model forecasts or run simple, client-side analyses.")
    symbol = st.text_input("Symbol", "SAMPLE")
    days = st.slider("Forecast horizon (days)", 1, 90, 14)
    if st.button("Request forecast"):
        try:
            resp = requests.get(
                f"{BASE_URL}/prediction/forecast",
                params={"symbol": symbol, "days": days},
                timeout=10,
            )
            if resp.status_code == 200:
                forecast = resp.json()
                st.json(forecast)
                # try to plot if timeseries present
                if "timeseries" in forecast:
                    df = pd.DataFrame(forecast["timeseries"]).set_index("date")
                    st.line_chart(df["value"])
            else:
                st.error(f"Forecast request failed (HTTP {resp.status_code})")
        except Exception as e:
            st.error(f"Forecast request error: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("OpenTrend AI — MVP frontend")

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px
# import numpy as np

# # Minimal Streamlit frontend for OpenTrend AI (MVP)
# BASE_URL = st.secrets.get("backend_url", "http://localhost:8000")

# if "access_token" not in st.session_state:
#     st.session_state.access_token = None


# def auth_headers():
#     return (
#         {"Authorization": f"Bearer {st.session_state.access_token}"}
#         if st.session_state.access_token
#         else {}
#     )


# with st.sidebar.expander("Sign in", expanded=(st.session_state.access_token is None)):
#     username = st.text_input("Username", key="auth_user")
#     password = st.text_input("Password", type="password", key="auth_pass")
#     if st.button("Login"):
#         try:
#             r = requests.post(
#                 f"{BASE_URL}/api/v1/auth/login",
#                 data={"username": username, "password": password},  # OAuth2 form
#                 timeout=8,
#             )
#             if r.status_code == 200:
#                 st.session_state.access_token = r.json()["access_token"]
#                 st.success("Logged in")
#             else:
#                 st.error(f"Login failed (HTTP {r.status_code})")
#         except Exception as e:
#             st.error(f"Login error: {e}")


# st.set_page_config(page_title="OpenTrend AI", layout="wide")

# st.sidebar.title("OpenTrend AI")
# page = st.sidebar.radio("Go to", ["Home", "Trends", "Data", "Analysis"])

# if page == "Home":
#     st.title("OpenTrend AI — Dashboard (MVP)")
#     st.markdown(
#         "This is a lightweight Streamlit dashboard that talks to the FastAPI backend."
#     )

#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("Service Health")
#         try:
#             r = requests.get(f"{BASE_URL}/health", timeout=3)
#             st.json(r.json())
#         except Exception as e:
#             st.error(f"Backend unreachable: {e}")

#     with col2:
#         st.subheader("Quick Sample Chart")
#         # sample synthetic data to show layout
#         df = pd.DataFrame(
#             {
#                 "date": pd.date_range(end=pd.Timestamp.today(), periods=30),
#                 "value": (np.random.randn(30).cumsum() + 100),
#             }
#         )
#         fig = px.line(df, x="date", y="value", title="Sample price series")
#         st.plotly_chart(fig, use_container_width=True)

# elif page == "Trends":
#     st.header("Identified Trends")
#     st.write(
#         "Fetches trend summaries from the backend endpoint /analysis/trends (if available)."
#     )
#     try:
#         resp = requests.get(
#             f"{BASE_URL}/api/v1/analysis/trends", headers=auth_headers(), timeout=8
#         )
#         if resp.status_code == 200:
#             payload = resp.json()
#             st.json(payload)
#             data = payload.get("data", [])
#             if data and isinstance(data, list) and "timeseries" in data[0]:
#                 ts = pd.DataFrame(data[0]["timeseries"])
#                 if {"date", "value"}.issubset(ts.columns):
#                     st.line_chart(ts.set_index("date")["value"])
#             # If backend returns time-series, render a chart for the first trend
#             # if (
#             #     isinstance(trends, list)
#             #     and len(trends) > 0
#             #     and "timeseries" in trends[0]
#             # ):
#             #     ts = pd.DataFrame(trends[0]["timeseries"])
#             #     st.line_chart(ts.set_index("date")["value"])
#         else:
#             st.warning(
#                 "No trends endpoint available on backend (HTTP %s)" % resp.status_code
#             )
#     except Exception as e:
#         st.error(f"Failed to fetch trends: {e}")

# elif page == "Data":
#     st.header("Data Explorer")
#     st.write("Upload a CSV or load sample market data for quick inspection.")
#     uploaded = st.file_uploader("Upload CSV", type=["csv"])
#     if uploaded is not None:
#         df = pd.read_csv(uploaded)
#         st.dataframe(df)
#         st.download_button("Download (CSV)", df.to_csv(index=False), mime="text/csv")

#     st.markdown("---")
#     st.subheader("Sample OHLC data")
#     # small synthetic ohlc
#     dates = pd.date_range(end=pd.Timestamp.today(), periods=60)
#     import numpy as np

#     price = 100 + np.cumsum(np.random.randn(len(dates)))
#     ohlc = pd.DataFrame(
#         {
#             "date": dates,
#             "open": price + np.random.rand(len(dates)),
#             "close": price,
#             "high": price + 2 * np.random.rand(len(dates)),
#             "low": price - 2 * np.random.rand(len(dates)),
#         }
#     )
#     st.line_chart(ohlc.set_index("date")["close"])

# elif page == "Analysis":
#     st.header("Analysis & Forecast")
#     st.write("Request model forecasts or run simple, client-side analyses.")
#     symbol = st.text_input("Symbol", "SAMPLE")
#     days = st.slider("Forecast horizon (days)", 1, 90, 14)
#     if st.button("Request forecast"):
#         try:
#             resp = requests.post(
#                 f"{BASE_URL}/api/v1/analysis/predictions/generate",
#                 params={
#                     "symbol": symbol,
#                     "prediction_type": "price",
#                     "horizon_days": days,
#                 },
#                 headers=auth_headers(),
#                 timeout=12,
#             )
#             if resp.status_code == 200:
#                 forecast = resp.json()
#                 st.json(forecast)
#                 # try to plot if timeseries present
#                 if "timeseries" in forecast:
#                     df = pd.DataFrame(forecast["timeseries"]).set_index("date")
#                     st.line_chart(df["value"])
#             else:
#                 st.error(f"Forecast request failed (HTTP {resp.status_code})")
#         except Exception as e:
#             st.error(f"Forecast request error: {e}")

# # Footer
# st.sidebar.markdown("---")
# st.sidebar.caption("OpenTrend AI — MVP frontend")


import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---------- App setup ----------
st.set_page_config(page_title="OpenTrend AI", layout="wide")
BASE_URL = st.secrets.get("backend_url", "http://localhost:8000")

# ---------- Auth state & helpers ----------
if "access_token" not in st.session_state:
    st.session_state.access_token = None


def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.access_token}"} if st.session_state.access_token else {}


def show_auth_error(resp, default_msg):
    detail = ""
    try:
        detail = resp.json().get("detail", "")
    except Exception:
        pass
    st.warning(f"{default_msg} (HTTP {resp.status_code}). {detail or 'Please sign in via the sidebar.'}")


# ---------- Sidebar / Auth UI ----------
st.sidebar.title("OpenTrend AI")

# --- Login form ---
with st.sidebar.expander("Login", expanded=(st.session_state.access_token is None)):
    login_user = st.text_input("Username", key="login_user")
    login_pass = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login"):
        try:
            r = requests.post(
                f"{BASE_URL}/api/v1/auth/login",
                data={"username": login_user, "password": login_pass},  # OAuth2 form
                timeout=8,
            )
            if r.status_code == 200:
                st.session_state.access_token = r.json()["access_token"]
                st.success("Logged in")
            else:
                show_auth_error(r, "Login failed")
        except Exception as e:
            st.error(f"Login error: {e}")

# --- Sign up form ---
with st.sidebar.expander("Sign up"):
    signup_user = st.text_input("Username", key="signup_user")
    signup_email = st.text_input("Email", key="signup_email")
    signup_pass = st.text_input("Password", type="password", key="signup_pass")
    confirm_pass = st.text_input("Confirm Password", type="password", key="signup_confirm")
    if st.button("Register"):
        if signup_pass != confirm_pass:
            st.error("Passwords do not match")
        else:
            try:
                payload = {
                    "username": signup_user,
                    "email": signup_email or f"{signup_user}@example.com",
                    "password": signup_pass,
                }
                r = requests.post(f"{BASE_URL}/api/v1/auth/register", json=payload, timeout=8)
                if r.status_code in (200, 201):
                    st.success("User registered. Please log in.")
                else:
                    show_auth_error(r, "Sign up failed")
            except Exception as e:
                st.error(f"Sign up error: {e}")

# --- Show current user + logout ---
if st.session_state.access_token:
    try:
        me = requests.get(f"{BASE_URL}/api/v1/auth/me", headers=auth_headers(), timeout=5)
        if me.status_code == 200:
            uname = me.json().get("username", "user")
            st.sidebar.success(f"Logged in as {uname}")
        else:
            st.sidebar.warning("Session invalid or expired. Please login again.")
            st.session_state.access_token = None
    except Exception:
        pass

    if st.sidebar.button("Logout"):
        st.session_state.access_token = None
        st.experimental_rerun()

# ---------- Page router ----------
page = st.sidebar.radio("Go to", ["Home", "Trends", "Data", "Analysis"])

# ---------- Home ----------
if page == "Home":
    st.title("OpenTrend AI — Dashboard (MVP)")
    st.markdown("This is a lightweight Streamlit dashboard that talks to the FastAPI backend.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Service Health")
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                health_data = r.json()

                # Display health status with better formatting
                status = health_data.get("status", "unknown")
                if status == "healthy":
                    st.success("🟢 **Status**: Healthy")
                else:
                    st.warning(f"🟡 **Status**: {status.title()}")

                # Show database status
                db_status = health_data.get("database", "unknown")
                if db_status == "connected":
                    st.success("🗄️ **Database**: Connected")
                else:
                    st.error(f"🗄️ **Database**: {db_status}")

                # Show timestamp
                timestamp = health_data.get("timestamp", "")
                if timestamp:
                    st.info(f"🕒 **Last Check**: {timestamp}")

                # Show detailed data in expander
                with st.expander("View detailed health data"):
                    st.json(health_data)
            else:
                st.error(f"Health check failed (HTTP {r.status_code})")
        except Exception as e:
            st.error(f"🔴 Backend unreachable: {e}")

    with col2:
        st.subheader("Quick Sample Chart")
        df = pd.DataFrame(
            {
                "date": pd.date_range(end=pd.Timestamp.today(), periods=30),
                "value": (np.random.randn(30).cumsum() + 100),
            }
        )
        fig = px.line(df, x="date", y="value", title="Sample price series")
        st.plotly_chart(fig, use_container_width=True)

# ---------- Trends ----------
elif page == "Trends":
    st.header("Identified Trends")

    # Add controls for trend analysis
    col1, col2 = st.columns(2)
    with col1:
        trend_symbol = st.text_input("Symbol for Trend Analysis", "AAPL", key="trend_symbol")
    with col2:
        timeframe = st.selectbox("Timeframe", ["short", "medium", "long"], index=1)

    if st.button("Analyze Trends"):
        with st.spinner(f"Analyzing trends for {trend_symbol}..."):
            try:
                resp = requests.post(
                    f"{BASE_URL}/api/v1/analysis/trends/analyze",
                    params={"symbol": trend_symbol, "timeframe": timeframe},
                    headers=auth_headers(),
                    timeout=15,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    analysis = result.get("analysis", {})

                    # Display trend analysis results
                    st.success("✅ Trend analysis completed!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Trend Type", analysis.get("trend_type", "Unknown"))
                    with col2:
                        st.metric("Confidence Score", f"{analysis.get('confidence_score', 0):.2%}")
                    with col3:
                        st.metric("Timeframe", timeframe.title())

                    # Show description
                    if analysis.get("description"):
                        st.info(f"📊 **Analysis**: {analysis['description']}")

                    # Show indicators
                    if analysis.get("indicators"):
                        st.subheader("Technical Indicators")
                        indicators = analysis["indicators"]
                        if isinstance(indicators, dict):
                            for key, value in indicators.items():
                                st.write(f"**{key.replace('_', ' ').title()}**: {value}")
                        else:
                            st.write(indicators)

                    # Show detailed data in expander
                    with st.expander("View detailed analysis data"):
                        st.json(result)

                elif resp.status_code in (401, 403):
                    show_auth_error(resp, "Unauthorized")
                else:
                    detail = ""
                    try:
                        detail = resp.json().get("detail", "")
                    except Exception:
                        pass
                    st.error(f"Trend analysis failed (HTTP {resp.status_code}). {detail}")
            except Exception as e:
                st.error(f"Failed to analyze trends: {e}")

    st.markdown("---")

    # Show existing trend analyses
    st.subheader("Recent Trend Analyses")
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/analysis/trends", headers=auth_headers(), timeout=8)
        if resp.status_code == 200:
            payload = resp.json()
            analyses = payload.get("data", [])

            if analyses:
                # Create a DataFrame for better display
                df_data = []
                for analysis in analyses[:10]:  # Show last 10
                    df_data.append(
                        {
                            "Symbol": analysis.get("symbol", ""),
                            "Trend Type": analysis.get("trend_type", ""),
                            "Confidence": f"{analysis.get('confidence_score', 0):.1%}",
                            "Timeframe": analysis.get("timeframe", ""),
                            "Date": analysis.get("analysis_date", "")[:10] if analysis.get("analysis_date") else "",
                        }
                    )

                if df_data:
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No trend analyses found. Try analyzing some symbols above!")
            else:
                st.info("No trend analyses found. Try analyzing some symbols above!")

        elif resp.status_code in (401, 403):
            show_auth_error(resp, "Unauthorized")
        else:
            st.warning(f"Could not fetch trend analyses (HTTP {resp.status_code}).")
    except Exception as e:
        st.error(f"Failed to fetch trend analyses: {e}")

# ---------- Data ----------
elif page == "Data":
    st.header("Data Explorer")

    # File upload section
    st.subheader("Upload CSV")
    uploaded = st.file_uploader("Drag and drop file here", type=["csv"], help="Limit 200MB per file • CSV")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")

            # Show basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Size", f"{uploaded.size / 1024:.1f} KB")

            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(100), use_container_width=True)

            # Download processed data
            st.download_button(
                "📥 Download (CSV)", df.to_csv(index=False), file_name=f"processed_{uploaded.name}", mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.markdown("---")

    # Market data section
    st.subheader("Market Data Viewer")

    col1, col2 = st.columns(2)
    with col1:
        data_symbol = st.text_input("Symbol", "AAPL", key="data_symbol")
    with col2:
        days_back = st.slider("Days of data", 7, 365, 30)

    if st.button("📊 Load Market Data"):
        with st.spinner(f"Loading market data for {data_symbol}..."):
            try:
                # First ensure data exists by triggering collection if needed
                prediction_resp = requests.post(
                    f"{BASE_URL}/api/v1/analysis/predictions/generate",
                    params={
                        "symbol": data_symbol,
                        "prediction_type": "price",
                        "horizon_days": 1,  # Minimal prediction just to trigger data collection
                    },
                    headers=auth_headers(),
                    timeout=30,
                )

                if prediction_resp.status_code == 200:
                    result = prediction_resp.json()
                    if result.get("data_ingested", False):
                        st.success(f"✅ Market data collected for {data_symbol}")
                    else:
                        st.info(f"📈 Using existing data for {data_symbol}")

                    # Now fetch the real market data from the database
                    market_resp = requests.get(
                        f"{BASE_URL}/api/v1/analysis/market-data/{data_symbol}",
                        params={"days": days_back},
                        headers=auth_headers(),
                        timeout=15,
                    )

                    if market_resp.status_code == 200:
                        market_data = market_resp.json()

                        if market_data["data_points"] > 0:
                            st.subheader(f"Real Market Data for {data_symbol}")

                            # Convert to DataFrame
                            df = pd.DataFrame(market_data["data"])
                            df["date"] = pd.to_datetime(df["date"])
                            df = df.sort_values("date")

                            # Display metrics using real data
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                current_price = df["close"].iloc[-1]
                                st.metric("Current Price", f"${current_price:.2f}")
                            with col2:
                                if len(df) > 1:
                                    change = df["close"].iloc[-1] - df["close"].iloc[-2]
                                    change_pct = (change / df["close"].iloc[-2]) * 100
                                    st.metric("Daily Change", f"${change:.2f}", f"{change_pct:+.2f}%")
                                else:
                                    st.metric("Daily Change", "N/A")
                            with col3:
                                current_volume = df["volume"].iloc[-1]
                                st.metric("Volume", f"{current_volume:,}")
                            with col4:
                                st.metric("Data Points", len(df))

                            # Price chart with real data
                            fig = px.line(df, x="date", y="close", title=f"{data_symbol} Price Chart (Real Data)")
                            fig.update_layout(xaxis_title="Date", yaxis_title="Price ($)")
                            st.plotly_chart(fig, use_container_width=True)

                            # Volume chart with real data
                            fig_vol = px.bar(df, x="date", y="volume", title=f"{data_symbol} Volume (Real Data)")
                            fig_vol.update_layout(xaxis_title="Date", yaxis_title="Volume")
                            st.plotly_chart(fig_vol, use_container_width=True)

                            # OHLC chart
                            if all(col in df.columns for col in ["open", "high", "low", "close"]):
                                fig_ohlc = go.Figure(
                                    data=go.Candlestick(
                                        x=df["date"],
                                        open=df["open"],
                                        high=df["high"],
                                        low=df["low"],
                                        close=df["close"],
                                        name=f"{data_symbol} OHLC",
                                    )
                                )
                                fig_ohlc.update_layout(
                                    title=f"{data_symbol} OHLC Chart (Real Data)",
                                    xaxis_title="Date",
                                    yaxis_title="Price ($)",
                                )
                                st.plotly_chart(fig_ohlc, use_container_width=True)

                            # Data table with real data
                            with st.expander("📋 View Raw Market Data"):
                                # Format the dataframe for better display
                                display_df = df.copy()
                                display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
                                display_df = display_df.round(2)
                                st.dataframe(display_df, use_container_width=True)

                                # Download button for real data
                                csv = display_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Real Market Data (CSV)",
                                    csv,
                                    file_name=f"{data_symbol}_real_market_data_{days_back}days.csv",
                                    mime="text/csv",
                                )

                            # Show data source info
                            st.info(
                                f"📊 **Data Source**: {df['source'].iloc[0] if 'source' in df.columns else 'yfinance'} | **Date Range**: {market_data['date_range']['start']} to {market_data['date_range']['end']}"
                            )

                        else:
                            st.warning(
                                f"No market data available for {data_symbol} in the last {days_back} days. The data might not have been collected yet."
                            )

                    else:
                        st.error(f"Failed to fetch market data from database (HTTP {market_resp.status_code})")

                else:
                    detail = ""
                    try:
                        detail = prediction_resp.json().get("detail", "")
                    except Exception:
                        pass
                    st.error(
                        f"Failed to ensure market data availability (HTTP {prediction_resp.status_code}). {detail}"
                    )

            except Exception as e:
                st.error(f"Error loading market data: {e}")

    st.markdown("---")
    st.subheader("Sample OHLC Data")
    st.info("💡 **Tip**: Use the Market Data Viewer above to load real market data for any stock symbol!")

    # Keep the original sample data as a fallback
    dates = pd.date_range(end=pd.Timestamp.today(), periods=60)
    price = 100 + np.cumsum(np.random.randn(len(dates)))
    sample_ohlc = pd.DataFrame(
        {
            "date": dates,
            "close": price,
        }
    )
    st.line_chart(sample_ohlc.set_index("date")["close"])

# ---------- Analysis / Forecast ----------
elif page == "Analysis":
    st.header("Analysis & Forecast")
    symbol = st.text_input("Symbol", "AAPL")
    days = st.slider("Forecast horizon (days)", 1, 90, 14)

    if st.button("Request forecast"):
        # Show loading spinner
        with st.spinner(
            f"Generating forecast for {symbol}... This may take a moment if we need to collect market data."
        ):
            try:
                resp = requests.post(
                    f"{BASE_URL}/api/v1/analysis/predictions/generate",
                    params={
                        "symbol": symbol,
                        "prediction_type": "price",
                        "horizon_days": days,
                    },
                    headers=auth_headers(),
                    timeout=30,  # Increased timeout for data ingestion
                )
                if resp.status_code == 200:
                    forecast = resp.json()

                    # Show success message with data ingestion info
                    if forecast.get("data_ingested", False):
                        st.success(f"✅ Successfully collected market data for {symbol} and generated forecast!")
                    else:
                        st.success(f"✅ Forecast generated successfully for {symbol}!")

                    # Display the prediction results
                    prediction = forecast.get("prediction", {})
                    if prediction:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Predicted Value", f"${prediction.get('predicted_value', 0):.2f}")
                        with col2:
                            st.metric("Lower Bound", f"${prediction.get('confidence_interval_lower', 0):.2f}")
                        with col3:
                            st.metric("Upper Bound", f"${prediction.get('confidence_interval_upper', 0):.2f}")

                        st.info(f"Model used: {prediction.get('model_used', 'Unknown')}")

                    # Show full JSON for debugging
                    with st.expander("View detailed forecast data"):
                        st.json(forecast)

                    # Plot if timeseries data is available
                    if "timeseries" in forecast:
                        df = pd.DataFrame(forecast["timeseries"]).set_index("date")
                        if "value" in df.columns:
                            st.line_chart(df["value"])

                elif resp.status_code in (401, 403):
                    show_auth_error(resp, "Unauthorized")
                else:
                    detail = ""
                    try:
                        detail = resp.json().get("detail", "")
                    except Exception:
                        pass
                    st.error(f"Forecast request failed (HTTP {resp.status_code}). {detail}")

            except requests.exceptions.Timeout:
                st.error("Request timed out. The server may be collecting market data. Please try again in a moment.")
            except Exception as e:
                st.error(f"Forecast request error: {e}")

# ---------- Footer ----------
st.sidebar.markdown("---")
st.sidebar.caption("OpenTrend AI — MVP frontend")

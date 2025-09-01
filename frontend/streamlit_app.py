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
import numpy as np

# ---------- App setup ----------
st.set_page_config(page_title="OpenTrend AI", layout="wide")
BASE_URL = st.secrets.get("backend_url", "http://localhost:8000")

# ---------- Auth state & helpers ----------
if "access_token" not in st.session_state:
    st.session_state.access_token = None


def auth_headers():
    return (
        {"Authorization": f"Bearer {st.session_state.access_token}"}
        if st.session_state.access_token
        else {}
    )


def show_auth_error(resp, default_msg):
    detail = ""
    try:
        detail = resp.json().get("detail", "")
    except Exception:
        pass
    st.warning(
        f"{default_msg} (HTTP {resp.status_code}). {detail or 'Please sign in via the sidebar.'}"
    )


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
    confirm_pass = st.text_input(
        "Confirm Password", type="password", key="signup_confirm"
    )
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
                r = requests.post(
                    f"{BASE_URL}/api/v1/auth/register", json=payload, timeout=8
                )
                if r.status_code in (200, 201):
                    st.success("User registered. Please log in.")
                else:
                    show_auth_error(r, "Sign up failed")
            except Exception as e:
                st.error(f"Sign up error: {e}")

# --- Show current user + logout ---
if st.session_state.access_token:
    try:
        me = requests.get(
            f"{BASE_URL}/api/v1/auth/me", headers=auth_headers(), timeout=5
        )
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
    try:
        resp = requests.get(
            f"{BASE_URL}/api/v1/analysis/trends", headers=auth_headers(), timeout=8
        )
        if resp.status_code == 200:
            payload = resp.json()
            st.json(payload)
            data = payload.get("data", [])
            if data and isinstance(data, list) and "timeseries" in data[0]:
                ts = pd.DataFrame(data[0]["timeseries"])
                if {"date", "value"}.issubset(ts.columns):
                    st.line_chart(ts.set_index("date")["value"])
        elif resp.status_code in (401, 403):
            show_auth_error(resp, "Unauthorized")
        else:
            st.warning(
                f"Unexpected response from trends endpoint (HTTP {resp.status_code})."
            )
    except Exception as e:
        st.error(f"Failed to fetch trends: {e}")

# ---------- Data ----------
elif page == "Data":
    st.header("Data Explorer")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df)
        st.download_button("Download (CSV)", df.to_csv(index=False), mime="text/csv")

    st.markdown("---")
    st.subheader("Sample OHLC data")
    dates = pd.date_range(end=pd.Timestamp.today(), periods=60)
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

# ---------- Analysis / Forecast ----------
elif page == "Analysis":
    st.header("Analysis & Forecast")
    symbol = st.text_input("Symbol", "SAMPLE")
    days = st.slider("Forecast horizon (days)", 1, 90, 14)

    if st.button("Request forecast"):
        try:
            resp = requests.post(
                f"{BASE_URL}/api/v1/analysis/predictions/generate",
                params={
                    "symbol": symbol,
                    "prediction_type": "price",
                    "horizon_days": days,
                },
                headers=auth_headers(),
                timeout=12,
            )
            if resp.status_code == 200:
                forecast = resp.json()
                st.json(forecast)
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

        except Exception as e:
            st.error(f"Forecast request error: {e}")

# ---------- Footer ----------
st.sidebar.markdown("---")
st.sidebar.caption("OpenTrend AI — MVP frontend")

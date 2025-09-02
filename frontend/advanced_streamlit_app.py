"""
Advanced OpenTrend AI Frontend - Complete Integration
Includes Prophet, LSTM, Ensemble forecasting, AI insights, and advanced visualizations
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import json

# Configuration
st.set_page_config(page_title="OpenTrend AI - Advanced Analytics", layout="wide", page_icon="📈")

# Base URL for API
BASE_URL = st.secrets.get("backend_url", "http://localhost:8000")

# Session state for authentication
if "access_token" not in st.session_state:
    st.session_state.access_token = None


def auth_headers():
    """Get authentication headers."""
    return {"Authorization": f"Bearer {st.session_state.access_token}"} if st.session_state.access_token else {}


def show_auth_error(resp, context=""):
    """Show authentication error message."""
    st.error(f"🔐 Authentication required. Please login to access {context}.")
    st.info("Use the login form in the sidebar.")


# Sidebar Authentication
with st.sidebar.expander("🔐 Authentication", expanded=(st.session_state.access_token is None)):
    if st.session_state.access_token:
        st.success("✅ Logged in successfully!")
        if st.button("🚪 Logout"):
            st.session_state.access_token = None
            st.rerun()
    else:
        username = st.text_input("👤 Username", key="auth_user")
        password = st.text_input("🔒 Password", type="password", key="auth_pass")

        if st.button("🔑 Login"):
            try:
                r = requests.post(
                    f"{BASE_URL}/api/v1/auth/login",
                    data={"username": username, "password": password},
                    timeout=8,
                )
                if r.status_code == 200:
                    st.session_state.access_token = r.json()["access_token"]
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error(f"❌ Login failed (HTTP {r.status_code})")
            except Exception as e:
                st.error(f"❌ Login error: {e}")

# Main Application
st.title("📈 OpenTrend AI - Advanced Financial Analytics")
st.markdown("*Professional-grade forecasting, AI insights, and portfolio analysis*")

# Navigation
page = st.sidebar.radio(
    "🧭 Navigation",
    ["🏠 Dashboard", "🔮 Advanced Forecasting", "🧠 AI Insights", "📊 Visualizations", "💼 Portfolio Analysis"],
)

# Dashboard Page
if page == "🏠 Dashboard":
    st.header("🏠 System Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔧 System Health")
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                health_data = r.json()

                status = health_data.get("status", "unknown")
                if status == "healthy":
                    st.success("🟢 **Status**: Healthy")
                else:
                    st.warning(f"🟡 **Status**: {status.title()}")

                db_status = health_data.get("database", "unknown")
                if db_status == "connected":
                    st.success("🗄️ **Database**: Connected")
                else:
                    st.error(f"🗄️ **Database**: {db_status}")

                with st.expander("📋 Detailed Health Data"):
                    st.json(health_data)
            else:
                st.error(f"❌ Health check failed (HTTP {r.status_code})")
        except Exception as e:
            st.error(f"🔴 Backend unreachable: {e}")

    with col2:
        st.subheader("🤖 AI Models Status")
        if st.session_state.access_token:
            try:
                # Check LLM model status
                llm_resp = requests.get(f"{BASE_URL}/api/v1/llm/model-status", headers=auth_headers(), timeout=5)

                if llm_resp.status_code == 200:
                    llm_data = llm_resp.json()

                    if llm_data.get("sentiment_analyzer"):
                        st.success("🧠 **Sentiment AI**: Operational")
                    else:
                        st.warning("🧠 **Sentiment AI**: Limited")

                    device = llm_data.get("model_info", {}).get("device", "CPU")
                    st.info(f"💻 **Device**: {device}")

                # Check forecasting models status
                forecast_resp = requests.get(
                    f"{BASE_URL}/api/v1/forecasting/models/status", headers=auth_headers(), timeout=5
                )

                if forecast_resp.status_code == 200:
                    forecast_data = forecast_resp.json()
                    models = forecast_data.get("forecasting_models", {})

                    for model_name, model_info in models.items():
                        if model_info.get("available"):
                            st.success(f"🔮 **{model_name.title()}**: Available")
                        else:
                            st.error(f"🔮 **{model_name.title()}**: Unavailable")

            except Exception as e:
                st.warning(f"⚠️ Could not check AI models: {e}")
        else:
            st.info("🔐 Login required to check AI model status")

# Advanced Forecasting Page
elif page == "🔮 Advanced Forecasting":
    st.header("🔮 Advanced Forecasting")

    if not st.session_state.access_token:
        show_auth_error(None, "forecasting features")
    else:
        # Symbol and model selection
        col1, col2 = st.columns(2)

        with col1:
            symbol = st.text_input("📈 Stock Symbol", "AAPL", help="Enter a stock symbol (e.g., AAPL, GOOGL, MSFT)")

        with col2:
            model_type = st.selectbox(
                "🤖 Forecasting Model",
                ["Prophet", "LSTM", "Ensemble", "Compare All Models"],
                help="Select the ML model for forecasting",
            )

        # Model-specific parameters
        if model_type == "Prophet":
            max_days = 365
            default_days = 30
        elif model_type == "LSTM":
            max_days = 90
            default_days = 30
        else:
            max_days = 90
            default_days = 30

        horizon_days = st.slider("📅 Forecast Horizon (days)", 1, max_days, default_days)

        if model_type == "LSTM":
            sequence_length = st.slider(
                "🔢 LSTM Sequence Length", 30, 120, 60, help="Number of past days to use for prediction"
            )

        # Generate forecast
        if st.button(f"🚀 Generate {model_type} Forecast", type="primary"):
            with st.spinner(f"Generating {model_type} forecast for {symbol}..."):
                try:
                    # Determine endpoint based on model type
                    if model_type == "Prophet":
                        endpoint = f"{BASE_URL}/api/v1/forecasting/prophet/{symbol}"
                        params = {"horizon_days": horizon_days}
                    elif model_type == "LSTM":
                        endpoint = f"{BASE_URL}/api/v1/forecasting/lstm/{symbol}"
                        params = {"horizon_days": horizon_days, "sequence_length": sequence_length}
                    elif model_type == "Ensemble":
                        endpoint = f"{BASE_URL}/api/v1/forecasting/ensemble/{symbol}"
                        params = {"horizon_days": horizon_days}
                    else:  # Compare All Models
                        endpoint = f"{BASE_URL}/api/v1/forecasting/compare/{symbol}"
                        params = {"horizon_days": horizon_days}

                    resp = requests.post(endpoint, params=params, headers=auth_headers(), timeout=60)

                    if resp.status_code == 200:
                        result = resp.json()

                        if model_type == "Compare All Models":
                            # Handle model comparison
                            st.success("✅ Model comparison completed!")

                            comparisons = result.get("model_comparisons", {})
                            consensus = result.get("consensus_metrics", {})

                            # Consensus metrics
                            if consensus:
                                st.subheader("📊 Consensus Forecast")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Average Prediction", f"${consensus.get('average_prediction', 0):.2f}")
                                with col2:
                                    st.metric("Min Prediction", f"${consensus.get('min_prediction', 0):.2f}")
                                with col3:
                                    st.metric("Max Prediction", f"${consensus.get('max_prediction', 0):.2f}")
                                with col4:
                                    st.metric("Models Agreeing", consensus.get("models_agreeing", 0))

                            # Individual model results
                            st.subheader("🔍 Individual Model Results")
                            for model_name, model_result in comparisons.items():
                                if "error" not in model_result:
                                    with st.expander(f"{model_name.title()} Model Results"):
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric(
                                                "Predicted Price", f"${model_result.get('predicted_price', 0):.2f}"
                                            )
                                        with col2:
                                            price_change = model_result.get("price_change_percent", 0)
                                            st.metric(
                                                "Price Change", f"{price_change:+.2f}%", delta=f"{price_change:.2f}%"
                                            )
                                        with col3:
                                            conf_int = model_result.get("confidence_interval", {})
                                            range_val = conf_int.get("upper", 0) - conf_int.get("lower", 0)
                                            st.metric("Confidence Range", f"${range_val:.2f}")
                                else:
                                    st.error(f"❌ {model_name.title()} model failed: {model_result['error']}")

                        else:
                            # Handle individual model results
                            forecast_data = result.get("forecast", {})
                            if forecast_data and "error" not in forecast_data:
                                st.success(f"✅ {model_type} forecast generated successfully!")

                                # Key metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    current_price = forecast_data.get("current_price", 0)
                                    st.metric("Current Price", f"${current_price:.2f}")
                                with col2:
                                    predicted_price = forecast_data.get("predicted_price", 0)
                                    st.metric("Predicted Price", f"${predicted_price:.2f}")
                                with col3:
                                    price_change = forecast_data.get("price_change_percent", 0)
                                    st.metric("Expected Change", f"{price_change:+.2f}%", delta=f"{price_change:.2f}%")
                                with col4:
                                    st.metric("Model", forecast_data.get("model_type", "Unknown").title())

                                # Confidence interval
                                conf_int = forecast_data.get("confidence_interval", {})
                                if conf_int:
                                    st.subheader("📈 Confidence Interval")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Lower Bound", f"${conf_int.get('lower', 0):.2f}")
                                    with col2:
                                        st.metric("Upper Bound", f"${conf_int.get('upper', 0):.2f}")

                                # Forecast visualization
                                forecast_points = forecast_data.get("forecast_points", [])
                                if forecast_points:
                                    st.subheader("📊 Forecast Visualization")

                                    try:
                                        # Create forecast chart using visualization API
                                        viz_resp = requests.post(
                                            f"{BASE_URL}/api/v1/visualization/forecast-chart",
                                            json=forecast_data,
                                            headers=auth_headers(),
                                            timeout=30,
                                        )

                                        if viz_resp.status_code == 200:
                                            viz_result = viz_resp.json()
                                            viz_data = viz_result.get("visualization", {})

                                            if "plotly_json" in viz_data:
                                                fig_dict = json.loads(viz_data["plotly_json"])
                                                fig = go.Figure(fig_dict)
                                                st.plotly_chart(fig, use_container_width=True)
                                            else:
                                                # Fallback chart
                                                df = pd.DataFrame(forecast_points)
                                                df["date"] = pd.to_datetime(df["date"])
                                                st.line_chart(df.set_index("date")["predicted_price"])
                                        else:
                                            # Fallback chart
                                            df = pd.DataFrame(forecast_points)
                                            df["date"] = pd.to_datetime(df["date"])
                                            st.line_chart(df.set_index("date")["predicted_price"])
                                    except Exception as e:
                                        st.warning(f"Visualization failed, showing simple chart: {e}")
                                        df = pd.DataFrame(forecast_points)
                                        df["date"] = pd.to_datetime(df["date"])
                                        st.line_chart(df.set_index("date")["predicted_price"])

                            else:
                                error_msg = (
                                    forecast_data.get("error", "Unknown error") if forecast_data else "No forecast data"
                                )
                                st.error(f"❌ Forecast failed: {error_msg}")

                    elif resp.status_code in (401, 403):
                        show_auth_error(resp, "forecasting")
                    else:
                        try:
                            error_data = resp.json()
                            detail = error_data.get("detail", "Unknown error")
                        except Exception:
                            detail = f"HTTP {resp.status_code}"
                        st.error(f"❌ Forecast request failed: {detail}")

                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. The model might be training. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# AI Insights Page
elif page == "🧠 AI Insights":
    st.header("🧠 AI-Powered Market Insights")

    if not st.session_state.access_token:
        show_auth_error(None, "AI insights")
    else:
        col1, col2 = st.columns(2)

        with col1:
            symbol = st.text_input("📈 Stock Symbol", "AAPL")

        with col2:
            days_back = st.slider("📅 Analysis Period (days)", 7, 90, 30)

        if st.button("🤖 Generate AI Insights", type="primary"):
            with st.spinner(f"Analyzing market data and generating AI insights for {symbol}..."):
                try:
                    resp = requests.get(
                        f"{BASE_URL}/api/v1/llm/insights/{symbol}",
                        params={"days_back": days_back},
                        headers=auth_headers(),
                        timeout=45,
                    )

                    if resp.status_code == 200:
                        result = resp.json()
                        insights_data = result.get("insights", {})

                        if insights_data and "error" not in insights_data:
                            st.success("✅ AI insights generated successfully!")

                            # Key metrics
                            key_metrics = insights_data.get("key_metrics", {})
                            if key_metrics:
                                st.subheader("📊 Key Metrics")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Current Price", f"${key_metrics.get('current_price', 0):.2f}")
                                with col2:
                                    trend = key_metrics.get("price_trend", "unknown")
                                    st.metric("Trend", trend.title())
                                with col3:
                                    change = key_metrics.get("price_change_pct", 0)
                                    st.metric("Price Change", f"{change:+.2f}%")
                                with col4:
                                    sentiment = key_metrics.get("sentiment", "neutral")
                                    st.metric("News Sentiment", sentiment.title())

                            # AI Insights
                            insights_list = insights_data.get("insights", [])
                            if insights_list:
                                st.subheader("🔍 Market Insights")
                                for i, insight in enumerate(insights_list, 1):
                                    st.info(f"💡 **Insight {i}**: {insight}")

                            # Risk Assessment
                            risk_assessment = insights_data.get("risk_assessment", {})
                            if risk_assessment:
                                st.subheader("⚠️ Risk Assessment")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    risk_level = risk_assessment.get("risk_level", "UNKNOWN")
                                    if risk_level == "LOW":
                                        st.success(f"🟢 Risk Level: {risk_level}")
                                    elif risk_level == "MODERATE":
                                        st.warning(f"🟡 Risk Level: {risk_level}")
                                    else:
                                        st.error(f"🔴 Risk Level: {risk_level}")
                                with col2:
                                    st.metric("Risk Score", f"{risk_assessment.get('risk_score', 0)}/10")
                                with col3:
                                    st.metric("Volatility", f"{risk_assessment.get('volatility', 0):.2f}%")

                                risk_factors = risk_assessment.get("risk_factors", [])
                                if risk_factors:
                                    st.write("**Risk Factors:**")
                                    for factor in risk_factors:
                                        st.write(f"• {factor}")

                            # Recommendations
                            recommendations = insights_data.get("recommendations", [])
                            if recommendations:
                                st.subheader("💡 AI Recommendations")
                                for i, rec in enumerate(recommendations, 1):
                                    st.success(f"📈 **Recommendation {i}**: {rec}")

                            # News Sentiment Analysis
                            news_sentiment = result.get("news_sentiment", {})
                            if news_sentiment and "error" not in news_sentiment:
                                st.subheader("📰 News Sentiment Analysis")

                                overall_sentiment = news_sentiment.get("overall_sentiment", "NEUTRAL")

                                col1, col2 = st.columns(2)
                                with col1:
                                    if overall_sentiment == "POSITIVE":
                                        st.success(f"📈 Overall Sentiment: {overall_sentiment}")
                                    elif overall_sentiment == "NEGATIVE":
                                        st.error(f"📉 Overall Sentiment: {overall_sentiment}")
                                    else:
                                        st.info(f"➡️ Overall Sentiment: {overall_sentiment}")

                                with col2:
                                    st.metric("Articles Analyzed", news_sentiment.get("total_articles", 0))

                                # Sentiment visualization
                                distribution = news_sentiment.get("sentiment_distribution", {})
                                if distribution:
                                    try:
                                        viz_resp = requests.post(
                                            f"{BASE_URL}/api/v1/visualization/sentiment-chart",
                                            json=news_sentiment,
                                            headers=auth_headers(),
                                            timeout=15,
                                        )

                                        if viz_resp.status_code == 200:
                                            viz_result = viz_resp.json()
                                            viz_data = viz_result.get("visualization", {})

                                            if "plotly_json" in viz_data:
                                                fig_dict = json.loads(viz_data["plotly_json"])
                                                fig = go.Figure(fig_dict)
                                                st.plotly_chart(fig, use_container_width=True)
                                    except Exception:
                                        st.write("Sentiment Distribution:")
                                        st.bar_chart(distribution)

                        else:
                            error_msg = insights_data.get("error", "No insights available")
                            st.error(f"❌ AI insights failed: {error_msg}")

                    elif resp.status_code in (401, 403):
                        show_auth_error(resp, "AI insights")
                    else:
                        try:
                            error_data = resp.json()
                            detail = error_data.get("detail", "Unknown error")
                        except Exception:
                            detail = f"HTTP {resp.status_code}"
                        st.error(f"❌ AI insights request failed: {detail}")

                except Exception as e:
                    st.error(f"❌ Error generating AI insights: {e}")

# Visualizations Page
elif page == "📊 Visualizations":
    st.header("📊 Advanced Visualizations")

    if not st.session_state.access_token:
        show_auth_error(None, "visualizations")
    else:
        col1, col2 = st.columns(2)

        with col1:
            symbol = st.text_input("📈 Stock Symbol", "AAPL")

        with col2:
            chart_type = st.selectbox(
                "📈 Chart Type",
                ["Price Chart", "Technical Analysis", "Dashboard"],
                help="Choose the type of chart to generate",
            )

        days_back = st.slider("📅 Historical Data (days)", 30, 365, 90)

        if chart_type == "Price Chart":
            price_chart_type = st.selectbox("Chart Style", ["candlestick", "line", "ohlc"])

            if st.button("📊 Generate Price Chart", type="primary"):
                with st.spinner(f"Creating {price_chart_type} chart for {symbol}..."):
                    try:
                        resp = requests.get(
                            f"{BASE_URL}/api/v1/visualization/price-chart/{symbol}",
                            params={"days_back": days_back, "chart_type": price_chart_type},
                            headers=auth_headers(),
                            timeout=30,
                        )

                        if resp.status_code == 200:
                            result = resp.json()
                            viz_data = result.get("visualization", {})

                            if "plotly_json" in viz_data:
                                fig_dict = json.loads(viz_data["plotly_json"])
                                fig = go.Figure(fig_dict)
                                st.plotly_chart(fig, use_container_width=True)
                                st.success(f"✅ {price_chart_type.title()} chart created successfully!")
                            else:
                                st.error("❌ Chart data not available")
                        else:
                            st.error(f"❌ Failed to create chart: HTTP {resp.status_code}")

                    except Exception as e:
                        st.error(f"❌ Error creating chart: {e}")

        elif chart_type == "Technical Analysis":
            if st.button("📊 Generate Technical Analysis Chart", type="primary"):
                with st.spinner(f"Creating technical analysis chart for {symbol}..."):
                    try:
                        resp = requests.get(
                            f"{BASE_URL}/api/v1/visualization/technical-analysis/{symbol}",
                            params={"days_back": days_back},
                            headers=auth_headers(),
                            timeout=30,
                        )

                        if resp.status_code == 200:
                            result = resp.json()
                            viz_data = result.get("visualization", {})

                            if "plotly_json" in viz_data:
                                fig_dict = json.loads(viz_data["plotly_json"])
                                fig = go.Figure(fig_dict)
                                st.plotly_chart(fig, use_container_width=True)
                                st.success("✅ Technical analysis chart created successfully!")

                                # Show chart info
                                st.info("📊 Chart includes: Price & Moving Averages, Volume, RSI, MACD")
                                st.info(f"📈 Data points: {viz_data.get('data_points', 'N/A')}")
                            else:
                                st.error("❌ Chart data not available")
                        else:
                            st.error(f"❌ Failed to create chart: HTTP {resp.status_code}")

                    except Exception as e:
                        st.error(f"❌ Error creating technical analysis: {e}")

        else:  # Dashboard
            if st.button("📊 Generate Complete Dashboard", type="primary"):
                with st.spinner(f"Creating comprehensive dashboard for {symbol}..."):
                    try:
                        resp = requests.get(
                            f"{BASE_URL}/api/v1/visualization/dashboard/{symbol}",
                            params={"days_back": days_back},
                            headers=auth_headers(),
                            timeout=45,
                        )

                        if resp.status_code == 200:
                            result = resp.json()
                            dashboard_components = result.get("dashboard_components", {})
                            successful_components = result.get("successful_components", [])

                            st.success(f"✅ Dashboard created with {len(successful_components)} components!")

                            # Display each dashboard component
                            for component_name, component_data in dashboard_components.items():
                                if "error" not in component_data:
                                    st.subheader(f"📊 {component_name.replace('_', ' ').title()}")

                                    if "plotly_json" in component_data:
                                        fig_dict = json.loads(component_data["plotly_json"])
                                        fig = go.Figure(fig_dict)
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error(f"❌ {component_name}: {component_data['error']}")

                        else:
                            st.error(f"❌ Failed to create dashboard: HTTP {resp.status_code}")

                    except Exception as e:
                        st.error(f"❌ Error creating dashboard: {e}")

# Portfolio Analysis Page
elif page == "💼 Portfolio Analysis":
    st.header("💼 Portfolio Analysis")

    if not st.session_state.access_token:
        show_auth_error(None, "portfolio analysis")
    else:
        # Portfolio input
        st.subheader("📝 Portfolio Input")

        input_method = st.radio("Input Method", ["Manual Entry", "JSON Format"])

        if input_method == "Manual Entry":
            num_holdings = st.number_input("Number of holdings", min_value=2, max_value=10, value=3)

            portfolio_data = {}

            for i in range(num_holdings):
                col1, col2 = st.columns(2)
                with col1:
                    portfolio_symbol = st.text_input(f"Symbol {i + 1}", value="", key=f"symbol_{i}")
                with col2:
                    weight = st.number_input(
                        f"Weight {i + 1}", value=0.0, min_value=0.0, max_value=1.0, key=f"weight_{i}"
                    )

                if portfolio_symbol:
                    portfolio_data[portfolio_symbol.upper()] = weight

        else:  # JSON Format
            portfolio_json = st.text_area(
                "Portfolio JSON",
                value='{"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3}',
                help="Enter portfolio as JSON with symbol: weight pairs",
            )

            try:
                portfolio_data = json.loads(portfolio_json)
            except Exception:
                portfolio_data = {}
                st.error("❌ Invalid JSON format")

        # Portfolio Analysis
        if portfolio_data and st.button("📊 Analyze Portfolio", type="primary"):
            # Validate weights
            total_weight = sum(portfolio_data.values())
            if abs(total_weight - 1.0) > 0.01:
                st.error(f"❌ Portfolio weights must sum to 1.0, got {total_weight:.3f}")
            else:
                with st.spinner("Analyzing portfolio performance..."):
                    try:
                        # Portfolio performance chart
                        perf_resp = requests.post(
                            f"{BASE_URL}/api/v1/visualization/portfolio-performance",
                            json=portfolio_data,
                            headers=auth_headers(),
                            timeout=30,
                        )

                        if perf_resp.status_code == 200:
                            perf_result = perf_resp.json()
                            viz_data = perf_result.get("visualization", {})

                            if "plotly_json" in viz_data:
                                fig_dict = json.loads(viz_data["plotly_json"])
                                fig = go.Figure(fig_dict)
                                st.plotly_chart(fig, use_container_width=True)

                                # Portfolio metrics
                                metrics = viz_data.get("portfolio_metrics", {})
                                if metrics:
                                    st.subheader("📊 Portfolio Metrics")
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        total_return = metrics.get("total_return_30d", 0)
                                        st.metric("30-Day Return", f"{total_return:+.2f}%")
                                    with col2:
                                        st.metric("Best Performer", metrics.get("best_performer", "N/A"))
                                    with col3:
                                        st.metric("Worst Performer", metrics.get("worst_performer", "N/A"))
                                    with col4:
                                        st.metric("Holdings", metrics.get("symbols_count", 0))

                        # Correlation analysis
                        symbols_list = list(portfolio_data.keys())
                        if len(symbols_list) > 1:
                            st.subheader("🔗 Correlation Analysis")

                            corr_resp = requests.get(
                                f"{BASE_URL}/api/v1/visualization/correlation-matrix",
                                params={"symbols": symbols_list, "days_back": 90},
                                headers=auth_headers(),
                                timeout=30,
                            )

                            if corr_resp.status_code == 200:
                                corr_result = corr_resp.json()
                                viz_data = corr_result.get("visualization", {})

                                if "plotly_json" in viz_data:
                                    fig_dict = json.loads(viz_data["plotly_json"])
                                    fig = go.Figure(fig_dict)
                                    st.plotly_chart(fig, use_container_width=True)

                                    st.info("💡 Lower correlations indicate better diversification")

                    except Exception as e:
                        st.error(f"❌ Portfolio analysis error: {e}")

        # Batch Forecasting
        if portfolio_data:
            st.subheader("🔮 Batch Forecasting")

            col1, col2 = st.columns(2)
            with col1:
                batch_model = st.selectbox("Model for Batch Forecast", ["ensemble", "prophet", "lstm"])
            with col2:
                batch_horizon = st.slider("Forecast Horizon", 7, 30, 14)

            if st.button("🚀 Forecast All Holdings"):
                with st.spinner("Generating forecasts for all portfolio holdings..."):
                    try:
                        symbols_list = list(portfolio_data.keys())

                        batch_resp = requests.get(
                            f"{BASE_URL}/api/v1/forecasting/batch",
                            params={"symbols": symbols_list, "model_type": batch_model, "horizon_days": batch_horizon},
                            headers=auth_headers(),
                            timeout=120,
                        )

                        if batch_resp.status_code == 200:
                            batch_result = batch_resp.json()
                            forecasts = batch_result.get("forecasts", {})
                            summary = batch_result.get("summary", {})

                            # Summary metrics
                            st.subheader("📊 Batch Forecast Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Symbols", summary.get("total_symbols", 0))
                            with col2:
                                st.metric("Successful", summary.get("successful_forecasts", 0))
                            with col3:
                                st.metric("Failed", summary.get("failed_forecasts", 0))
                            with col4:
                                avg_change = summary.get("average_price_change", 0)
                                st.metric("Avg. Change", f"{avg_change:+.2f}%")

                            # Individual forecasts
                            st.subheader("📊 Individual Forecasts")
                            for symbol, forecast in forecasts.items():
                                if "error" not in forecast:
                                    with st.expander(f"{symbol} - {forecast.get('model_type', 'Unknown')} Forecast"):
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Current", f"${forecast.get('current_price', 0):.2f}")
                                        with col2:
                                            st.metric("Predicted", f"${forecast.get('predicted_price', 0):.2f}")
                                        with col3:
                                            change = forecast.get("price_change_percent", 0)
                                            st.metric("Change", f"{change:+.2f}%")
                                else:
                                    st.error(f"❌ {symbol}: {forecast['error']}")

                        else:
                            st.error(f"❌ Batch forecast failed: HTTP {batch_resp.status_code}")

                    except Exception as e:
                        st.error(f"❌ Batch forecast error: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 OpenTrend AI")
st.sidebar.caption("Advanced Financial Analytics Platform")
st.sidebar.caption("Powered by Prophet, LSTM, and AI")

# Quick stats in sidebar
if st.session_state.access_token:
    st.sidebar.markdown("### 🚀 Available Features")
    st.sidebar.success("✅ Prophet Forecasting")
    st.sidebar.success("✅ LSTM Neural Networks")
    st.sidebar.success("✅ Ensemble Models")
    st.sidebar.success("✅ AI Market Insights")
    st.sidebar.success("✅ Advanced Visualizations")
    st.sidebar.success("✅ Portfolio Analysis")
else:
    st.sidebar.info("🔐 Login to access all features")

#!/usr/bin/env python3
"""
OpenTrend AI - Advanced Frontend Launcher
Starts the advanced Streamlit interface with all features
"""

import subprocess
import sys
import os


def main():
    print("🚀 Starting OpenTrend AI Advanced Frontend...")
    print("📊 Features: Prophet, LSTM, AI Insights, Advanced Visualizations, Portfolio Analysis")
    print("🔗 Make sure the backend is running at http://localhost:8000")
    print("-" * 60)

    try:
        # Change to the correct directory
        frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "advanced_streamlit_app.py")

        # Start Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", frontend_path, "--server.port", "8501"]
        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n👋 Shutting down OpenTrend AI Advanced Frontend...")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        print("💡 Make sure you have streamlit installed: uv add streamlit plotly")


if __name__ == "__main__":
    main()

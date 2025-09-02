#!/usr/bin/env python3
"""Setup script for OpenTrend AI application."""

import sys
import subprocess
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"🔄 {description}...")
    try:
        _result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    if sys.version_info < (3, 13):
        print("❌ Python 3.13+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def check_uv_installation() -> bool:
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ uv is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ uv is not installed. Installing...")
        return run_command("pip install uv", "Installing uv")


def setup_environment() -> bool:
    """Setup environment file."""
    env_file = Path(".env")
    env_example = Path("env.example")

    if env_file.exists():
        print("✅ .env file already exists")
        return True

    if not env_example.exists():
        print("❌ env.example file not found")
        return False

    print("📝 Creating .env file from template...")
    try:
        with open(env_example, "r") as f:
            content = f.read()

        with open(env_file, "w") as f:
            f.write(content)

        print("✅ .env file created. Please edit it with your configuration.")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def install_dependencies() -> bool:
    """Install Python dependencies."""
    return run_command("uv sync", "Installing Python dependencies")


def create_directories() -> bool:
    """Create necessary directories."""
    directories = ["logs", "models", "data"]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    print("✅ Created necessary directories")
    return True


def main():
    """Main setup function."""
    print("🚀 OpenTrend AI Setup")
    print("=" * 50)

    # Check prerequisites
    if not check_python_version():
        sys.exit(1)

    if not check_uv_installation():
        sys.exit(1)

    # Setup environment
    if not setup_environment():
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        sys.exit(1)

    # Create directories
    if not create_directories():
        sys.exit(1)

    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your configuration")
    print("2. Start PostgreSQL and Redis")
    print("3. Run: python main.py")
    print("4. Access the API at: http://localhost:8000/docs")


if __name__ == "__main__":
    main()

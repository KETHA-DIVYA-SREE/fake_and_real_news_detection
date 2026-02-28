"""
Setup  the environment and download required models.
"""

import subprocess
import sys
import os


def install_requirements():
    print("Installing required packages...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✓ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False


def download_spacy_model():
    """Download spaCy English model."""
    print("\nDownloading spaCy English model (en_core_web_sm)...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
        )
        print("✓ spaCy model downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading spaCy model: {e}")
        return False


def check_dataset():
    """Check if dataset exists."""
    if os.path.exists("fake_and_real_news.csv"):
        print("✓ Dataset found: fake_and_real_news.csv")
        return True
    else:
        print("⚠ Dataset not found: fake_and_real_news.csv")
        print("  The classifier will need to be trained when you first run the app.")
        return False


def main():
    print("=" * 60)
    print("Fake News Classifier Chatbot - Setup")
    print("=" * 60)

    # Install requirements
    if not install_requirements():
        print("\nSetup failed. Please install requirements manually:")
        print("  pip install -r requirements.txt")
        return

    # Download spaCy model
    if not download_spacy_model():
        print("\nSetup partially complete. Please download spaCy model manually:")
        print("  python -m spacy download en_core_web_sm")

    # Check dataset
    check_dataset()

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Create a .env file and set NVIDIA_NIM_API_KEY for NVIDIA NIM chat")
    print("2. Run the app: streamlit run app.py")
    print(
        "\nNote: The Word2Vec model (~1.6GB) will be downloaded automatically on first run."
    )


if __name__ == "__main__":
    main()

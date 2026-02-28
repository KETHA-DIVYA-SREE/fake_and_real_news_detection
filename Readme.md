# Fake News Classifier Chatbot - Streamlit Application

A Streamlit-based chatbot application that classifies news articles as Real or Fake using Gensim Word2Vec embeddings, and provides a conversational interface with memory using LangChain.

## Features

- **Multiple Input Methods:**
  - Upload PDF files
  - Enter news article URLs
  - Paste plain text content

- **Automatic Classification:**
  - Uses pre-trained GradientBoostingClassifier with Word2Vec embeddings
  - Provides confidence scores for predictions
  - Displays classification results immediately

- **JSON Data Display:**
  - Shows processed document structure in JSON format
  - Includes metadata and extracted content

- **Conversational Chatbot:**
  - Memory-based conversation using LangChain
  - Maintains context across the conversation
  - Can discuss classification results and document content

## Installation

### Quick Setup (Recommended)

Run the setup script:
```bash
python setup.py
```

This will:
- Install all required packages
- Download the spaCy English model
- Check for the dataset

### Manual Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download spaCy English model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Set up NVIDIA NIM (Required for Real-news conversational Q&A + abstractive summarization):**
   - Create a `.env` file and add:
     ```bash
     NVIDIA_NIM_API_KEY=NVIDIA_NIM_YOUR_API_KEY
     NVIDIA_NIM_MODEL=meta/llama-3.1-70b-instruct
     ```
   - Replace `NVIDIA_NIM_YOUR_API_KEY` with your real key.

### Note on Dependencies

- **NVIDIA NIM + LangChain components**: Recommended for Real-news conversational flow (context-grounded Q&A + abstractive summary). The app will still work with rule-based responses if these are not installed.

## Usage

1. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

2. **First Run:**
   - The app will automatically download the Word2Vec model (word2vec-google-news-300) on first run
   - If the classifier model doesn't exist, it will train on `fake_and_real_news.csv`
   - This may take several minutes on the first run

3. **Using the Application:**
   - **Upload PDF:** Use the sidebar to upload a PDF file, then click "Process PDF"
   - **Enter URL:** Paste a news article URL in the sidebar, then click "Process URL"
   - **Paste Text:** Type or paste news content in the text area, then click "Process Text"
   - **View JSON:** Expand "View Processed Data (JSON)" in the sidebar to see the structured data
   - **Chat:** After classification, use the chat interface to discuss the content

## File Structure

```
IBM/
├── app.py                      # Main Streamlit application
├── classification_utils.py     # Classification model and utilities
├── document_processor.py       # PDF and URL processing
├── requirements.txt            # Python dependencies
├── fake_and_real_news.csv     # Training dataset
├── fake_and_real_news.ipynb    # Original notebook
└── README_STREAMLIT.md        # This file
```

## Model Details

- **Word Embeddings:** Google News Word2Vec (300 dimensions)
- **Preprocessing:** spaCy (stop word removal, lemmatization)
- **Classifier:** GradientBoostingClassifier
- **Performance:** ~98% accuracy on test set

## Notes

- The Word2Vec model (~1.6GB) will be downloaded automatically on first use
- PDF processing requires valid PDF files
- URL processing may fail for sites with anti-scraping measures
- The classifier model will be saved as `news_classifier.pkl` after training

## Troubleshooting

1. **spaCy model error:**
   - Run: `python -m spacy download en_core_web_sm`

2. **Memory issues:**
   - The Word2Vec model is large (~1.6GB). Ensure sufficient RAM.

3. **NVIDIA NIM errors:**
   - Ensure `NVIDIA_NIM_API_KEY` is set in `.env` and that your selected `NVIDIA_NIM_MODEL` is available for your account/region.

4. **PDF processing errors:**
   - Ensure PDF files are not encrypted or corrupted
   - Some PDFs may have images only (no extractable text)

"""
Chatbot application for Fake/Real News Classification.
Features:
- Upload PDF files, news article URLs, or plain text
- Classify content as Real or Fake
- Memory-based conversational chatbot using NVIDIA NIM
"""

import streamlit as st
import json
from typing import List, Dict, Any, Optional
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    NIM_AVAILABLE = True
except ImportError:
    NIM_AVAILABLE = False
from classification_utils import get_classifier
from document_processor import DocumentProcessor

# Page configuration
st.set_page_config(
    page_title="Fake News Classifier Chatbot",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []
if "classification_results" not in st.session_state:
    st.session_state.classification_results = []
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "text_splitter" not in st.session_state:
    st.session_state.text_splitter = None
if "classifier" not in st.session_state:
    st.session_state.classifier = None
if "document_processor" not in st.session_state:
    st.session_state.document_processor = DocumentProcessor()


def _latest_classification() -> Optional[Dict[str, Any]]:
    if not st.session_state.classification_results:
        return None
    return st.session_state.classification_results[-1]


def _latest_label() -> Optional[str]:
    latest = _latest_classification()
    if not latest:
        return None
    return latest.get("prediction")


def _has_any_real() -> bool:
    """
    Return True if there is at least one item classified as Real so far.
    """
    for res in st.session_state.classification_results:
        if res.get("prediction") == "Real":
            return True
    return False


def _history_text(max_turns: int = 10) -> str:
    msgs = st.session_state.messages[-(max_turns * 2) :]
    lines: List[str] = []
    for m in msgs:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _real_context_texts() -> List[str]:
    """
    Collect full-text content for all documents that were classified as Real.
    """
    texts: List[str] = []
    for res in st.session_state.classification_results:
        if res.get("prediction") != "Real":
            continue
        idx = res.get("document_index")
        if idx is None:
            continue
        if 0 <= idx < len(st.session_state.processed_documents):
            doc = st.session_state.processed_documents[idx]
            if doc.get("status") == "success" and doc.get("full_text"):
                texts.append(doc["full_text"])
    return texts


def _nim_llm():
    """
    NVIDIA NIM chat model instance.
    """
    if not NIM_AVAILABLE:
        return None
    nim_key = os.getenv("NVIDIA_NIM_API_KEY")
    if not nim_key or nim_key.strip() == "NVIDIA_NIM_YOUR_API_KEY":
        return None
    model_name = os.getenv("NVIDIA_NIM_MODEL", "meta/llama-3.1-70b-instruct")
    try:
        return ChatNVIDIA(model=model_name, api_key=nim_key)
    except TypeError:
        return ChatNVIDIA(model_name=model_name, api_key=nim_key)


def _ensure_vectorstore():
    """
    FAISS vector store from Real-classified uploaded content.
    """
    if not NIM_AVAILABLE:
        st.session_state.vectors = None
        return
    texts = _real_context_texts()
    if not texts:
        st.session_state.vectors = None
        return
    if st.session_state.text_splitter is None:
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
    if st.session_state.embeddings is None:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    chunks: List[str] = []
    for t in texts:
        chunks.extend(st.session_state.text_splitter.split_text(t))
    st.session_state.vectors = FAISS.from_texts(chunks, st.session_state.embeddings)


def _extractive_summary(text: str, max_sentences: int = 5) -> str:
    """
    Simple extractive summarization using TF-IDF sentence scoring.
    """
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()
    ]
    if not sentences:
        return "No text available to summarize."
    if len(sentences) <= max_sentences:
        return "\n".join(f"- {s}" for s in sentences)
    X = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()
    top_idx = scores.argsort()[::-1][:max_sentences]
    top_idx_sorted = sorted(top_idx)
    return "\n".join(f"- {sentences[i]}" for i in top_idx_sorted)


def generate_rule_based_response(
    user_input: str,
    classification_results: List[Dict],
    processed_docs: List[Dict],
    message_history: List[Dict] = None,
) -> str:
    """
    Generate a rule-based response Pattern matching and maintains basic context awareness.
    """
    user_lower = user_input.lower()
    message_history = message_history or []
    # Check for greeting
    if any(word in user_lower for word in ["hello", "hi", "hey", "greetings"]):
        if classification_results:
            return f"Hello! I've classified {len(classification_results)} piece(s) of content. How can I help you today?"
        else:
            return "Hello! I'm a Fake News Classifier Chatbot. Please upload a PDF, enter a news article URL, or paste text content from the sidebar to get started!"
    # Check for classification-related queries
    if any(
        word in user_lower
        for word in [
            "classify",
            "classification",
            "result",
            "prediction",
            "fake",
            "real",
        ]
    ):
        if classification_results:
            latest = classification_results[-1]
            response = f"**Latest Classification Result:**\n\n- **Type:** {latest['type']}\n- **Prediction:** {latest['prediction']}\n- **Confidence:** {latest['confidence']:.2%}\n"
            # Add context if available
            if latest.get("text_preview"):
                response += f"\n**Content Preview:**\n{latest['text_preview']}\n"
            response += (
                "\nThis prediction is based on the **full text content** of the news "
                "(from the URL, PDF, or pasted text), not just the URL or source domain.\n"
                "Would you like to know more about this classification or discuss the content?"
            )
            return response
        else:
            return "No content has been classified yet. Please upload a PDF, enter a URL, or paste text content from the sidebar first."
    # Check for document-related queries
    if any(
        word in user_lower
        for word in ["document", "content", "text", "article", "what", "show"]
    ):
        if processed_docs:
            latest = processed_docs[-1]
            if latest["status"] == "success":
                preview = (
                    latest["full_text"][:500] + "..."
                    if len(latest["full_text"]) > 500
                    else latest["full_text"]
                )
                response = f"**Latest Document Summary:**\n\n- **Type:** {latest['type']}\n- **Length:** {latest.get('text_length', 0)} characters\n"
                if latest.get("title"):
                    response += f"- **Title:** {latest['title']}\n"
                if latest.get("url"):
                    response += f"- **URL:** {latest['url']}\n"
                response += f"\n**Content Preview:**\n{preview}\n\nWould you like to see more details or discuss specific aspects?"
                return response
            else:
                return (
                    f"Error processing document: {latest.get('error', 'Unknown error')}"
                )
        else:
            return "No documents have been processed yet. Please upload content from the sidebar."
    # Check for help requests
    if any(word in user_lower for word in ["help", "how", "what can", "explain"]):
        return """I can help you with:
- **Classifying news:** Upload PDFs, URLs, or paste text to classify as Real or Fake
- **Viewing results:** Ask about classification results and confidence scores
- **Discussing content:** Talk about the classified documents and their content
Use the sidebar to upload content, then ask me questions about it!"""
    # Check for thank you / goodbye
    if any(word in user_lower for word in ["thank", "thanks", "bye", "goodbye"]):
        return (
            "You're welcome! Feel free to upload more content or ask questions anytime."
        )
    # General response with context
    if classification_results:
        return f"I'm here to help! I've classified {len(classification_results)} piece(s) of content. You can ask me about:\n- Classification results (ask 'what is the classification?')\n- Document details (ask 'show me the document')\n- Content analysis\n\nWhat would you like to know?"
    else:
        return "I'm a Fake News Classifier Chatbot. Please upload a PDF, enter a news article URL, or paste text content from the sidebar to get started. I'll classify it as Real or Fake, and then we can discuss it! You can also ask me 'help' to see what I can do."


# Initialize classifier (lazy loading)
@st.cache_resource
def load_classifier():
    """Load the news classifier model."""
    try:
        return get_classifier()
    except Exception as e:
        st.error(f"Error loading classifier: {str(e)}")
        return None


if st.session_state.classifier is None:
    with st.spinner(
        "Loading classification model (this may take a minute on first run)..."
    ):
        st.session_state.classifier = load_classifier()

# Sidebar for file/URL upload
with st.sidebar:
    st.header("📤 Upload Content")
    # File uploader for PDF
    st.subheader("Upload PDF File")
    uploaded_pdf = st.file_uploader(
        "Choose a PDF file", type=["pdf"], key="pdf_uploader"
    )
    # Process uploaded PDF
    if uploaded_pdf is not None:
        if st.button("Process PDF", key="process_pdf"):
            with st.spinner("Processing PDF..."):
                result = st.session_state.document_processor.extract_text_from_pdf(
                    uploaded_pdf
                )
                st.session_state.processed_documents.append(result)
                if result["status"] == "success" and result["full_text"]:
                    # Classify the content
                    if st.session_state.classifier:
                        prediction, confidence, label = (
                            st.session_state.classifier.predict(result["full_text"])
                        )
                        classification_result = {
                            "document_index": len(st.session_state.processed_documents)
                            - 1,
                            "type": "PDF",
                            "filename": uploaded_pdf.name,
                            "prediction": label,
                            "confidence": float(confidence),
                            "text_preview": (
                                result["full_text"][:200] + "..."
                                if len(result["full_text"]) > 200
                                else result["full_text"]
                            ),
                        }
                        st.session_state.classification_results.append(
                            classification_result
                        )
                        # Add classification message to chat
                        classification_msg = f"📄 **PDF Uploaded and Classified**\n\n**File:** {uploaded_pdf.name}\n**Classification:** {label}\n**Confidence:** {confidence:.2%}\n\nI've analyzed the PDF content. How can I help you with this article?"
                        st.session_state.messages.append(
                            {"role": "assistant", "content": classification_msg}
                        )
                        st.rerun()
                    else:
                        st.error(
                            "Classifier not available. Please check the model setup."
                        )
                else:
                    st.error(
                        f"Error processing PDF: {result.get('error', 'Unknown error')}"
                    )
    # URL input
    st.subheader("Enter News Article URL")
    url_input = st.text_input(
        "Paste news article URL here:",
        key="url_input",
        placeholder="https://example.com/news-article",
    )
    url_submit = st.button("Process URL", key="url_submit")
    # Plain text input
    st.subheader("Enter Plain Text")
    text_input = st.text_area(
        "Paste or type news content here:",
        key="text_input",
        height=150,
        placeholder="Enter news article text...",
    )
    text_submit = st.button("Process Text", key="text_submit")
    # Process URL
    if url_submit and url_input:
        with st.spinner("Fetching and processing URL..."):
            result = st.session_state.document_processor.extract_text_from_url(
                url_input
            )
            st.session_state.processed_documents.append(result)
            if result["status"] == "success" and result["full_text"]:
                # Classify the content
                if st.session_state.classifier:
                    prediction, confidence, label = st.session_state.classifier.predict(
                        result["full_text"]
                    )
                    classification_result = {
                        "document_index": len(st.session_state.processed_documents) - 1,
                        "type": "URL",
                        "url": url_input,
                        "prediction": label,
                        "confidence": float(confidence),
                        "text_preview": (
                            result["full_text"][:200] + "..."
                            if len(result["full_text"]) > 200
                            else result["full_text"]
                        ),
                    }
                    st.session_state.classification_results.append(
                        classification_result
                    )
                    # Add classification message to chat
                    classification_msg = f"🔗 **URL Processed and Classified**\n\n**URL:** {url_input}\n**Title:** {result.get('title', 'N/A')}\n**Classification:** {label}\n**Confidence:** {confidence:.2%}\n\nI've analyzed the article. How can I help you with this news?"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": classification_msg}
                    )
                    st.rerun()
                else:
                    st.error("Classifier not available. Please check the model setup.")
            else:
                st.error(
                    f"Error processing URL: {result.get('error', 'Unknown error')}"
                )
    # Process plain text
    if text_submit and text_input:
        with st.spinner("Processing text..."):
            result = st.session_state.document_processor.process_plain_text(text_input)
            st.session_state.processed_documents.append(result)
            if result["status"] == "success" and result["full_text"]:
                # Classify the content
                if st.session_state.classifier:
                    prediction, confidence, label = st.session_state.classifier.predict(
                        result["full_text"]
                    )
                    classification_result = {
                        "document_index": len(st.session_state.processed_documents) - 1,
                        "type": "Plain Text",
                        "prediction": label,
                        "confidence": float(confidence),
                        "text_preview": (
                            result["full_text"][:200] + "..."
                            if len(result["full_text"]) > 200
                            else result["full_text"]
                        ),
                    }
                    st.session_state.classification_results.append(
                        classification_result
                    )
                    # Add classification message to chat
                    classification_msg = f"📝 **Text Processed and Classified**\n\n**Classification:** {label}\n**Confidence:** {confidence:.2%}\n\nI've analyzed the text. How can I help you with this content?"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": classification_msg}
                    )
                    st.rerun()
                else:
                    st.error("Classifier not available. Please check the model setup.")
    # Display processed documents JSON
    if st.session_state.processed_documents:
        st.divider()
        st.subheader("📋 Processed Documents")
        with st.expander("View Processed Data (JSON)", expanded=False):
            for idx, doc in enumerate(st.session_state.processed_documents):
                st.write(f"**Document {idx + 1}:**")
                json_str = st.session_state.document_processor.format_as_json(doc)
                st.json(json.loads(json_str))
                st.write("---")
    st.divider()
    st.subheader("🧠 Real-news Context (Embeddings)")
    if st.button("Create / Rebuild Embeddings (Real only)"):
        if not NIM_AVAILABLE:
            st.error(
                "Missing RAG dependencies. Install requirements.txt to enable NVIDIA NIM + embeddings."
            )
        else:
            with st.spinner("Building vector store from Real-classified content..."):
                _ensure_vectorstore()
            if st.session_state.vectors is None:
                st.warning("No Real-classified content available to embed yet.")
            else:
                st.success("Vector store is ready.")
    if st.button("Delete Document Embeddings"):
        st.session_state.vectors = None
        st.session_state.embeddings = None
        st.session_state.text_splitter = None
        st.info("Deleted vector store from session.")
# Main chat interface
st.title("📰 Fake News Classifier Chatbot")
st.markdown(
    "Upload news content (PDF, URL, or text) to classify it as Real or Fake, then chat about it!"
)
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# Chat input
if prompt := st.chat_input(
    "Ask me about the classified news or upload content from the sidebar..."
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # Generate response using NVIDIA NIM (Real only) or simple rule-based
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            latest_label = _latest_label()
            llm = _nim_llm()
            # Use NVIDIA NIM whenever there is at least one Real-classified item so earlier Real context is not "forgotten" after a Fake item.
            if _has_any_real() and llm is not None:
                # Build vector store if missing (use uploaded content as context)
                if st.session_state.vectors is None:
                    _ensure_vectorstore()
                lower = prompt.lower()
                # 1) Purely extractive summary (local, no NIM)
                if "extractive" in lower and "summary" in lower:
                    texts = _real_context_texts()
                    latest_text = texts[-1] if texts else ""
                    response = (
                        "**Extractive Summary (from your Real-news context):**\n\n"
                        + _extractive_summary(latest_text)
                    )
                # 2) Any other kind of summary (e.g. 'summary', 'summarize') -> feed full article to NIM
                elif "summary" in lower or "summarize" in lower:
                    texts = _real_context_texts()
                    latest_text = texts[-1] if texts else ""
                    history_text = _history_text()
                    system_instructions = (
                        "You are a helpful assistant for news summarization.\n"
                        "Use the provided NEWS_CONTENT to create a clear, concise summary.\n"
                        "If the user asks for specific types of summaries (e.g., key points, short, detailed), follow that.\n"
                    )
                    user_prompt = (
                        f"{system_instructions}\n\n"
                        f"NEWS_CONTENT:\n{latest_text}\n\n"
                        f"CHAT HISTORY:\n{history_text}\n\n"
                        f"USER REQUEST:\n{prompt}\n"
                    )
                    try:
                        resp = llm.invoke(user_prompt)
                        response = getattr(resp, "content", None) or str(resp)
                    except Exception as e:
                        st.warning(
                            f"NVIDIA NIM error: {str(e)}. Using fallback responses."
                        )
                        response = generate_rule_based_response(
                            prompt,
                            st.session_state.classification_results,
                            st.session_state.processed_documents,
                            st.session_state.messages,
                        )
                # 3) General Q&A / analysis over Real-news context (RAG style)
                else:
                    # Retrieve relevant context chunks
                    context_chunks: List[str] = []
                    if st.session_state.vectors is not None:
                        try:
                            docs = st.session_state.vectors.similarity_search(
                                prompt, k=4
                            )
                            context_chunks = [d.page_content for d in docs]
                        except Exception:
                            context_chunks = []
                    # Fallback: use latest document text
                    if not context_chunks:
                        texts = _real_context_texts()
                        if texts:
                            context_chunks = [texts[-1][:4000]]
                    context_text = "\n\n---\n\n".join(context_chunks)
                    history_text = _history_text()
                    system_instructions = (
                        "You are a helpful assistant for news analysis.\n"
                        "Only use the provided CONTEXT to answer.\n"
                        "You can do Q&A and high-level analysis.\n"
                        "If the answer is not in the context, say you don't have enough information.\n"
                    )
                    user_prompt = (
                        f"{system_instructions}\n\n"
                        f"CONTEXT:\n{context_text}\n\n"
                        f"CHAT HISTORY:\n{history_text}\n\n"
                        f"USER INPUT:\n{prompt}\n"
                    )
                    try:
                        resp = llm.invoke(user_prompt)
                        response = getattr(resp, "content", None) or str(resp)
                    except Exception as e:
                        st.warning(
                            f"NVIDIA NIM error: {str(e)}. Using fallback responses."
                        )
                        response = generate_rule_based_response(
                            prompt,
                            st.session_state.classification_results,
                            st.session_state.processed_documents,
                            st.session_state.messages,
                        )
            else:
                if _has_any_real() and llm is None:
                    st.info(
                        "To enable NVIDIA NIM chat on Real news, set NVIDIA_NIM_API_KEY in your .env file."
                    )
                response = generate_rule_based_response(
                    prompt,
                    st.session_state.classification_results,
                    st.session_state.processed_documents,
                    st.session_state.messages,
                )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


# Display classification results in main area
if st.session_state.classification_results:
    st.divider()
    st.subheader("📊 Classification Results")

    for idx, result in enumerate(st.session_state.classification_results):
        with st.expander(
            f"Result {idx + 1}: {result['type']} - {result['prediction']} ({result['confidence']:.2%})",
            expanded=False,
        ):
            st.json(result)

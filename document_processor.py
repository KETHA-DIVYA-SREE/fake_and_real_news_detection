"""
Document processing module for extracting text from PDFs and URLs.
"""

import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import json
from typing import Dict, Any, Optional
from urllib.parse import urlparse


class DocumentProcessor:
    """Class for processing various document types (PDF, URL, plain text)."""

    @staticmethod
    def extract_text_from_pdf(pdf_file) -> Dict[str, Any]:
        """
        Extract text from a PDF file.
        """
        try:
            if isinstance(pdf_file, bytes):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
            else:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = []
            num_pages = len(pdf_reader.pages)
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text.strip():
                    text_content.append({"page": page_num + 1, "text": text.strip()})
            full_text = "\n\n".join([page["text"] for page in text_content])
            return {
                "type": "PDF",
                "status": "success",
                "num_pages": num_pages,
                "pages": text_content,
                "full_text": full_text,
                "text_length": len(full_text),
                "metadata": {
                    "title": (
                        pdf_reader.metadata.get("/Title", "Unknown")
                        if pdf_reader.metadata
                        else "Unknown"
                    ),
                    "author": (
                        pdf_reader.metadata.get("/Author", "Unknown")
                        if pdf_reader.metadata
                        else "Unknown"
                    ),
                },
            }
        except Exception as e:
            return {"type": "PDF", "status": "error", "error": str(e), "full_text": ""}

    @staticmethod
    def extract_text_from_url(url: str) -> Dict[str, Any]:
        """
        Extract text content from a news article URL.
        """
        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return {
                    "type": "URL",
                    "status": "error",
                    "error": "Invalid URL format",
                    "full_text": "",
                }
            # Fetch the webpage
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            # Parse HTML
            soup = BeautifulSoup(response.content, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            # Try to find article content (common patterns)
            article_text = ""
            article_title = ""
            # Try to find title
            title_tag = soup.find("title")
            if title_tag:
                article_title = title_tag.get_text().strip()
            # Try common article selectors
            article_selectors = [
                "article",
                '[role="article"]',
                ".article-content",
                ".article-body",
                ".post-content",
                ".entry-content",
                "main",
                ".content",
            ]
            article_content = None
            for selector in article_selectors:
                article_content = soup.select_one(selector)
                if article_content:
                    break
            if article_content:
                article_text = article_content.get_text(separator="\n", strip=True)
            else:
                # Fallback: get all paragraph text
                paragraphs = soup.find_all("p")
                article_text = "\n\n".join(
                    [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
                )
            return {
                "type": "URL",
                "status": "success",
                "url": url,
                "title": article_title,
                "full_text": article_text,
                "text_length": len(article_text),
                "metadata": {"domain": parsed.netloc, "path": parsed.path},
            }
        except requests.exceptions.RequestException as e:
            return {
                "type": "URL",
                "status": "error",
                "error": f"Failed to fetch URL: {str(e)}",
                "full_text": "",
            }
        except Exception as e:
            return {"type": "URL", "status": "error", "error": str(e), "full_text": ""}

    @staticmethod
    def process_plain_text(text: str) -> Dict[str, Any]:
        """
        Process plain text input.
        """
        return {
            "type": "Plain Text",
            "status": "success",
            "full_text": text.strip(),
            "text_length": len(text.strip()),
            "metadata": {},
        }

    @staticmethod
    def format_as_json(data: Dict[str, Any], indent: int = 2) -> str:
        """
        Format processed data as JSON string.
        Args:
            data: Dictionary with processed document data
            indent: JSON indentation level
        Returns:
            Formatted JSON string
        """
        display_data = data.copy()
        if "full_text" in display_data and len(display_data["full_text"]) > 500:
            display_data["full_text_preview"] = display_data["full_text"][:500] + "..."
            display_data["full_text"] = "[Truncated - see full_text_preview]"
        return json.dumps(display_data, indent=indent, ensure_ascii=False)

import fitz
import json
import os
import time
import itertools
import re
import base64
from flask import Flask, request, jsonify
import google.generativeai as genai
import traceback

app = Flask(__name__)

# --- Configuration ---
GEMINI_MODEL = 'gemini-2.5-flash'
# In a serverless environment, API keys are typically passed via environment variables
# or securely fetched at runtime. We'll expect them from environment variables.

# Global cycler for API keys - will be initialized once per instance
api_key_cycler = None
API_KEYS_LIST = []

def initialize_gemini_api_keys():
    """Initializes the Gemini API keys from environment variable."""
    global api_key_cycler, API_KEYS_LIST
    if api_key_cycler is None: # Initialize only once per worker instance
        API_KEYS_STR = os.environ.get("GEMINI_API_KEYS")
        if not API_KEYS_STR:
            raise ValueError("GEMINI_API_KEYS environment variable not set.")
        API_KEYS_LIST = [key.strip() for key in API_KEYS_STR.split(',') if key.strip()]
        if not API_KEYS_LIST:
            raise ValueError("No valid API keys found in GEMINI_API_KEYS.")
        api_key_cycler = itertools.cycle(API_KEYS_LIST)
        app.logger.info(f"Initialized {len(API_KEYS_LIST)} API key(s).")
    
    # Configure genai with the first key immediately to avoid delays
    current_api_key = next(api_key_cycler)
    genai.configure(api_key=current_api_key)
    app.logger.info(f"Initial API key configured (starts with: {current_api_key[:5]}...)")


# --- Functions ---
def extract_text_from_pdf_bytes(pdf_bytes: bytes, page_range_str: str = None) -> str:
    """
    Extracts text from PDF bytes, with optional page range.
    """
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        
        pages_to_extract = []
        if page_range_str:
            for part in page_range_str.split(','):
                part = part.strip()
                if '-' in part:
                    start_str, end_str = part.split('-')
                    start = int(start_str.strip())
                    end = int(end_str.strip())
                    pages_to_extract.extend(range(start, end + 1))
                else:
                    pages_to_extract.append(int(part))
            
            # Ensure pages are within document bounds and unique
            pages_to_extract = sorted(list(set(p for p in pages_to_extract if 1 <= p <= len(document))))
        else:
            pages_to_extract = range(1, len(document) + 1) # All pages if no range specified

        for page_num in pages_to_extract:
            page = document.load_page(page_num - 1) # fitz is 0-indexed
            text += page.get_text()
        
        document.close()
        return text
    except Exception as e:
        app.logger.error(f"Error extracting text from PDF: {e}")
        traceback.print_exc()
        raise

def call_gemini_api_for_extraction(text_chunk: str, custom_prompt: str, retries: int = 5, delay: int = 2):
    """
    Calls Gemini API to extract flashcard data. Uses a global API key cycler.
    """
    global api_key_cycler, API_KEYS_LIST

    # Ensure API keys are initialized
    if api_key_cycler is None:
        initialize_gemini_api_keys()

    for i in range(retries):
        try:
            current_api_key = next(api_key_cycler) # Get next API key from the cycle
            genai.configure(api_key=current_api_key) # Reconfigure with the new key

            model = genai.GenerativeModel(GEMINI_MODEL)
            app.logger.info(f"Attempting API call for chunk (attempt {i+1}/{retries}) with key '{current_api_key[:5]}...')...")
            
            # Use the custom_prompt provided by the frontend
            prompt_with_text = f"{custom_prompt}\n\n---Text to analyze:---\n{text_chunk}\n---End Text---"

            response = model.generate_content(
                contents=[{"role": "user", "parts": [{"text": prompt_with_text}]}]
            )
            raw_text = response.text.strip()
            app.logger.info(f"Gemini API responded for chunk. Raw response starts with: {raw_text[:100]}...")

            match = re.search(r'```json\s*(\[.*?\])\s*```', raw_text, re.DOTALL)
            if match:
                json_str = match.group(1)
                app.logger.info("Extracted JSON using regex.")
            else:
                json_str = raw_text
                app.logger.warning("No markdown JSON block found, assuming raw response is JSON.")

            try:
                extracted_data = json.loads(json_str)
                app.logger.info(f"Successfully parsed JSON. Found {len(extracted_data)} flashcard entries.")
                return extracted_data if isinstance(extracted_data, list) else []
            except json.JSONDecodeError as e:
                app.logger.error(f"JSON parse error after cleaning: {e}")
                app.logger.error(f"Problematic JSON string: {json_str[:200]}...")
                return []
        except Exception as e:
            app.logger.error(f"API call failed (attempt {i+1}/{retries}) with key '{current_api_key[:5]}...'): {e}")
            traceback.print_exc()
            if i < retries - 1:
                sleep_time = delay * (2 ** i)
                app.logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                app.logger.error("Max retries reached. Skipping this chunk.")
                return []
    return []

def process_text_in_chunks_for_api(full_text: str, custom_prompt: str, chunk_size: int = 3000):
    """
    Processes the full text in chunks and collects all flashcards.
    Designed for API response, not direct file writing.
    """
    text_len = len(full_text)
    all_flashcards = []
    
    for i in range(0, text_len, chunk_size):
        chunk = full_text[i:i + chunk_size]
        app.logger.info(f"Processing chunk {int(i/chunk_size) + 1}...")
        
        chunk_data = call_gemini_api_for_extraction(chunk, custom_prompt)
        if chunk_data:
            all_flashcards.extend(chunk_data)
        time.sleep(1) # Small delay to respect API limits

    return all_flashcards


# --- Flask Route ---
@app.route('/extract-flashcards', methods=['POST'])
def extract_flashcards_endpoint():
    """
    Handles POST requests to extract flashcards from a PDF.
    Expects JSON body with:
    {
        "pdfContent": "base64_encoded_pdf_string",
        "pageRange": "optional_page_range_string_e.g._1-5,7",
        "customPrompt": "Your detailed Gemini prompt string"
    }
    """
    # Ensure API keys are initialized when a request comes in
    # This might run once per cold start of the serverless function.
    try:
        initialize_gemini_api_keys()
    except ValueError as e:
        app.logger.error(f"Initialization error: {e}")
        return jsonify({"error": f"Server configuration error: {e}"}), 500

    data = request.json
    if not data or 'pdfContent' not in data or 'customPrompt' not in data:
        return jsonify({"error": "Missing pdfContent or customPrompt in request body"}), 400

    pdf_content_base64 = data['pdfContent']
    page_range = data.get('pageRange')
    custom_prompt = data['customPrompt']

    if not pdf_content_base64:
        return jsonify({"error": "pdfContent cannot be empty"}), 400

    try:
        pdf_bytes = base64.b64decode(pdf_content_base64)
        
        app.logger.info("Extracting text from PDF bytes...")
        pdf_text = extract_text_from_pdf_bytes(pdf_bytes, page_range)
        
        if not pdf_text:
            return jsonify({"error": "No text could be extracted from the PDF."}), 400

        app.logger.info("Sending extracted text to Gemini API for processing...")
        flashcards = process_text_in_chunks_for_api(pdf_text, custom_prompt)
        
        app.logger.info(f"Successfully extracted {len(flashcards)} flashcards.")
        return jsonify({"flashcards": flashcards}), 200

    except Exception as e:
        app.logger.error(f"Processing error: {e}")
        traceback.print_exc() # Log full traceback for debugging
        return jsonify({"error": str(e)}), 500

# To run locally for testing:
# if __name__ == '__main__':
#     # Set a dummy API key for local testing
#     os.environ["GEMINI_API_KEYS"] = "YOUR_GEMINI_API_KEY_HERE"
#     app.run(debug=True, host='0.0.0.0', port=8000)

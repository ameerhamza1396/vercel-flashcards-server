import base64
import fitz
import pandas as pd
import json
import os
import time
import itertools
import re
from pathlib import Path
import google.generativeai as genai
import tempfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler
import traceback
from io import BytesIO

# Load env variables from Vercel's environment
API_KEYS_STR = os.environ.get("GEMINI_API_KEYS", "")
if not API_KEYS_STR:
    raise ValueError("GEMINI_API_KEYS environment variable not set")
API_KEYS = [key.strip() for key in API_KEYS_STR.split(',') if key.strip()]
api_key_cycler = itertools.cycle(API_KEYS)
current_api_key = next(api_key_cycler)

GEMINI_MODEL = 'gemini-2.5-flash'

def extract_text_from_pdf_bytes(pdf_bytes):
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    document.close()
    return text

def call_gemini_api_for_extraction(text_chunk, prompt_text):
    global current_api_key, api_key_cycler
    prompt = f"{prompt_text}\n---\n{text_chunk}\n---"

    for i in range(3):
        try:
            genai.configure(api_key=current_api_key)
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(
                contents=[{"role": "user", "parts": [{"text": prompt}]}]
            )
            raw_text = response.text.strip()

            match = re.search(r'```json\s*(\[.*?\])\s*```', raw_text, re.DOTALL)
            json_str = match.group(1) if match else raw_text
            extracted_data = json.loads(json_str)
            return extracted_data if isinstance(extracted_data, list) else []
        except Exception as e:
            current_api_key = next(api_key_cycler)
            time.sleep(1)
    return []

def process_text_in_chunks(full_text, prompt_text, chunk_size=3000):
    all_data = []
    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i + chunk_size]
        chunk_data = call_gemini_api_for_extraction(chunk, prompt_text)
        all_data.extend(chunk_data)
    return all_data

def handler(request):
    try:
        body = json.loads(request.rfile.read(int(request.headers['Content-Length'])))
        pdf_base64 = body.get("pdfContent")
        prompt_text = body.get("promptText", "Default prompt")
        output_filename = body.get("outputFilename", f"flashcards_{int(time.time())}")

        if not pdf_base64:
            return (400, {"error": "No PDF content provided"})

        pdf_bytes = base64.b64decode(pdf_base64)
        pdf_text = extract_text_from_pdf_bytes(pdf_bytes)
        flashcards = process_text_in_chunks(pdf_text, prompt_text)

        # Save to CSV
        temp_dir = tempfile.gettempdir()
        csv_path = os.path.join(temp_dir, f"{output_filename}.csv")
        pd.DataFrame(flashcards).to_csv(csv_path, index=False)

        # For now, return CSV content as base64 (can later switch to Supabase Storage)
        with open(csv_path, "rb") as f:
            csv_base64 = base64.b64encode(f.read()).decode()

        return (200, {
            "flashcardsCount": len(flashcards),
            "fileName": f"{output_filename}.csv",
            "fileContentBase64": csv_base64
        })

    except Exception as e:
        traceback.print_exc()
        return (500, {"error": str(e)})

# Vercel entrypoint
def handler_main(request, response):
    status, data = handler(request)
    response.status_code = status
    response.headers["Content-Type"] = "application/json"
    response.write(json.dumps(data))

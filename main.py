import base64
import io
import csv
import PyPDF2
import requests
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Your Gemini API key

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def call_gemini_api(prompt, instructions, text):
    """
    Calls Gemini API with the provided prompt, instructions, and text.
    You can adjust this to fit your exact Gemini request structure.
    """
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}

    payload = {
        "contents": [{
            "parts": [{"text": f"{prompt}\n\nInstructions: {instructions}\n\nText:\n{text}"}]
        }]
    }

    response = requests.post(url, headers=headers, params=params, json=payload)
    response.raise_for_status()
    data = response.json()

    # Extract generated text
    return data["candidates"][0]["content"]["parts"][0]["text"]

def create_csv_base64(original_text, generated_text):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Original Text", "Generated Text"])
    writer.writerow([original_text, generated_text])
    csv_bytes = output.getvalue().encode("utf-8")
    return base64.b64encode(csv_bytes).decode("utf-8")

@app.route("/process-pdf", methods=["POST"])
def process_pdf():
    try:
        data = request.get_json()

        pdf_base64 = data.get("pdf_base64")
        prompt = data.get("prompt", "")
        instructions = data.get("instructions", "")

        if not pdf_base64 or not prompt:
            return jsonify({"error": "PDF and prompt are required"}), 400

        pdf_bytes = base64.b64decode(pdf_base64)
        pdf_file = io.BytesIO(pdf_bytes)

        extracted_text = extract_text_from_pdf(pdf_file)
        generated_text = call_gemini_api(prompt, instructions, extracted_text)
        csv_base64 = create_csv_base64(extracted_text, generated_text)

        return jsonify({"csv_base64": csv_base64}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

@app.route("/process-pdf", methods=["POST"])
def process_pdf():
    try:
        data = request.get_json()
        pdf_url = data.get("pdfUrl")
        page_range = data.get("pageRange")
        output_filename = data.get("outputFilename", "output.txt")

        if not pdf_url:
            return jsonify({"error": "Missing pdfUrl"}), 400

        # 1️⃣ Download the PDF from Supabase
        pdf_response = requests.get(pdf_url)
        if pdf_response.status_code != 200:
            return jsonify({"error": "Failed to download PDF"}), 500

        pdf_bytes = pdf_response.content

        # 2️⃣ Save locally (optional)
        with open("temp.pdf", "wb") as f:
            f.write(pdf_bytes)

        # 3️⃣ Process the PDF (replace with your actual logic)
        processed_text = f"Processed {len(pdf_bytes)} bytes from PDF."
        if page_range:
            processed_text += f" Pages: {page_range}"

        # 4️⃣ Return result
        return jsonify({
            "status": "success",
            "outputFilename": output_filename,
            "data": processed_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

client = OpenAI()
@app.route("/embed", methods=["POST"])
def embed():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    texts = data["text"]
    if isinstance(texts, str):
        texts = [texts]

    try:
        
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small",
            encoding_format="float"
        )

        embeddings = [item.input for item in response.output_text]
        return jsonify({"embeddings": embeddings})
        

    except Exception as e:
        return jsonify({"error": str(e)}), 500


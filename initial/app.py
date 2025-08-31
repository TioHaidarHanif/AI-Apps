from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)  # ✅ aktifkan CORS untuk semua route

# load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# contoh data dokumen
documents = [
    "Python adalah bahasa pemrograman populer.",
    "Flask digunakan untuk membuat API.",
    "FastAPI lebih modern daripada Flask.",
    "React adalah library frontend untuk JavaScript.",
]

doc_embeddings = model.encode(documents, convert_to_tensor=True)

@app.route("/search", methods=["POST"])
def search():
    query = request.json.get("query")
    query_embedding = model.encode(query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_results = cos_scores.topk(3)

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        results.append({
            "text": documents[idx],
            "score": float(score)
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)

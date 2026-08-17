import os
import sys
from flask import Flask, render_template, request, jsonify
try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False

# Add root project directory to sys.path so src imports work cleanly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import get_classifier

TEMPLATE_FOLDER = os.path.join(ROOT_DIR, "frontend", "templates")
STATIC_FOLDER = os.path.join(ROOT_DIR, "frontend", "static")

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
if HAS_CORS:
    CORS(app)

# Initialize predictor on startup
classifier = get_classifier()

@app.route("/")
def index():
    """Render main application frontend."""
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Cyberbullying Detection & Explainability API",
        "version": "2.0.0",
        "models_available": ["Google MuRIL Transformer", "Linear SVM Baseline"]
    })

@app.route("/api/analyze", methods=["POST"])
def analyze_text():
    """
    API Endpoint to analyze social media text/tweet.
    Request JSON: { "text": string, "model_choice": "muril" | "baseline" }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text", "").strip()
        model_choice = data.get("model_choice", "muril").strip()

        if not text:
            return jsonify({
                "error": "Input text is required.",
                "status": "bad_request"
            }), 400

        result = classifier.predict(text, model_choice=model_choice)
        return jsonify(result), 200

    except Exception as e:
        app.logger.error(f"Error during analysis: {str(e)}")
        return jsonify({
            "error": "An internal server error occurred.",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Cyberbullying Detection Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

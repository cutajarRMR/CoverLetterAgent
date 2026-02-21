"""
Flask backend for CoverLetterAgent web UI.
Run with: python app.py
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from cover_letter_agent import CoverLetterAgent
import os
import tempfile
import shutil

app = Flask(__name__, static_folder="static")
CORS(app)

# Singleton agent instance
agent = None


def get_agent():
    global agent
    if agent is None:
        agent = CoverLetterAgent()
    return agent


# ── Serve the frontend ──────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ── Upload & embed resume (PDF) ─────────────────────────────────────
@app.route("/api/upload-resume", methods=["POST"])
def upload_resume():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Resume must be a PDF file"}), 400

    tmp_dir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp_dir, file.filename)
        file.save(path)
        get_agent().enter_resume(path)
        return jsonify({"status": "ok", "filename": file.filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Upload & embed cover letters (DOCX) ─────────────────────────────
@app.route("/api/upload-cover-letters", methods=["POST"])
def upload_cover_letters():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    docx_files = [f for f in files if f.filename.lower().endswith(".docx")]
    if not docx_files:
        return jsonify({"error": "Please upload .docx files"}), 400

    tmp_dir = tempfile.mkdtemp()
    try:
        paths = []
        for f in docx_files:
            path = os.path.join(tmp_dir, f.filename)
            f.save(path)
            paths.append(path)

        get_agent().enter_cover_letter_files(paths)
        return jsonify({
            "status": "ok",
            "count": len(paths),
            "filenames": [f.filename for f in docx_files],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Generate cover letter from job description ──────────────────────
@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    jd = data.get("job_description", "").strip()
    if not jd:
        return jsonify({"error": "Job description is required"}), 400

    try:
        a = get_agent()
        a.build_cover_letter(jd)
        return jsonify({
            "cover_letter": a.cover_letter,
            "message": a.message,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Edit / iterate on the cover letter ──────────────────────────────
@app.route("/api/edit", methods=["POST"])
def edit():
    data = request.get_json(force=True)
    feedback = data.get("feedback", "").strip()
    cover_letter = data.get("cover_letter", "").strip() or None

    if not feedback:
        return jsonify({"error": "Feedback is required"}), 400

    try:
        a = get_agent()
        a.edit_cover_letter(feedback, cover_letter)
        return jsonify({
            "cover_letter": a.cover_letter,
            "message": a.message,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)

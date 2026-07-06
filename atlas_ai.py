"""
AtlasAI - An internal AI assistant for answering employee questions
about organizational documents.

Features:
    1. Admin Panel        -> add / remove / list organizational documents (Google Drive links)
    2. Learning FAQ        -> the tool "learns" new Q&A pairs over time and reuses them
    3. Ask AtlasAI          -> employees ask questions, AtlasAI answers from FAQ or documents
    4. Usage Report         -> tracks who asked what, and how well AtlasAI performed

Storage:
    Everything is stored locally as simple JSON files (no external DB needed):
        - documents.json    : list of registered documents (title, drive link, tags)
        - faq.json          : list of learned question/answer pairs
        - usage_log.json    : list of every question asked + how it was answered

Run:
    python atlas_ai.py
"""

import json
import os
import difflib
from datetime import datetime
from collections import Counter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DOCS_FILE = os.path.join(DATA_DIR, "documents.json")
FAQ_FILE = os.path.join(DATA_DIR, "faq.json")
LOG_FILE = os.path.join(DATA_DIR, "usage_log.json")

ADMIN_PASSWORD = "admin123"        # change this for real deployments
FAQ_MATCH_THRESHOLD = 0.72         # similarity cutoff for reusing a learned FAQ answer
DOC_MATCH_MIN_SCORE = 1            # minimum keyword overlap to consider a document relevant


# ---------------------------------------------------------------------------
# Small helpers for reading / writing JSON "tables"
# ---------------------------------------------------------------------------

def _load(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save(path, data):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _tokenize(text):
    return {w.strip(".,?!:;()[]").lower() for w in text.split() if len(w.strip(".,?!:;()[]")) > 2}


# ---------------------------------------------------------------------------
# Core AtlasAI engine
# ---------------------------------------------------------------------------

class AtlasAI:
    def __init__(self):
        self.documents = _load(DOCS_FILE)
        self.faq = _load(FAQ_FILE)
        self.usage_log = _load(LOG_FILE)

    # ------------------------- persistence -------------------------

    def _save_all(self):
        _save(DOCS_FILE, self.documents)
        _save(FAQ_FILE, self.faq)
        _save(LOG_FILE, self.usage_log)

    # ------------------------- admin panel -------------------------

    def add_document(self, title, drive_link, tags=""):
        """Register a new organizational document via its Google Drive link."""
        doc = {
            "id": len(self.documents) + 1,
            "title": title.strip(),
            "drive_link": drive_link.strip(),
            "tags": [t.strip().lower() for t in tags.split(",") if t.strip()],
            "added_on": _now(),
        }
        self.documents.append(doc)
        self._save_all()
        return doc

    def remove_document(self, doc_id):
        before = len(self.documents)
        self.documents = [d for d in self.documents if d["id"] != doc_id]
        self._save_all()
        return len(self.documents) < before

    def list_documents(self):
        return self.documents

    def teach_faq(self, question, answer):
        """Manually teach AtlasAI a question/answer pair (admin or auto-learning)."""
        entry = {
            "question": question.strip(),
            "answer": answer.strip(),
            "learned_on": _now(),
            "times_used": 0,
        }
        self.faq.append(entry)
        self._save_all()
        return entry

    def remove_faq(self, index):
        if 0 <= index < len(self.faq):
            self.faq.pop(index)
            self._save_all()
            return True
        return False

    def list_faq(self):
        return self.faq

    # ------------------------- answering logic -------------------------

    def _search_faq(self, question):
        """Find the closest learned FAQ entry using fuzzy string matching."""
        best_match, best_score = None, 0.0
        for entry in self.faq:
            score = difflib.SequenceMatcher(None, question.lower(), entry["question"].lower()).ratio()
            if score > best_score:
                best_match, best_score = entry, score
        if best_match and best_score >= FAQ_MATCH_THRESHOLD:
            return best_match, best_score
        return None, best_score

    def _search_documents(self, question):
        """Find relevant documents by keyword overlap with title/tags."""
        q_words = _tokenize(question)
        scored = []
        for doc in self.documents:
            doc_words = _tokenize(doc["title"]) | set(doc["tags"])
            overlap = len(q_words & doc_words)
            if overlap >= DOC_MATCH_MIN_SCORE:
                scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored]

    def ask(self, question, user="employee", auto_teach_answer=None):
        """
        Main entry point for employees asking AtlasAI a question.

        Resolution order:
            1. Check learned FAQ (fuzzy match)
            2. Check registered documents (keyword match) -> point to Drive links
            3. If nothing found and auto_teach_answer is provided, learn it on the spot
            4. Otherwise, mark as unanswered for admin follow-up
        """
        faq_hit, score = self._search_faq(question)
        if faq_hit:
            faq_hit["times_used"] += 1
            self._save_all()
            result = {
                "answer": faq_hit["answer"],
                "source": "faq",
                "confidence": round(score, 2),
            }
            self._log(user, question, result)
            return result

        matching_docs = self._search_documents(question)
        if matching_docs:
            top = matching_docs[:3]
            answer_lines = ["Here are documents that likely answer your question:"]
            for d in top:
                answer_lines.append(f"  - {d['title']}: {d['drive_link']}")
            result = {
                "answer": "\n".join(answer_lines),
                "source": "documents",
                "confidence": None,
                "matches": [d["title"] for d in top],
            }
            self._log(user, question, result)
            return result

        if auto_teach_answer:
            entry = self.teach_faq(question, auto_teach_answer)
            result = {"answer": entry["answer"], "source": "newly_learned", "confidence": 1.0}
            self._log(user, question, result)
            return result

        result = {
            "answer": "I don't have an answer for that yet. This has been flagged for an admin to review.",
            "source": "unanswered",
            "confidence": 0.0,
        }
        self._log(user, question, result)
        return result

    def _log(self, user, question, result):
        self.usage_log.append({
            "timestamp": _now(),
            "user": user,
            "question": question,
            "source": result["source"],
            "answered": result["source"] != "unanswered",
        })
        self._save_all()

    # ------------------------- usage report -------------------------

    def usage_report(self):
        total = len(self.usage_log)
        answered = sum(1 for e in self.usage_log if e["answered"])
        unanswered = total - answered
        by_source = Counter(e["source"] for e in self.usage_log)
        by_user = Counter(e["user"] for e in self.usage_log)
        top_questions = Counter(e["question"].lower() for e in self.usage_log).most_common(5)
        unanswered_questions = [e["question"] for e in self.usage_log if not e["answered"]]

        report = {
            "total_questions": total,
            "answered": answered,
            "unanswered": unanswered,
            "answer_rate": round((answered / total) * 100, 1) if total else 0.0,
            "by_source": dict(by_source),
            "by_user": dict(by_user),
            "top_questions": top_questions,
            "unanswered_questions": unanswered_questions,
        }
        return report

    def print_usage_report(self):
        r = self.usage_report()
        print("\n===== AtlasAI Usage Report =====")
        print(f"Total questions asked : {r['total_questions']}")
        print(f"Answered              : {r['answered']}")
        print(f"Unanswered            : {r['unanswered']}")
        print(f"Answer rate           : {r['answer_rate']}%")
        print("\nBreakdown by answer source:")
        for source, count in r["by_source"].items():
            print(f"  - {source}: {count}")
        print("\nBreakdown by user:")
        for user, count in r["by_user"].items():
            print(f"  - {user}: {count}")
        print("\nTop 5 most asked questions:")
        for q, count in r["top_questions"]:
            print(f"  - ({count}x) {q}")
        if r["unanswered_questions"]:
            print("\nUnanswered questions awaiting admin review:")
            for q in r["unanswered_questions"]:
                print(f"  - {q}")
        print("=================================\n")


# ---------------------------------------------------------------------------
# Simple CLI (menu-driven) so the tool is usable end-to-end
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Full Web API Configuration (Handles Employee Mode & Admin Panel)
# ---------------------------------------------------------------------------

from flask import Flask, request, jsonify

app = Flask(__name__)
atlas = AtlasAI()

# --- Employee Route ---
@app.route("/ask", methods=["POST"])
def ask_atlas():
    data = request.json or {}
    question = data.get("question", "").strip()
    user = data.get("user", "employee")
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
        
    result = atlas.ask(question, user=user)
    return jsonify(result)


# --- Admin Panel Routes ---

@app.route("/admin/documents", methods=["GET", "POST", "DELETE"])
def manage_documents():
    # Quick password check for admin security
    password = request.headers.get("Authorization")
    if password != ADMIN_PASSWORD:
        return jsonify({"error": "Unauthorized"}), 401

    if request.method == "GET":
        return jsonify(atlas.list_documents())

    elif request.method == "POST":
        data = request.json or {}
        title = data.get("title", "").strip()
        link = data.get("drive_link", "").strip()
        tags = data.get("tags", "")
        if not title or not link:
            return jsonify({"error": "Title and drive_link are required"}), 400
        doc = atlas.add_document(title, link, tags)
        return jsonify({"message": "Document added successfully", "document": doc})

    elif request.method == "DELETE":
        data = request.json or {}
        doc_id = data.get("doc_id")
        if not doc_id:
            return jsonify({"error": "doc_id is required"}), 400
        if atlas.remove_document(int(doc_id)):
            return jsonify({"message": "Document removed successfully"})
        return jsonify({"error": "Document not found"}), 404


@app.route("/admin/faq", methods=["GET", "POST", "DELETE"])
def manage_faq():
    password = request.headers.get("Authorization")
    if password != ADMIN_PASSWORD:
        return jsonify({"error": "Unauthorized"}), 401

    if request.method == "GET":
        return jsonify(atlas.list_faq())

    elif request.method == "POST":
        data = request.json or {}
        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()
        if not question or not answer:
            return jsonify({"error": "Question and answer are required"}), 400
        entry = atlas.teach_faq(question, answer)
        return jsonify({"message": "FAQ entry learned successfully", "entry": entry})

    elif request.method == "DELETE":
        data = request.json or {}
        index = data.get("index")
        if index is None:
            return jsonify({"error": "Index is required"}), 400
        if atlas.remove_faq(int(index)):
            return jsonify({"message": "FAQ entry removed successfully"})
        return jsonify({"error": "Index not found"}), 404


@app.route("/admin/report", methods=["GET"])
def get_report():
    password = request.headers.get("Authorization")
    if password != ADMIN_PASSWORD:
        return jsonify({"error": "Unauthorized"}), 401
        
    return jsonify(atlas.usage_report())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

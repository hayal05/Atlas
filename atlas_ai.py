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

def admin_panel(atlas: AtlasAI):
    pwd = input("Enter admin password: ").strip()
    if pwd != ADMIN_PASSWORD:
        print("Incorrect password.\n")
        return

    while True:
        print("\n--- AtlasAI Admin Panel ---")
        print("1. Add document (Google Drive link)")
        print("2. List documents")
        print("3. Remove document")
        print("4. Teach a new FAQ answer")
        print("5. List learned FAQ")
        print("6. Remove FAQ entry")
        print("7. View usage report")
        print("8. Back to main menu")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            title = input("Document title: ").strip()
            link = input("Google Drive link: ").strip()
            tags = input("Tags/keywords (comma separated, optional): ").strip()
            doc = atlas.add_document(title, link, tags)
            print(f"Added document #{doc['id']}: {doc['title']}")

        elif choice == "2":
            docs = atlas.list_documents()
            if not docs:
                print("No documents registered yet.")
            for d in docs:
                print(f"[{d['id']}] {d['title']} -> {d['drive_link']} (tags: {', '.join(d['tags']) or '-'})")

        elif choice == "3":
            try:
                doc_id = int(input("Document ID to remove: ").strip())
                print("Removed." if atlas.remove_document(doc_id) else "Document not found.")
            except ValueError:
                print("Please enter a valid numeric ID.")

        elif choice == "4":
            q = input("Question: ").strip()
            a = input("Answer: ").strip()
            atlas.teach_faq(q, a)
            print("FAQ learned.")

        elif choice == "5":
            faq = atlas.list_faq()
            if not faq:
                print("No FAQ entries yet.")
            for i, entry in enumerate(faq):
                print(f"[{i}] Q: {entry['question']}  ->  A: {entry['answer']}  (used {entry['times_used']}x)")

        elif choice == "6":
            try:
                idx = int(input("FAQ index to remove: ").strip())
                print("Removed." if atlas.remove_faq(idx) else "Index not found.")
            except ValueError:
                print("Please enter a valid numeric index.")

        elif choice == "7":
            atlas.print_usage_report()

        elif choice == "8":
            break
        else:
            print("Invalid option, try again.")


def employee_mode(atlas: AtlasAI):
    user = input("Enter your name (for usage tracking): ").strip() or "employee"
    print("\nAsk AtlasAI anything about company documents. Type 'exit' to return to the main menu.\n")
    while True:
        question = input(f"{user} > ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        result = atlas.ask(question, user=user)
        print(f"AtlasAI: {result['answer']}")

        if result["source"] == "unanswered":
            teach = input("  (Optional) Provide the correct answer to teach AtlasAI now, or press Enter to skip: ").strip()
            if teach:
                atlas.teach_faq(question, teach)
                print("  Thanks! AtlasAI has learned this for next time.")
        print()


def main():
    atlas = AtlasAI()
    print("=====================================")
    print("            Welcome to AtlasAI        ")
    print("  Your organization's document assistant")
    print("=====================================")

    while True:
        print("\n1. Ask AtlasAI a question (Employee mode)")
        print("2. Admin Panel")
        print("3. Exit")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            employee_mode(atlas)
        elif choice == "2":
            admin_panel(atlas)
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid option, try again.")


if __name__ == "__main__":
    main()

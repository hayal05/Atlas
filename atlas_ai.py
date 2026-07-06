"""
AtlasAI - An internal AI assistant for answering employee questions
about organizational documents.

Features:
    1. Admin Panel        -> add / remove / list organizational documents (Google Drive links)
    2. Learning FAQ       -> the tool "learns" new Q&A pairs over time and reuses them
    3. Ask AtlasAI        -> employees ask questions, AtlasAI answers from FAQ or documents
    4. Usage Report       -> tracks who asked what, and how well AtlasAI performed

Storage:
    Everything is stored locally as simple JSON files (no external DB needed):
        - documents.json    : list of registered documents (title, drive link, tags)
        - faq.json          : list of learned question/answer pairs
        - usage_log.json    : list of every question asked + how it was answered

Run:
    gunicorn atlas_ai:app
"""

import json
import os
import difflib
from datetime import datetime
from collections import Counter
from flask import Flask, request, jsonify, render_template_string

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
                "matches": [{"title": d["title"], "link": d["drive_link"]} for d in top],
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


# ---------------------------------------------------------------------------
# Flask Web Application & Premium Interface Layout
# ---------------------------------------------------------------------------

app = Flask(__name__)
atlas = AtlasAI()

# Minimalist white plate dashboard layout highlighted with vibrant orange accents
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AtlasAI Portal</title>
    <style>
        :root {
            --bg-color: #f8f9fa;
            --plate-color: #ffffff;
            --primary-orange: #ff6b35;
            --primary-orange-hover: #e85a24;
            --text-dark: #212529;
            --text-muted: #6c757d;
            --border-color: #e9ecef;
            --shadow: 0 4px 20px rgba(0,0,0,0.05);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: var(--bg-color); color: var(--text-dark); padding: 40px 20px; display: flex; justify-content: center; }
        .container { width: 100%; max-width: 900px; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }
        h1 { font-size: 24px; font-weight: 700; display: flex; align-items: center; gap: 10px; }
        h1 span { color: var(--primary-orange); }
        .nav-btn { background: none; border: 1px solid var(--border-color); padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 500; transition: all 0.2s; background-color: var(--plate-color); }
        .nav-btn:hover { border-color: var(--primary-orange); color: var(--primary-orange); }
        
        /* White Plate Cards */
        .plate-card { background-color: var(--plate-color); border-radius: 12px; padding: 30px; box-shadow: var(--shadow); border: 1px solid var(--border-color); margin-bottom: 24px; }
        .hidden { display: none; }
        
        .form-group { margin-bottom: 20px; }
        label { display: block; font-size: 14px; font-weight: 600; margin-bottom: 8px; color: var(--text-dark); }
        input[type="text"], input[type="password"] { width: 100%; padding: 12px; border: 1px solid var(--border-color); border-radius: 8px; background-color: #fafafa; font-size: 15px; transition: all 0.2s; }
        input[type="text"]:focus, input[type="password"]:focus { outline: none; border-color: var(--primary-orange); background-color: #fff; box-shadow: 0 0 0 3px rgba(255,107,53,0.1); }
        
        .btn { background-color: var(--primary-orange); color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; transition: background 0.2s; width: 100%; }
        .btn:hover { background-color: var(--primary-orange-hover); }
        
        /* Chat UI styling */
        .chat-output { margin-top: 24px; border-top: 1px solid var(--border-color); padding-top: 20px; display: none; }
        .response-box { background-color: #fff8f5; border-left: 4px solid var(--primary-orange); padding: 15px; border-radius: 4px; margin-top: 10px; white-space: pre-line; line-height: 1.6; }
        .doc-link { display: inline-block; background: #fff; border: 1px solid var(--border-color); padding: 8px 12px; margin-top: 8px; border-radius: 6px; text-decoration: none; color: var(--text-dark); font-size: 14px; transition: all 0.2s; }
        .doc-link:hover { border-color: var(--primary-orange); color: var(--primary-orange); }
        
        /* Admin Grid & Dashboard layouts */
        .admin-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media(max-width: 768px) { .admin-grid { grid-template-columns: 1fr; } }
        .stat-bar { display: flex; justify-content: space-between; padding: 12px; border-bottom: 1px solid var(--border-color); font-size: 14px; }
        .stat-bar span:first-child { font-weight: 500; }
        .data-list { max-height: 250px; overflow-y: auto; border: 1px solid var(--border-color); border-radius: 8px; margin-top: 10px; }
        .list-item { display: flex; justify-content: space-between; align-items: center; padding: 12px; border-bottom: 1px solid var(--border-color); font-size: 14px; }
        .list-item:last-child { border-bottom: none; }
        .delete-btn { background: none; border: none; color: #dc3545; cursor: pointer; font-size: 13px; font-weight: 600; }
        .delete-btn:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><span>Atlas</span>AI</h1>
            <button id="toggleViewBtn" class="nav-btn" onclick="toggleView()">Admin Access</button>
        </header>

        <!-- Employee Front-End Portal -->
        <main id="employeePortal" class="plate-card">
            <div class="form-group">
                <label for="empName">Your Name</label>
                <input type="text" id="empName" placeholder="Enter name for usage tracking" value="employee">
            </div>
            <div class="form-group">
                <label for="empQuestion">What is your question?</label>
                <input type="text" id="empQuestion" placeholder="Ask about company matching tags, files, policies..." onkeypress="if(event.key==='Enter') askAtlas()">
            </div>
            <button class="btn" onclick="askAtlas()">Consult AtlasAI</button>

            <div id="chatOutput" class="chat-output">
                <label>AtlasAI Response:</label>
                <div id="responseBox" class="response-box"></div>
                <div id="linksContainer"></div>
                
                <!-- On-the-spot Interactive Self-Learning Module -->
                <div id="teachModule" class="hidden" style="margin-top:20px; border-top:1px dashed var(--border-color); padding-top:15px;">
                    <p style="font-size:13px; color:var(--text-muted); margin-bottom:10px;">Help Atlas learn: If you know the verified answer, teach it directly below:</p>
                    <input type="text" id="teachAnswer" placeholder="Type verified answer here..." style="margin-bottom:10px;">
                    <button class="btn" style="background:#212529;" onclick="submitAutoTeach()">Teach AtlasAI</button>
                </div>
            </div>
        </main>

        <!-- Complete Administrative Suite -->
        <main id="adminPortal" class="hidden">
            <div id="adminAuth" class="plate-card">
                <div class="form-group">
                    <label for="adminPass">Admin Password</label>
                    <input type="password" id="adminPass" placeholder="Enter system security credential">
                </div>
                <button class="btn" onclick="verifyAdmin()">Authenticate</button>
            </div>

            <div id="adminDashboard" class="hidden">
                <!-- Row 1: System Analytics & Reports -->
                <div class="plate-card">
                    <h3 style="margin-bottom:15px;">System Diagnostic & Utilization Profile</h3>
                    <div id="reportMetrics"></div>
                </div>

                <div class="admin-grid">
                    <!-- Column 2: Document Index Management -->
                    <div class="plate-card">
                        <h3 style="margin-bottom:15px;">Register Document Link</h3>
                        <div class="form-group"><input type="text" id="docTitle" placeholder="Document Title"></div>
                        <div class="form-group"><input type="text" id="docLink" placeholder="Google Drive Link"></div>
                        <div class="form-group"><input type="text" id="docTags" placeholder="Tags (comma separated)"></div>
                        <button class="btn" onclick="addDocument()">Bind Document</button>
                        
                        <h4 style="margin-top:20px; font-size:14px;">Active Indexed Repositories</h4>
                        <div id="docList" class="data-list"></div>
                    </div>

                    <!-- Column 3: Learned FAQ Repository Knowledge Tuning -->
                    <div class="plate-card">
                        <h3 style="margin-bottom:15px;">Teach Database Manually</h3>
                        <div class="form-group"><input type="text" id="faqQ" placeholder="Target Phrase/Question"></div>
                        <div class="form-group"><input type="text" id="faqA" placeholder="Response Payload"></div>
                        <button class="btn" onclick="addFaq()">Inject Entry</button>
                        
                        <h4 style="margin-top:20px; font-size:14px;">Learned Knowledge Bases</h4>
                        <div id="faqList" class="data-list"></div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        let currentAuthToken = "";
        let globalLastQuestion = "";

        function toggleView() {
            const emp = document.getElementById("employeePortal");
            const adm = document.getElementById("adminPortal");
            const btn = document.getElementById("toggleViewBtn");
            if (adm.classList.contains("hidden")) {
                adm.classList.remove("hidden");
                emp.classList.add("hidden");
                btn.innerText = "Employee Access";
            } else {
                adm.classList.add("hidden");
                emp.classList.remove("hidden");
                btn.innerText = "Admin Access";
            }
        }

        async function askAtlas() {
            const question = document.getElementById("empQuestion").value.trim();
            const user = document.getElementById("empName").value.trim() || "employee";
            if(!question) return;

            globalLastQuestion = question;
            const res = await fetch("/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question, user })
            });
            const data = await res.json();
            
            document.getElementById("chatOutput").style.display = "block";
            document.getElementById("responseBox").innerText = data.answer;
            
            const linksContainer = document.getElementById("linksContainer");
            linksContainer.innerHTML = "";
            if(data.matches) {
                data.matches.forEach(d => {
                    const a = document.createElement("a");
                    a.className = "doc-link";
                    a.href = d.link;
                    a.target = "_blank";
                    a.innerText = "📁 Open Drive: " + d.title;
                    linksContainer.appendChild(a);
                });
            }

            if(data.source === "unanswered") {
                document.getElementById("teachModule").classList.remove("hidden");
            } else {
                document.getElementById("teachModule").classList.add("hidden");
            }
        }

        async function submitAutoTeach() {
            const answer = document.getElementById("teachAnswer").value.trim();
            if(!answer || !globalLastQuestion) return;

            await fetch("/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: globalLastQuestion,
                    user: document.getElementById("empName").value.trim() || "employee",
                    auto_teach_answer: answer
                })
            });
            
            document.getElementById("teachAnswer").value = "";
            document.getElementById("teachModule").classList.add("hidden");
            document.getElementById("responseBox").innerText = "Success! Core engine has ingested this entry into the knowledge cluster.";
        }

        async function verifyAdmin() {
            const pass = document.getElementById("adminPass").value.trim();
            const res = await fetch("/admin/report", { headers: { "Authorization": pass } });
            if(res.status === 200) {
                currentAuthToken = pass;
                document.getElementById("adminAuth").classList.add("hidden");
                document.getElementById("adminDashboard").classList.remove("hidden");
                loadAdminDashboard();
            } else {
                alert("Authentication failed.");
            }
        }

        async function loadAdminDashboard() {
            // Metrics
            const repRes = await fetch("/admin/report", { headers: { "Authorization": currentAuthToken } });
            const rep = await repRes.json();
            document.getElementById("reportMetrics").innerHTML = `
                <div class="stat-bar"><span>Total Consultations Indexed</span><span>${rep.total_questions}</span></div>
                <div class="stat-bar"><span>Successful Resolutions</span><span>${rep.answered}</span></div>
                <div class="stat-bar"><span>Unresolved/Flagged Queries</span><span>${rep.unanswered}</span></div>
                <div class="stat-bar"><span>Precision System Rating</span><span>${rep.answer_rate}%</span></div>
            `;

            // Active Documents Index
            const docRes = await fetch("/admin/documents", { headers: { "Authorization": currentAuthToken } });
            const docs = await docRes.json();
            const docList = document.getElementById("docList");
            docList.innerHTML = docs.map(d => `
                <div class="list-item">
                    <div><strong>${d.title}</strong><br><small style="color:var(--text-muted)">${d.drive_link.substring(0,40)}...</small></div>
                    <button class="delete-btn" onclick="deleteDoc(${d.id})">Purge</button>
                </div>
            `).join("");

            // Active FAQ System
            const faqRes = await fetch("/admin/faq", { headers: { "Authorization": currentAuthToken } });
            const faqs = await faqRes.json();
            const faqList = document.getElementById("faqList");
            faqList.innerHTML = faqs.map((f, index) => `
                <div class="list-item">
                    <div><strong>Q: ${f.question}</strong><br><small style="color:var(--text-muted)">A: ${f.answer}</small></div>
                    <button class="delete-btn" onclick="deleteFaq(${index})">Purge</button>
                </div>
            `).join("");
        }

        async function addDocument() {
            const title = document.getElementById("docTitle").value.trim();
            const drive_link = document.getElementById("docLink").value.trim();
            const tags = document.getElementById("docTags").value.trim();
            if(!title || !drive_link) return;

            await fetch("/admin/documents", {
                method: "POST",
                headers: { "Content-Type": "application/json", "Authorization": currentAuthToken },
                body: JSON.stringify({ title, drive_link, tags })
            });
            document.getElementById("docTitle").value = "";
            document.getElementById("docLink").value = "";
            document.getElementById("docTags").value = "";
            loadAdminDashboard();
        }

        async function deleteDoc(id) {
            await fetch("/admin/documents", {
                method: "DELETE",
                headers: { "Content-Type": "application/json", "Authorization": currentAuthToken },
                body: JSON.stringify({ doc_id: id })
            });
            loadAdminDashboard();
        }

        async function addFaq() {
            const question = document.getElementById("faqQ").value.trim();
            const answer = document.getElementById("faqA").value.trim();
            if(!question || !answer) return;

            await fetch("/admin/faq", {
                method: "POST",
                headers: { "Content-Type": "application/json", "Authorization": currentAuthToken },
                body: JSON.stringify({ question, answer })
            });
            document.getElementById("faqQ").value = "";
            document.getElementById("faqA").value = "";
            loadAdminDashboard();
        }

        async function deleteFaq(index) {
            await fetch("/admin/faq", {
                method: "DELETE",
                headers: { "Content-Type": "application/json", "Authorization": currentAuthToken },
                body: JSON.stringify({ index })
            });
            loadAdminDashboard();
        }
    </script>
</body>
</html>
"""

# --- Server Routes ---

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/ask", methods=["POST"])
def ask_atlas():
    data = request.json or {}
    question = data.get("question", "").strip()
    user = data.get("user", "employee")
    auto_teach_answer = data.get("auto_teach_answer")
    
    if not question:
        return jsonify({"error": "Question parameter omitted"}), 400
        
    result = atlas.ask(question, user=user, auto_teach_answer=auto_teach_answer)
    return jsonify(result)


@app.route("/admin/documents", methods=["GET", "POST", "DELETE"])
def manage_documents():
    password = request.headers.get("Authorization")
    if password != ADMIN_PASSWORD:
        return jsonify({"error": "Unauthorized Access Validation Fault"}), 401

    if request.method == "GET":
        return jsonify(atlas.list_documents())

    elif request.method == "POST":
        data = request.json or {}
        title = data.get("title", "").strip()
        link = data.get("drive_link", "").strip()
        tags = data.get("tags", "")
        if not title or not link:
            return jsonify({"error": "Missing key elements"}), 400
        doc = atlas.add_document(title, link, tags)
        return jsonify({"message": "Document entry created", "document": doc})

    elif request.method == "DELETE":
        data = request.json or {}
        doc_id = data.get("doc_id")
        if doc_id is None:
            return jsonify({"error": "Target mapping ID null"}), 400
        if atlas.remove_document(int(doc_id)):
            return jsonify({"message": "Target document cleared"})
        return jsonify({"error": "Resource missing"}), 404


@app.route("/admin/faq", methods=["GET", "POST", "DELETE"])
def manage_faq():
    password = request.headers.get("Authorization")
    if password != ADMIN_PASSWORD:
        return jsonify({"error": "Unauthorized Access Validation Fault"}), 401

    if request.method == "GET":
        return jsonify(atlas.list_faq())

    elif request.method == "POST":
        data = request.json or {}
        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()
        if not question or not answer:
            return jsonify({"error": "Payload incomplete"}), 400
        entry = atlas.teach_faq(question, answer)
        return jsonify({"message": "Faq dataset updated", "entry": entry})

    elif request.method == "DELETE":
        data = request.json or {}
        index = data.get("index")
        if index is None:
            return jsonify({"error": "Target node reference null"}), 400
        if atlas.remove_faq(int(index)):
            return jsonify({"message": "Knowledge target purged"})
        return jsonify({"error": "Reference node out of bounds"}), 404


@app.route("/admin/report", methods=["GET"])
def get_report():
    password = request.headers.get("Authorization")
    if password != ADMIN_PASSWORD:
        return jsonify({"error": "Unauthorized Access Validation Fault"}), 401
        
    return jsonify(atlas.usage_report())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

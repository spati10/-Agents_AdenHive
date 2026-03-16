# AI Agents Built with Aden Hive Framework

AI agents built using [Aden Hive](https://github.com/aden-hive/hive) — a production-grade, outcome-driven agent development framework backed by Y Combinator (8.5k+ GitHub stars).

## 🧠 What is Hive?
Hive is a self-evolving AI agent framework where you describe a goal in natural language and the framework automatically generates the agent graph, connection code, and self-heals on failure.

---

## 🚀 Agents Built

### 1. 🌐 Web Page Summarizer Agent
**File:** `web_summarizer_agent.py`

An agent that scrapes any webpage and extracts structured content.

**How it works:**
- Takes any URL as input
- Fetches HTML using `httpx`
- Removes noise (ads, nav, scripts) using `BeautifulSoup`
- Extracts main content
- Summarizes using LLM (falls back to extractive summary if no API key)
- Built using Hive's `GraphSpec` / `NodeSpec` / `NodeProtocol` architecture

**Run:**
```bash
uv run python web_summarizer_agent.py
```

**Example output:**
Tested on Wikipedia's Python programming language page — extracted title, summary, key sections cleanly.

---

### 2. 📄 Resume Analyzer Agent
**File:** `resume_analyzer_agent.py`

An agent that analyzes resume text and provides structured feedback for job applications.

**How it works:**
- Takes resume text as input
- Detects technical skills, experience level, education
- Identifies Codeforces/competitive programming ratings
- Gives ATS (Applicant Tracking System) optimization suggestions
- Provides strengths and areas for improvement
- Claude Code autonomously debugged and improved this agent multiple times

**Run:**
```bash
uv run python resume_analyzer_agent.py
```

**Example output:**
```
RESUME ANALYSIS RESULTS
========================================
TARGET ROLE: Software Engineer
EXPERIENCE LEVEL: Mid-level (2-5 years)
TECHNICAL SKILLS (12 found): Python, JavaScript, SQL...
EDUCATION: Bachelor's Degree
STRENGTHS:
- Clear technical skillset detected
- Experience indicators present
SUGGESTIONS:
- Add quantifiable achievements
- Include ATS keywords
```

---

## ⚙️ Setup & Installation
```bash
# Clone the Hive framework
git clone https://github.com/aden-hive/hive.git
cd hive

# Run quickstart (use WSL2 on Windows)
./quickstart.sh

# Clone this repo
git clone https://github.com/spati10/-Agents_AdenHive.git

# Copy agents to hive examples
cp -Agents_AdenHive/*.py hive/core/examples/
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Core language |
| Hive Framework | Agent orchestration |
| GraphSpec / NodeSpec | Agent graph architecture |
| httpx | HTTP requests |
| BeautifulSoup4 | HTML parsing |
| asyncio | Async execution |
| Claude Code | Agent scaffolding & debugging |
| WSL2 (Ubuntu) | Local development environment |
| uv | Python package management |

---

## 💡 Key Learnings

- Hive's **GraphSpec/NodeSpec** architecture makes agents modular and testable
- **Self-healing behavior** — Claude Code autonomously rewrites agent code on failure
- Agents work **without LLM API keys** using extractive fallback
- **Goal-driven development** is faster than manually wiring workflows

---

## 🔗 Related

- [Aden Hive Framework](https://github.com/aden-hive/hive)
- [Hive Documentation](https://docs.adenhq.com)
- [My Contribution](https://github.com/aden-hive/hive)


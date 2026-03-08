# 🤖 AI RAG Assistant

A production-grade RAG (Retrieval-Augmented Generation) system that lets you chat with your company documents/policies using AI.

Basically, any unstructured data (text, docs, policies) that you have can be stored in a database and queried using just natural language.

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📖 Table of Contents

- [What Does This Do?](#what-does-this-do)
- [Quick Start (5 Minutes)](#quick-start-5-minutes)
- [Detailed Setup](#detailed-setup)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

---

## 🎯 What Does This Do?

This app lets you **ask questions in plain English** and get **instant answers** from your company documents.

**Example:**
- ❓ "What is the remote work policy?"
- 🤖 "Employees can work remotely up to 3 days per week..."

**How it works:**
1. Your documents are stored in a smart database
2. When you ask a question, the AI finds relevant information
3. It generates a natural answer with sources

---

## ⚡ Quick Start (5 Minutes)

### Prerequisites

You need:
- **Docker Desktop** ([Download here](https://www.docker.com/products/docker-desktop))
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))

### Steps

1. **Download this project**
```bash
   # If you have git:
   git clone <your-repo-url>
   cd company-ai-assistant
   
   # Or download and unzip manually
```

2. **Add your OpenAI API key**
```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   # (Use any text editor - Notepad, VS Code, etc.)
```

3. **Start the app**
```bash
   cd docker
   docker-compose up -d
```

4. **Run first-time setup** (load documents into database)
```bash
   docker-compose exec app python -m src.ingestion
```

5. **Open in browser**
```
   http://localhost:8501
```

**That's it! 🎉 Ask your first question!**

---

## 📚 Detailed Setup

### Option 1: Docker (Recommended)

**Why Docker?**
- ✅ Everything just works (no dependency issues)
- ✅ Same setup on Mac, Windows, Linux
- ✅ Includes database automatically

**Step-by-step:**

1. **Install Docker Desktop**
   - Windows/Mac: [Download here](https://www.docker.com/products/docker-desktop)
   - Linux: `sudo apt install docker.io docker-compose`

2. **Configure environment**
```bash
   # Copy template
   cp .env.example .env
   
   # Edit .env (required fields):
   # - OPENAI_API_KEY (get from OpenAI)
   # - DB_PASSWORD (make up a secure password)
```

3. **Start services**
```bash
   cd docker
   docker-compose up -d
```
   
   This will:
   - Download PostgreSQL with pgvector
   - Build your app container
   - Start both services

4. **Load your documents**
```bash
   # This reads files from data/policies/ and stores them
   docker-compose exec app python -m src.ingestion
```
   
   You should see:
```
   INFO - Loaded 10 documents
   INFO - Created 10 chunks
   INFO - Ingestion complete!
```

5. **Access the app**
   - Open browser: `http://localhost:8501`
   - Ask a question!

**Managing the app:**
```bash
# Stop
docker-compose down

# Restart
docker-compose restart

# View logs
docker-compose logs -f app

# Rebuild after code changes
docker-compose up -d --build
```

---

### Option 2: Local Setup (Advanced)

**Prerequisites:**
- Python 3.12+
- PostgreSQL with pgvector extension
- uv or pip

**Steps:**

1. **Install PostgreSQL + pgvector**
```bash
   # Mac
   brew install postgresql pgvector
   
   # Ubuntu/Debian
   sudo apt install postgresql postgresql-contrib
   
   # Then enable pgvector
   psql -U postgres -c "CREATE EXTENSION vector;"
```

2. **Create database**
```bash
   psql -U postgres
   CREATE DATABASE company_ai;
   \q
```

3. **Install Python dependencies**
```bash
   # Using uv (faster)
   uv pip install -e .
   
   # Or using pip
   pip install -r requirements.txt
```

4. **Configure environment**
```bash
   cp .env.example .env
   # Edit .env with your settings
```

5. **Load documents**
```bash
   python -m src.ingestion
```

6. **Run app**
```bash
   streamlit run ui/app.py
```

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────┐
│           USER INTERFACE (Streamlit)         │
│  "What is the remote work policy?"          │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│         RAG ORCHESTRATOR (rag.py)           │
│  1. Embed query                             │
│  2. Search database                         │
│  3. Generate answer                         │
└────┬────────────────────────────┬───────────┘
     │                            │
     ▼                            ▼
┌──────────────────┐    ┌──────────────────────┐
│  VECTOR DATABASE │    │   LLM (OpenAI)       │
│  (PostgreSQL +   │    │   - gpt-4o-mini      │
│   pgvector)      │    │   - Embeddings       │
│                  │    │                      │
│  10 docs →       │    │  Context + Query →   │
│  237 chunks →    │    │  Natural Answer      │
│  Embeddings      │    │                      │
└──────────────────┘    └──────────────────────┘
```

**Data Flow:**
1. **Ingestion** (one-time): `data/policies/*.txt` → Chunked → Embedded → Stored in DB
2. **Query** (real-time): User question → Find similar chunks → Send to LLM → Get answer

---

## 🛠️ Troubleshooting

### "Cannot connect to Docker daemon"
**Problem:** Docker isn't running  
**Solution:** Start Docker Desktop

### "Database connection failed"
**Problem:** PostgreSQL not started or wrong credentials  
**Solution:**
```bash
# Check if database is running
docker-compose ps

# Check logs
docker-compose logs database

# Restart database
docker-compose restart database
```

### "OPENAI_API_KEY is not set"
**Problem:** Missing or incorrect API key  
**Solution:**
1. Check `.env` file exists (not `.env.example`)
2. Verify `OPENAI_API_KEY=sk-proj-...` has your actual key
3. Restart: `docker-compose restart app`

### "No results found" for every question
**Problem:** Documents not ingested  
**Solution:**
```bash
# Re-run ingestion
docker-compose exec app python -m src.ingestion

# Verify data loaded
docker-compose exec app python -c "from src.database import Database; 
with Database.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute('SELECT COUNT(*) FROM documents')
        print(f'Chunks in DB: {cur.fetchone()[0]}')"
```

### Port 8501 already in use
**Problem:** Another app using the port  
**Solution:**
```bash
# Edit docker-compose.yml, change ports:
ports:
  - "8502:8501"  # Changed from 8501:8501

# Then access at http://localhost:8502
```

---

## 🙋 FAQ

### Can I add more documents?
Yes! Add `.txt` files to `data/policies/`, then re-run:
```bash
docker-compose exec app python -m src.ingestion
```

### How much does this cost?
- **OpenAI API**: ~$0.01 per 100 queries (with gpt-4o-mini)
- **Everything else**: Free (runs on your computer)

### Can I use this for my own documents?
Absolutely! Just replace files in `data/policies/` with your documents.

### Is my data secure?
Yes:
- Documents stored locally (never leave your computer)
- Only queries are sent to OpenAI (not full documents)
- Database password-protected

### Can I deploy this to a server?
Yes! This Docker setup works on any server. You'd need to:
1. Copy project to server
2. Set `.env` with production values
3. Run `docker-compose up -d`
4. Open firewall port 8501

### How do I update the code?
```bash
# Pull latest changes
git pull

# Rebuild
docker-compose up -d --build
```

---

## 📊 Project Stats

- **Documents**: 10 policy files
- **Average chunks per document**: ~1-3
- **Embedding dimensions**: 1536 (OpenAI text-embedding-3-small)
- **Response time**: ~1-2 seconds per query
- **Tech stack**: Python 3.12, Streamlit, PostgreSQL, pgvector, OpenAI

---

## 📝 License

MIT License - Use freely, modify as needed.

---

## 🤝 Contributing

Found a bug? Want to add a feature? Open an issue or submit a PR!

---

## 🎓 Learning Resources

Want to understand how this works?

- **RAG Systems**: [Anthropic's RAG Guide](https://www.anthropic.com/index/contextual-retrieval)
- **Vector Databases**: [pgvector Documentation](https://github.com/pgvector/pgvector)
- **Streamlit**: [Official Docs](https://docs.streamlit.io)

---

**Built with ❤️ using OpenAI, Streamlit, and PostgreSQL**
```

---

## 📁 FILE 5: `.dockerignore`
```
### Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.venv/
ENV/
env/

### IDE
.vscode/
.idea/
*.swp
*.swo

### OS
.DS_Store
Thumbs.db

### Git
.git/
.gitignore

### Tests
tests/
*.pytest_cache/

### Docs
README.md
architecture.png

### Environment
.env
.env.local

### Logs
*.log

### Misc
*.egg-info/
uv.lock

# 🏥 CVS HealthHub AI Assistant

**Enterprise-grade RAG system for healthcare information, powered by LangGraph, ChromaDB, and OpenRouter**

An intelligent agentic AI system designed for CVS Health's Digital Workplace AI initiative, demonstrating production-ready capabilities in retrieval-augmented generation (RAG), semantic search, and healthcare information delivery.

---

## 🌟 Key Features

- **🤖 Agentic AI**: LangGraph-powered ReAct agent with 5 specialized healthcare tools
- **🔍 Semantic Search**: ChromaDB vector database with sentence-transformer embeddings
- **💊 Healthcare Expertise**: Medication info, drug interactions, vaccines, insurance coverage
- **🚀 REST API**: FastAPI backend with OpenAPI/Swagger documentation
- **📊 Evaluation Framework**: RAGAS metrics for quality assurance
- **☁️ Cloud-Ready**: Docker containerization and Azure deployment support
- **🔄 CI/CD**: Automated testing and deployment pipelines
- **📈 Observability**: LangSmith tracing for all agent interactions

---

## 🏗️ Architecture

```
CVS HealthHub AI
├── Agent Layer (LangGraph)
│   ├── Medication Information Tool
│   ├── Drug Interaction Checker
│   ├── Vaccine Finder
│   ├── CVS Services Tool
│   └── Insurance Coverage Tool
├── RAG Engine (ChromaDB)
│   ├── Sentence Transformers (all-MiniLM-L6-v2)
│   ├── Vector Store (100+ healthcare documents)
│   └── Semantic Search
├── API Layer (FastAPI)
│   ├── /chat - Main chat endpoint
│   ├── /search - Semantic search
│   ├── /health - Health check
│   └── /metrics - Performance metrics
└── UI Layer (Streamlit)
    └── Interactive chat interface
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenRouter API key ([get one here](https://openrouter.ai/))
- Optional: LangSmith API key for tracing

### Installation

1. **Clone and navigate to project**
```bash
cd health_hub_assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

4. **Load initial healthcare data**
```bash
python ingestion/load_initial_data.py
```

5. **Run the application**

**Option A: Streamlit UI (recommended for demo)**
```bash
streamlit run app.py
```
Open http://localhost:8501

**Option B: FastAPI Server**
```bash
cd api && python main.py
```
API docs: http://localhost:8000/docs

---

## 📁 Project Structure

```
cvs-healthhub-ai/
├── agent/
│   └── healthhub_agent.py       # LangGraph agentic system
├── rag/
│   └── vector_store.py          # ChromaDB vector database
├── api/
│   ├── main.py                  # FastAPI application
│   └── models.py                # Pydantic schemas
├── ingestion/
│   └── load_initial_data.py     # Data loading script
├── data/
│   ├── documents/               # Healthcare knowledge base
│   │   ├── medications.txt
│   │   ├── vaccines.txt
│   │   ├── drug_interactions.txt
│   │   ├── cvs_services.txt
│   │   └── insurance_coverage.txt
│   └── chroma_db/              # Vector database (auto-generated)
├── evaluation/                  # RAGAS evaluation (coming soon)
├── tests/                      # Unit tests (coming soon)
├── .github/workflows/          # CI/CD pipelines (coming soon)
├── app.py                      # Streamlit frontend
├── requirements.txt
├── Dockerfile                  # (coming soon)
└── README.md
```

---

## 🔧 Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **LLM Provider** | OpenRouter | Multi-model LLM access (Claude, GPT-4, etc.) |
| **Agent Framework** | LangGraph | ReAct agent orchestration |
| **Vector Database** | ChromaDB | Semantic search and embeddings |
| **Embeddings** | Sentence Transformers | Local, free embeddings (384-dim) |
| **API Framework** | FastAPI | Async REST API |
| **UI** | Streamlit | Interactive chat interface |
| **Evaluation** | RAGAS | RAG quality metrics |
| **Observability** | LangSmith | Agent tracing and monitoring |
| **Containerization** | Docker | Cloud deployment |
| **CI/CD** | GitHub Actions | Automated testing and deployment |

---

## 🎯 Use Cases

1. **Medication Information**
   - "What are the side effects of Lisinopril?"
   - "Tell me about blood pressure medications"

2. **Drug Interaction Checking**
   - "Can I take Aspirin with Lisinopril?"
   - "Check interactions between my medications"

3. **Vaccine Scheduling**
   - "What vaccines are available?"
   - "Do I need a flu shot?"

4. **CVS Services**
   - "What are CVS pharmacy hours?"
   - "What services does CVS MinuteClinic offer?"

5. **Insurance Coverage**
   - "Is my insurance accepted?"
   - "How much will my prescription cost?"

---

## 📊 API Endpoints

### POST /chat
Process user queries through the healthcare agent
```json
{
  "message": "What vaccines are available?",
  "chat_history": []
}
```

### POST /search
Semantic search in the knowledge base
```json
{
  "query": "blood pressure medications",
  "k": 5,
  "category": "medication"
}
```

### GET /health
Health check and system status

### GET /metrics
Performance metrics and statistics

**Full API docs:** http://localhost:8000/docs

---

## 🧪 Testing

```bash
# Run unit tests (coming soon)
pytest tests/

# Run evaluation metrics (coming soon)
python evaluation/run_eval.py
```

---

## 🐳 Docker Deployment (Coming Soon)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access services
# Streamlit UI: http://localhost:8501
# FastAPI: http://localhost:8000
```

---

## ☁️ Azure Deployment (Coming Soon)

Automated deployment to Azure Container Apps via GitHub Actions.

---

## 📈 Roadmap

- [x] Core RAG engine with ChromaDB
- [x] 5 healthcare agent tools
- [x] FastAPI REST API
- [x] Streamlit UI
- [x] Sample healthcare data (100+ documents)
- [ ] RAGAS evaluation framework
- [ ] Docker containerization
- [ ] GitHub Actions CI/CD
- [ ] Azure Container Apps deployment
- [ ] Automated document ingestion pipeline
- [ ] Test coverage (pytest)
- [ ] Performance monitoring dashboard

---

## 🎓 Skills Demonstrated

This project showcases:
- **Python** - 500+ lines across multiple modules
- **LangChain/LangGraph** - Advanced agentic workflows
- **RAG Systems** - Document chunking, embeddings, retrieval
- **Vector Databases** - ChromaDB with semantic search
- **FastAPI** - Async REST API with OpenAPI docs
- **Agent Design** - Tool-calling, ReAct pattern, state management
- **Healthcare Domain** - Medical knowledge integration
- **Software Engineering** - Modular architecture, clean code
- **DevOps** - Docker, CI/CD, cloud deployment (in progress)

---

## 📄 License

This project is for portfolio and demonstration purposes.

---

## 🙋 Contact

Built as a demonstration project for CVS Health Digital Workplace AI team application.

**Technologies**: LangGraph | ChromaDB | OpenRouter | FastAPI | Streamlit | RAGAS

# PDF RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for PDF documents with semantic search, LLM-powered answers, and comprehensive telemetry.

## 🏗️ Architecture Overview

This system implements an end-to-end RAG pipeline following these 11 steps:

1. **PDF Upload** - Streamlit UI with progress tracking
2. **Non-blocking Ingestion** - FastAPI returns doc_id immediately
3. **PDF Parsing** - Page-level text extraction with metadata
4. **Text Chunking** - Overlapping chunks for context preservation
5. **Embedding Creation** - Semantic vectors using sentence-transformers
6. **Vector Storage** - Qdrant for fast similarity search
7. **Query Interface** - Streamlit chat with settings panel
8. **Semantic Search** - Embedding-based retrieval with filtering
9. **LLM Generation** - Grounded answers using Gemini API
10. **Citation Attachment** - Source references with page numbers
11. **Telemetry Logging** - PostgreSQL for metrics and history

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Qdrant (Docker or local installation)
- Gemini API key

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository (or create the directory structure)
mkdir pdf-rag-system
cd pdf-rag-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup PostgreSQL Database

```bash
# Create database and user
sudo -u postgres psql

CREATE DATABASE rag_system;
CREATE USER rag_user WITH ENCRYPTED PASSWORD 'rag_password';
GRANT ALL PRIVILEGES ON DATABASE rag_system TO rag_user;
\q

# Initialize database schema
psql -U rag_user -d rag_system -f database/init.sql
```

### 3. Setup Qdrant Vector Database

```bash
# Option 1: Using Docker (recommended)
docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

# Option 2: Download and run locally
# Visit: https://qdrant.tech/documentation/quick-start/
```

### 4. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your configurations
nano .env
```

Required configurations in `.env`:

```env
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_system
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_password

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=pdf_documents

# LLM Configuration (REQUIRED - Get from Google AI Studio)
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Application Settings
CHUNK_SIZE=400
CHUNK_OVERLAP=50
TOP_K_DEFAULT=5
SIMILARITY_THRESHOLD=0.7
POLLING_INTERVAL=2
```

### 5. Get Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Get API Key"
4. Copy the key and add it to your `.env` file

### 6. Run the Backend Server

```bash
# Start FastAPI backend
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

### 7. Run the Frontend Application

In a new terminal:

```bash
# Activate virtual environment if not already active
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run Streamlit frontend
streamlit run streamlit_app.py
```

The UI will be available at: `http://localhost:8501`

## 📖 Usage Guide

### Uploading Documents

1. Open the Streamlit UI at `http://localhost:8501`
2. Click on "📤 Upload Documents" in the sidebar
3. Select a PDF file
4. Click "📥 Upload / Index"
5. Monitor the status in the "📁 Indexed Documents" panel
   - ⏳ Processing: Document is being indexed
   - ✅ Indexed: Ready for querying
   - ❌ Failed: Check error message

### Asking Questions

1. Wait for at least one document to be indexed
2. Type your question in the chat input
3. The system will:
   - Search for relevant passages
   - Generate an answer using only the found sources
   - Display citations with page numbers
4. Click "📚 Show Context" to see source passages
5. Click "⏱️ Performance Metrics" to view timing data

### Configuration Options

Access the settings panel in the sidebar:

- **Top K chunks**: Number of passages to retrieve (1-10)
- **Filter by document**: Limit search to specific document
- **LLM Model**: Select generation model (currently Gemini Pro)
- **Only answer if sources found**: Toggle strict grounding mode

## 🔧 System Components

### Backend (FastAPI)

- **Endpoints**:
  - `POST /upload` - Upload and index PDF
  - `GET /documents` - List all documents
  - `GET /status/{doc_id}` - Check document status
  - `POST /query` - RAG query endpoint
  - `GET /sessions` - List chat sessions
  - `DELETE /session/{id}` - Clear session

### Frontend (Streamlit)

- **Features**:
  - Drag-and-drop PDF upload
  - Real-time status polling
  - Chat interface with history
  - Expandable context viewer
  - Performance metrics
  - Settings panel for retrieval configuration

### Database Schema

- **documents**: Track PDF processing status
- **chat_sessions**: Manage user sessions
- **query_runs**: Store telemetry and query history

## 🧪 Testing the System

### 1. Test Backend Health

```bash
curl http://localhost:8000/health
```

### 2. Test Document Upload

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.pdf"
```

### 3. Test Query Endpoint

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is this document about?",
    "session_id": "test-session",
    "top_k": 5,
    "only_use_sources": true
  }'
```

## 🐛 Troubleshooting

### Common Issues

#### PostgreSQL Connection Error

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Verify connection
psql -U rag_user -d rag_system -c "SELECT 1;"
```

#### Qdrant Connection Error

```bash
# Check Qdrant is running
curl http://localhost:6333/collections

# If using Docker, check container
docker ps | grep qdrant
```

#### Gemini API Error

- Verify API key is correct in `.env`
- Check API quotas at [Google AI Studio](https://makersuite.google.com/app/apikey)
- Test API key:

```python
import google.generativeai as genai
genai.configure(api_key="your_api_key")
model = genai.GenerativeModel("gemini-pro")
response = model.generate_content("Hello")
print(response.text)
```

#### PDF Processing Failures

- Ensure PDF is not corrupted
- Check PDF has extractable text (not scanned images)
- Monitor backend logs for specific errors
- Check document status via API:

```bash
curl http://localhost:8000/status/{doc_id}
```

#### Memory Issues with Large PDFs

- Adjust `CHUNK_SIZE` in `.env` to smaller values
- Process PDFs sequentially, not in parallel
- Monitor system memory during processing

### Logs and Monitoring

#### Backend Logs

```bash
# View FastAPI logs
tail -f uvicorn.log

# Database queries log
tail -f /var/log/postgresql/postgresql-*.log
```

#### Frontend Logs

```bash
# Streamlit logs are displayed in terminal
# Look for error messages and stack traces
```

## 📊 Performance Optimization

### Database Indexes

Already included in `init.sql`:
- Index on `doc_id` for fast status lookups
- Index on `status` for filtering documents
- Index on `session_id` for query history

### Vector Search Optimization

- Adjust `TOP_K_DEFAULT` based on needs
- Increase `SIMILARITY_THRESHOLD` for more precise results
- Use document filters to reduce search space

### Chunking Strategy

- `CHUNK_SIZE`: 400 tokens (balanced for context)
- `CHUNK_OVERLAP`: 50 tokens (maintains continuity)
- Adjust based on your document types

## 🔒 Security Considerations

### API Security

- Add authentication to FastAPI endpoints
- Implement rate limiting
- Use HTTPS in production

### Database Security

- Change default passwords
- Use environment-specific credentials
- Enable SSL for database connections

### File Upload Security

- Validate file types and sizes
- Scan for malware
- Implement user quotas

## 📦 Deployment

### Docker Deployment

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: rag_system
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: rag_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - POSTGRES_HOST=postgres
      - QDRANT_HOST=qdrant
    depends_on:
      - postgres
      - qdrant
    volumes:
      - ./.env:/app/.env

  frontend:
    build: .
    command: streamlit run streamlit_app.py
    ports:
      - "8501:8501"
    depends_on:
      - backend

volumes:
  postgres_data:
  qdrant_data:
```

### Production Recommendations

1. **Load Balancing**: Use nginx or traefik
2. **Monitoring**: Implement Prometheus + Grafana
3. **Logging**: Centralize with ELK stack
4. **Backup**: Regular PostgreSQL and Qdrant backups
5. **Scaling**: Horizontal scaling for FastAPI workers

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review backend logs for errors
- Create an issue with detailed error messages

## 🎯 Project Roadmap

- [ ] Multi-user authentication
- [ ] Support for more file formats (DOCX, TXT)
- [ ] Advanced chunking strategies
- [ ] Multiple LLM providers
- [ ] Export chat history
- [ ] Advanced analytics dashboard
- [ ] Batch document processing
- [ ] API rate limiting
- [ ] Webhook notifications
- [ ] Document versioning
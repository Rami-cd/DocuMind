# PDF Chat Assistant

AI-powered document Q&A system using free, local LLMs with modern web interface.

## Features

- 💰 **Cost-Free** - Uses local Ollama LLMs (no API costs)
- 🔒 **Privacy-First** - All processing happens locally
- 📚 **Multi-PDF Support** - Upload and query multiple documents
- ⚡ **Fast Search** - Vector-based similarity search with FAISS
- 🐳 **Easy Deployment** - One-command Docker setup

## Quick Start

### Run with Docker
```bash
git clone https://github.com/Rami-cd/DocuMind.git
cd DocuMind
docker-compose up -d

# First run takes 10-15 minutes for model downloads
# Access at http://localhost:3000
```

### Local Development
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download models
ollama serve &
ollama pull llama2
ollama pull nomic-embed-text

# Setup Python environment
python -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt

# Start services
cd api && uvicorn main:app --reload &
cd frontend && python -m http.server 3000
```

## Project Structure

```
pdf-chat-assistant/
├── docker-compose.yml     # Container setup
├── api/
│   ├── main.py           # FastAPI backend
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html        # Web interface
└── nginx.conf            # Reverse proxy
```

## Usage

1. **Upload PDFs** - Drag & drop or browse files
2. **Wait for Processing** - Real-time status updates
3. **Ask Questions** - Natural language queries

Example queries:
- "What are the main findings?"
- "Summarize the financial results"
- "What safety protocols are mentioned?"

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload PDF files |
| POST | `/ask` | Ask questions |
| GET | `/health` | System status |
| GET | `/status/{id}` | Processing status |

## Configuration

```bash
# Environment variables
OLLAMA_BASE_URL=http://localhost:11434
CHUNK_SIZE=1000
LLM_MODEL=llama2
EMBEDDING_MODEL=nomic-embed-text
```

## Troubleshooting

### Common Issues

**Ollama not available:**
```bash
docker-compose logs ollama
docker-compose restart ollama
```

**High memory usage:**
```bash
# Use smaller model
ollama pull llama2:7b-chat
```

**Port conflicts:**
```bash
# Change ports in docker-compose.yml
ports: ["8080:80"]
```

## Tech Stack

- **Backend:** FastAPI, Python
- **Frontend:** HTML, JavaScript
- **AI:** Ollama (Llama2), LangChain
- **Database:** FAISS vector store
- **Deployment:** Docker, Nginx

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push and create Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.
# Tractian RAG API

A Retrieval Augmented Generation (RAG) API that allows users to upload PDF documents and ask questions about their content using natural language. The system uses OpenAI embeddings and Elasticsearch for efficient document retrieval, combined with LLM-powered question answering.

## Features

- PDF Document Processing
  - Upload multiple PDF files concurrently
  - Automatic text extraction and chunking
  - Vector embeddings generation using OpenAI's embedding model
  - Efficient storage and indexing in Elasticsearch

- Question Answering
  - Natural language question processing
  - Semantic search using vector embeddings
  - Context-aware answers using RAG
  - Source chunks provided with answers

- System Health Monitoring
  - Elasticsearch cluster health checks
  - Index status monitoring
  - Real-time system status reporting

## Prerequisites

- Docker and Docker Compose
- OpenAI API Key
- (Optional) DeepSeek API Key for alternative LLM provider

## Environment Setup

1. Clone the repository:
```bash
git clone git@github.com:TulioChiodi/rag_challenge_api.git
cd rag_challenge_api
```

2. Create a `.env` file in the project root by copying `.env.example`:


## Running with Docker

The project includes three services:
- API Service (FastAPI): Handles document processing and RAG queries
- UI Service (Streamlit): Provides a user-friendly web interface
- Elasticsearch: Stores and indexes document embeddings

1. Build and start all services:
```bash
docker compose up --build
```

2. Access the services:
- Web UI: `http://0.0.0.0:8501`
- API Documentation: `http://localhost:8000/docs`
- Elasticsearch: `http://localhost:9200`

## API Documentation

Once the service is running, you can access:
- Swagger UI documentation: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

### Quick Start Guide

You can interact with the system either through the Web UI or directly via the API:

#### Using the Web UI (Recommended for exploration)

1. Open the Streamlit interface in your browser:
```
http://0.0.0.0:8501
```

2. Use the intuitive interface to:
   - Upload PDF documents using the file upload widget
   - View processing status and document information
   - Ask questions and see answers with source context
   - Monitor system health

#### Using the API directly

1. Upload PDF documents:
```bash
curl -X POST "http://localhost:8000/documents" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@manual1.pdf" \
     -F "files=@manual2.pdf"
```

2. Ask questions about the documents:
```bash
curl -X POST "http://localhost:8000/question" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the maintenance procedures?"}'
```

## Development

For local development, we'll use a Python virtual environment and run Elasticsearch in Docker:

1. Create virtual environment
```bash
python -m venv venv
```

2. Activate it
```bash
source venv/bin/activate
```

3. Install project dependencies:
```bash
# For API development
pip install -r requirements.txt

# For UI development (Streamlit)
pip install -r ui-requirements.txt
```

4. Setup Elasticsearch:
```bash
# Pull the image
docker pull elasticsearch:9.0.1

# Run Elasticsearch container
docker run -d \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
  --name es-dev \
  elasticsearch:9.0.1
```

5. Run the services:
```bash
# Terminal 1: Run the FastAPI application
python -m src.main

# Terminal 2: Run the Streamlit UI
streamlit run src/ui/main.py
```

The services will be available at:
- API: http://localhost:8000
- UI: http://0.0.0.0:8501
- Elasticsearch: http://localhost:9200

## Future Improvements

### Monitoring
- Prometheus + Grafana
  - System metrics collection and visualization
  - Performance monitoring
  - Resource usage tracking

### App Features
- Conversational interface
  - Chat history persistence
  - Context-aware follow-up questions
- Improve retrieval query quality
  - Two-step prompts for query preprocessing
  - Agentic RAG
- Multilayer storage
  - File metadata (name, description)
  - Graph database integration for relationships
- Hybrid Search on Elasticsearch
  - Combine semantic and keyword search
  - Boost results based on metadata
- Semantic Chunker
  - Experimental implementation
  - Parameter tuning for optimal chunks

### Management Features
- Database control endpoints
  - Recreate database
  - Remove files (whole document)
  - Retrieve file list
  - Change document content/metadata


Contributions and feedback on these improvements are welcome!

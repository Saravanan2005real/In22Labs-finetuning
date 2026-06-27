# Running LexAI with Docker and Docker Compose

This guide provides instructions on how to set up, build, and run the LexAI Legal Chatbot pipeline using Docker and Docker Compose.

---

## 🛠️ Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
- (Optional) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) if running model inference on a GPU inside Docker.

---

## 🚀 Quick Start (Mock Mode)

By default, the application is configured to run in **Mock Mode**, which is the fastest way to verify the UI, Flask server, and Elasticsearch integration without needing a GPU or downloading large model weights.

### 1. Build and Start Services
From the project root directory, run:
```bash
docker compose up --build
```
This command:
- Starts Elasticsearch on `http://localhost:9200` (with security features disabled for development).
- Performs a health check on Elasticsearch.
- Builds the Flask web server image and starts the container on `http://localhost:5000` once Elasticsearch is healthy.

### 2. Extract Text & Index Legal Documents
With the containers running, open a new terminal window and trigger the ingestion pipeline to parse PDFs in the `Act_Elastic search` folder and index them into the Elasticsearch container:
```bash
docker compose exec app python extract_and_index.py
```
This runs the text parser inside the app container, writes the indexed content to the Elasticsearch container, and updates the local `dataset.json` file.

### 3. Open the UI
Go to your browser and open:
```text
http://localhost:5000
```
You can query the legal chatbot, view document counts from Elasticsearch, and inspect mock RAG responses.

---

## ⚙️ GPU Acceleration & Standard Mode (Optional)

To load the full fine-tuned Llama model (`llama_base_model` / `lora_model`) in standard GPU mode inside Docker, follow these steps:

### 1. Enable NVIDIA GPU in `docker-compose.yml`
Update your `app` service in `docker-compose.yml` to use your GPU by setting `MOCK_MODE=false` and adding the `deploy` configuration block:

```yaml
  app:
    build: .
    container_name: lexai-app
    ports:
      - "5000:5000"
    environment:
      - ELASTICSEARCH_URL=http://elasticsearch:9200
      - MOCK_MODE=false  # <--- Change this to false
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./lora_model:/app/lora_model
      - ./llama_base_model:/app/llama_base_model
      - ./Act_Elastic search:/app/Act_Elastic search
      - ./templates:/app/templates
      - ./dataset.json:/app/dataset.json
    depends_on:
      elasticsearch:
        condition: service_healthy
```

### 2. Restart Containers
```bash
docker compose down
docker compose up --build
```

---

## 🧹 Housekeeping & Commands

- **Stop Services**: Press `Ctrl+C` or run `docker compose down`.
- **Remove Volumes** (clear Elasticsearch data): `docker compose down -v`.
- **Inspect Logs**: `docker compose logs -f app` or `docker compose logs -f elasticsearch`.

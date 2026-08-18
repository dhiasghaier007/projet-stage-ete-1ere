# Full RAG pipeline — one image covering every stage (ingest, classify,
# chunk, index, quality, dashboard, qa), run as one-off jobs via
# docker-entrypoint.sh. See that file for the full list of stages.
#
# This supersedes src/ingestion/Dockerfile as the project's containerized
# AI service — that one only covered ingestion; this covers the whole
# pipeline so a single image is the "Containerized AI Service" deliverable.
FROM python:3.11-slim

WORKDIR /app

# System deps Docling needs for PDF/image handling (same as the old
# ingestion-only Dockerfile — every other stage is pure Python).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir -r requirements-pipeline.txt

# Only the code — data, .env, and outputs are all mounted as volumes (see
# docker-compose.yml) so nothing document-specific or secret is baked into
# the image itself.
COPY src/ ./src/
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["--help"]

FROM python:3.11-slim

WORKDIR /app

# System deps Docling needs for PDF/image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ingestion.py .

# Source, output, and manifest are all mounted as volumes (see docker-compose.yml)
# so nothing here is baked into the image — just the code + dependencies.

ENTRYPOINT ["python", "ingestion.py"]
CMD ["--source", "/data/source", "--output", "/data/processed", "--manifest", "/data/manifest.json"]

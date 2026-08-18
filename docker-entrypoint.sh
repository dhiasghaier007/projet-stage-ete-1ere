#!/bin/bash
# Entrypoint for the full RAG pipeline container.
#
# One image, one CLI, multiple stages — matches how this project already
# runs locally (a series of scripts you invoke, not a live service). This
# just makes those same invocations reproducible in a container, and gives
# n8n (or anything else that schedules jobs) a single stable command to
# call: `docker compose run pipeline all`.
#
# Usage:
#   docker compose run pipeline <stage> [stage-specific args...]
#
# Stages:
#   ingest      Parse source docs (PDF/DOCX/XLSX/CSV) into clean Markdown + metadata.
#   classify    LLM classification of ingested documents.
#   chunk       Structure-aware chunking with lineage tracking.
#   index       Build local index + pgvector tables (add --by-department for
#               separate department collections; add --no-cache to force a
#               full re-embed).
#   quality     Compute corpus quality score + drift check.
#   dashboard   Regenerate the static HTML QA dashboard.
#   qa          Ask a question (single-shot or --interactive).
#   all         Run ingest -> classify -> chunk -> index --by-department ->
#               quality -> dashboard end-to-end, using default /app/data
#               paths. This is the one n8n should schedule.
#
# All data paths default to /app/data/* (mount your host ./data there via
# docker-compose.yml) so the same commands work identically in and out of
# the container.

set -euo pipefail

STAGE="${1:-}"
shift || true

DATA_DIR="${DATA_DIR:-/app/data}"

case "$STAGE" in
  ingest)
    exec python -m src.ingestion.ingestion \
      --source "${DATA_DIR}/source" \
      --output "${DATA_DIR}/processed" \
      --manifest "${DATA_DIR}/manifests/ingestion_manifest.json" \
      "$@"
    ;;

  classify)
    exec python -m src.classification.classify \
      --processed "${DATA_DIR}/processed" \
      --output "${DATA_DIR}/classified" \
      --metadata "${DATA_DIR}/manifests/classified_metadata.json" \
      "$@"
    ;;

  chunk)
    exec python -m src.chunking.chunking \
      --classified "${DATA_DIR}/classified" \
      --output "${DATA_DIR}/chunks" \
      --registry "${DATA_DIR}/manifests/asset_registry.json" \
      "$@"
    ;;

  index)
    exec python -m src.indexing.index_cli \
      --chunks "${DATA_DIR}/chunks/chunks.jsonl" \
      --output "${DATA_DIR}/indexing" \
      --pgvector-dsn "${PGVECTOR_DSN:-}" \
      "$@"
    ;;

  quality)
    exec python -m src.qa.quality_report \
      --index "${DATA_DIR}/indexing/local_index.json" \
      "$@"
    ;;

  dashboard)
    exec python -m src.qa.dashboard "$@"
    ;;

  qa)
    exec python -m src.qa.qa_cli \
      --index "${DATA_DIR}/indexing" \
      "$@"
    ;;

  all)
    echo "=== [1/6] ingest ==="
    python -m src.ingestion.ingestion \
      --source "${DATA_DIR}/source" \
      --output "${DATA_DIR}/processed" \
      --manifest "${DATA_DIR}/manifests/ingestion_manifest.json"

    echo "=== [2/6] classify ==="
    python -m src.classification.classify \
      --processed "${DATA_DIR}/processed" \
      --output "${DATA_DIR}/classified" \
      --metadata "${DATA_DIR}/manifests/classified_metadata.json"

    echo "=== [3/6] chunk ==="
    python -m src.chunking.chunking \
      --classified "${DATA_DIR}/classified" \
      --output "${DATA_DIR}/chunks" \
      --registry "${DATA_DIR}/manifests/asset_registry.json"

    echo "=== [4/6] index (by department, incremental) ==="
    python -m src.indexing.index_cli \
      --chunks "${DATA_DIR}/chunks/chunks.jsonl" \
      --output "${DATA_DIR}/indexing" \
      --pgvector-dsn "${PGVECTOR_DSN:-}" \
      --by-department

    echo "=== [5/6] quality ==="
    python -m src.qa.quality_report \
      --index "${DATA_DIR}/indexing/local_index_general.json" || true
    # `|| true`: a quality dip is something to surface via drift detection,
    # not a reason to fail the whole scheduled run and skip the dashboard
    # regeneration below.

    echo "=== [6/6] dashboard ==="
    python -m src.qa.dashboard

    echo "=== pipeline run complete ==="
    ;;

  ""|-h|--help)
    echo "Usage: docker compose run pipeline <stage> [args...]"
    echo "Stages: ingest | classify | chunk | index | quality | dashboard | qa | all"
    exit 0
    ;;

  *)
    echo "Unknown stage: '${STAGE}'"
    echo "Usage: docker compose run pipeline <stage> [args...]"
    echo "Stages: ingest | classify | chunk | index | quality | dashboard | qa | all"
    exit 1
    ;;
esac

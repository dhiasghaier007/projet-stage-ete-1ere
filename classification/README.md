# Classification README

This folder-specific README is now only a pointer. Please use the main project guide in [../README.md](../README.md) for setup, commands, and workflow details.

For the current multilingual run, the key commands are:

```bash
python3 ingestion/ingestion.py --source test_multilingual_samples --output processed_multilingual --manifest manifest_multilingual.json
python3 classification/classify.py --processed processed_multilingual --output classified_multilingual --metadata classified_multilingual_metadata.json
python3 scripts/check_multilingual_accuracy.py
```


## Environment Variables

```bash
# Use heuristic (default)
python classify.py --processed ./processed --output ./classified

# Use LiteLLM with OpenAI
export OPENAI_API_KEY="sk-..."
python classify.py --processed ./processed --output ./classified --use_llm

# Use LiteLLM with Ollama
export LITELLM_MODEL_NAME="ollama/mistral"
python classify.py --processed ./processed --output ./classified --use_llm

# Use LiteLLM with specific model
export LITELLM_MODEL_NAME="gpt-4"
export OPENAI_API_KEY="sk-..."
python classify.py --processed ./processed --output ./classified --use_llm
```

---

## Testing & Evaluation

The `eval_classifiers.py` script includes a **labeled test dataset** with 6 documents and known correct labels:

1. HR policy document
2. Finance quarterly report
3. Invoice
4. CSV data table
5. Legal contract
6. Bonus calculation spreadsheet

This is tied to **Stage 5 (Quality Assurance)** — the same evaluation logic will be extended with RAGAS metrics later.

---

## Troubleshooting

**Q: LiteLLM says "No module named 'litellm'"**
```bash
pip install litellm
```

**Q: OpenAI API says "Unauthorized"**
- Check API key: `echo $OPENAI_API_KEY`
- Ensure it starts with `sk-`
- Create new key at https://platform.openai.com/api-keys

**Q: Ollama connection refused**
- Ensure Ollama is running: `ollama serve` in another terminal
- Check model exists: `ollama list`
- Try: `export LITELLM_MODEL_NAME="ollama/mistral"`

**Q: LLM takes too long**
- Use cheaper model: `export LITELLM_MODEL_NAME="gpt-3.5-turbo"`
- Or use local Ollama (faster after first call)

---

## Next Steps

- **Stage 3**: Chunks from classified documents
- **Stage 4**: Vector embeddings (use classification metadata for filtering)
- **Stage 5**: Full RAGAS evaluation with retrieved context

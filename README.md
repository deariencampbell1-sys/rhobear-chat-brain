# rhobear-chat-brain

OpenAI-compatible chat gateway with a **semantic cache** (sqlite-vec + BGE embeddings). Cache hits answer in milliseconds; misses forward to a local [llama.cpp](https://github.com/ggerganov/llama.cpp) server. Built to make a ~$0.06/hr CPU box feel instant for repeat questions.

## One-line install

```bash
pip install -r requirements.txt && ADMIN_TOKEN=changeme uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On first boot the service seeds `./data/cache.db` from `seeds/sales-faq.jsonl`.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_UPSTREAM` | `http://localhost:8080` | llama.cpp base URL |
| `CACHE_THRESHOLD` | `0.86` | Minimum cosine similarity for a cache hit |
| `ADMIN_TOKEN` | *(required for seeding)* | Token for `POST /admin/seed` (`X-Admin-Token` header) |
| `PORT` | `8000` | HTTP listen port |
| `DATA_DIR` | `./data` | Directory for `cache.db` and `requests.db` |
| `SEEDS_PATH` | `./seeds/sales-faq.jsonl` | Bootstrap seed file |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (supports `stream: true`) |
| `GET` | `/metrics.json` | Latency, throughput, cache hit rate (last 24 h) |
| `GET` | `/healthz` | 200 when sqlite + llama.cpp upstream are reachable |
| `POST` | `/admin/seed` | Bulk-add cached Q&A pairs (NDJSON body) |

## curl examples

**Cache hit** (seeded FAQ):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"demo","messages":[{"role":"user","content":"What model do you use?"}]}'
```

**Cache miss** (forwarded to llama.cpp):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"demo","messages":[{"role":"user","content":"Explain quantum foam"}]}'
```

**Streaming:**

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"demo","stream":true,"messages":[{"role":"user","content":"What model do you use?"}]}'
```

**Metrics:**

```bash
curl -s http://localhost:8000/metrics.json
```

**Seed new pairs:**

```bash
curl -s http://localhost:8000/admin/seed \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary '{"q":"What is your SLA?","a":"99.9% on the gateway; LLM latency varies."}'
```

## Docker

```bash
docker build -t rhobear-chat-brain .
docker run --rm -p 8000:8000 -e ADMIN_TOKEN=changeme rhobear-chat-brain
```

## systemd

Copy the app to `/opt/chat-brain`, install the unit from `systemd/chat-brain.service`, then:

```bash
sudo systemctl enable --now chat-brain
```

## Development

```bash
pip install -r requirements-dev.txt
pytest -v
```
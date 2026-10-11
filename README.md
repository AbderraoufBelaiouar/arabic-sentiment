# Arabic Sentiment Analysis

This project is a production-ready Deep Learning API for detecting Arabic sentiment (positive, negative, or neutral). It uses a fine-tuned AraBERT model that has been optimized and exported to ONNX to guarantee highly efficient inference latency. The service is packaged with `pyproject.toml`, tested with `pytest`, and fully containerized via a multi-stage Docker build acting as an automated REST API powered by FastAPI and Pydantic.

## Quickstart: From Zero to Prediction in 3 Commands

You do not need to install Python or clone the entire repository to run the model if the image is published. To build from source and run, execute these exact 3 commands:

```bash
# 1. Build the lightweight, CPU-optimized container image
docker build -f docker/Dockerfile -t arabic-sentiment:latest .

# 2. Run the Docker container exposing Port 8000
docker run -d --rm -p 8000:8000 --name arabic-sentiment-api arabic-sentiment:latest

# 3. Test the deployment by sending an Arabic sentence classification request!
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "المنتج رائع والتوصيل سريع"}'
```

*(You can also visit `http://localhost:8000/docs` in your browser to interact with the interactive Swagger dashboard!)*

## Repository Structure

```text
.
├── data
├── docker
│   └── Dockerfile
├── models
│   └── arabert-twitter-onnx
│       ├── model.onnx
│       ├── model.onnx.data
│       ├── tokenizer_config.json
│       └── tokenizer.json
├── notebooks
│   └── baseline.ipynb
├── pyproject.toml
├── README.md
├── reports
├── src
│   └── arabic_sentiment
│       ├── api
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── schemas.py
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── __init__.py
│       ├── metrics.py
│       ├── model.py
│       └── train.py
├── tests
│   ├── conftest.py
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_config.py
│   └── test_metrics.py
└── uv.lock
```
## CI/CD Pipeline & Quality Gates
This project utilizes GitHub Actions to execute a robust pipeline upon every push to [main](cci:1://file:///home/abderraouf/Desktop/mlops%20mena/arabic-sentiment/src/arabic_sentiment/train.py:41:0-152:23). 
It runs `ruff` linting and the `pytest` test suite. Crucially, it features an automated Quality Gate: it reads the generated [reports/metrics.json](cci:7://file:///home/abderraouf/Desktop/mlops%20mena/arabic-sentiment/reports/metrics.json:0:0-0:0) file, and if the newly trained model's `f1_macro` drops below the baseline threshold of `0.6900`, the pipeline forcibly fails with `sys.exit(1)`.
## Canary Deployment & Rollout Strategy

To safely release optimized models into production without risking service downtime or latency regressions, we utilize an **Nginx-based Canary Traffic Split** ([docker/nginx/canary.conf](cci:7://file:///home/abderraouf/Desktop/mlops%20mena/arabic-sentiment/docker/nginx/canary.conf:0:0-0:0)).

### Architecture & Routing Rules
* **Stable Upstream (95% Traffic):** Current production FastAPI service serving AraBERT ONNX baseline (`api:8000`).
* **Canary Upstream (5% Traffic):** Candidate BentoML micro-batched service (`bentoml:8000`).
* **Deterministic Splitting:** Uses Nginx `split_clients "${remote_addr}${http_user_agent}"` to ensure a consistent user experience during the trial.
* **Internal Override:** Teams can bypass traffic weights and test the canary directly by supplying the HTTP header `X-Canary: always`.

### Rollout Lifecycle

1. **Phase 1 — Dark Launch / Internal Validation (0% public):**
   * Deploy the candidate container. Route test traffic exclusively via `X-Canary: always` to confirm health and verify that predictions return HTTP 200 without memory leaks.
2. **Phase 2 — Initial Canary (5% weight — Rubric Baseline):**
   * Enable 5% public routing via `split_clients`.
   * Monitor error rates (HTTP 5xx) and latency (p95) for 30–60 minutes.
3. **Phase 3 — Progressive Ramp-Up (25% ➔ 50%):**
   * If error rate $< 0.1\%$ and p95 latency stays under 1000 ms, increase canary upstream weight to 25%, then 50%.
4. **Phase 4 — Full Promotion (100%):**
   * Promote canary to `stable_backend`. Decommission the old container.
5. **Rollback Policy:**
   * If the canary error rate exceeds $1\%$ or p95 latency exceeds $1500\text{ ms}$, immediately revert traffic to 100% stable by setting the canary weight to 0%.

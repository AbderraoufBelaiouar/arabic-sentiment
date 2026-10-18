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
## Observability & Drift Monitoring

To ensure reliability, low latency, and detect data distribution shifts in production, the service integrates **Evidently AI**, **Prometheus**, and **Grafana**.

### 1. Automated Drift Detection & Evidently AI
Every batch scoring run automatically triggers statistical drift analysis (`src/arabic_sentiment/monitoring/drift.py`) comparing incoming samples against training baselines:
* **Interactive HTML Report:** Stored at [`reports/data_drift_report.html`](reports/data_drift_report.html) using Evidently AI's `DataDriftPreset`.
* **Machine-Readable Metrics:** Exported to [`reports/drift_metrics.json`](reports/drift_metrics.json).

### 2. Population Stability Index (PSI) Benchmarks
PSI is computed for `text_length` and `confidence_score`:
* 🟢 **PSI < 0.10:** Stable (No distribution shift).
* 🟡 **0.10 ≤ PSI ≤ 0.25:** Moderate drift (Added to watchlist).
* 🔴 **PSI > 0.25:** Critical drift (Triggers automated alerts and retraining).

### 3. Prometheus Metrics (`/metrics`)
The FastAPI serving app exposes the following Prometheus gauges and counters at `/metrics`:
* `arabic_sentiment_psi_score{feature="text_length"}`: Population Stability Index for text length.
* `arabic_sentiment_psi_score{feature="confidence_score"}`: Population Stability Index for prediction confidence.
* `arabic_sentiment_p95_latency_ms`: Observed 95th percentile latency (ms).
* `arabic_sentiment_requests_total{status="success|failure"}`: Total requests counter.
* `arabic_sentiment_latency_seconds`: Prediction latency histogram.

### 4. Grafana Observability Dashboard
A production dashboard configuration is provided at [`docker/grafana/dashboard.json`](docker/grafana/dashboard.json) featuring:
* **Panel 1 — Serving p95 Latency:** Monitors SLA compliance ($< 1000\text{ ms}$).
* **Panel 2 — PSI Drift Monitor:** Visualizes stability across input features with green/yellow/red threshold zones.

### 5. Alert Rule & Incident Runbook (`PSI > 0.25`)
* **Alert Name:** `ArabicSentimentHighDataDrift`
* **Condition:** `arabic_sentiment_psi_score > 0.25` for `5m`.
* **Severity:** `Critical`
* **Notification Channel:** PagerDuty / Slack `#mlops-alerts`.
* **Incident Runbook:**
  1. On-call engineer inspects [`reports/data_drift_report.html`](reports/data_drift_report.html) to identify drifted features.
  2. If vocabulary or domain shift occurred, collect recent predictions from `data/scoring/output/`.
  3. Trigger automated retraining pipeline with newly labeled production data.
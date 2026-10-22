# Sift

**A classical-ML spam classifier with an explainable REST API and a modern React interface.**

Sift scores a message with word and character TF-IDF plus hand-built text
statistics, feeds those features to a linear model, and returns a calibrated
spam probability, a precision-weighted verdict, and the exact words that moved
the decision. No deep learning, no GPU, and no hidden preprocessing.

The trained artifact in `ml/artifacts/` is ready to serve. The full corpus and
training code are included so the model can be rebuilt from scratch.

---

## Contents

- [Results](#results)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Quick start with Docker](#quick-start-with-docker)
- [Manual installation](#manual-installation)
  - [Prerequisites](#prerequisites)
  - [1. Clone and create a virtual environment](#1-clone-and-create-a-virtual-environment)
  - [2. Add the dataset](#2-add-the-dataset)
  - [3. Train the model](#3-train-the-model)
  - [4. Prepare the database](#4-prepare-the-database)
  - [5. Run the Django API](#5-run-the-django-api)
  - [6. Run the React frontend](#6-run-the-react-frontend)
  - [7. Open the notebook](#7-open-the-notebook)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Batch CSV format](#batch-csv-format)
- [Tests and validation](#tests-and-validation)
- [Model card](#model-card)
- [Troubleshooting](#troubleshooting)
- [Deployment notes](#deployment-notes)

---

## Results

Evaluated on **7,613 held-out test messages** that were never used for feature
fitting, model selection, or threshold tuning.

| Metric | Tuned cut-off (`0.7104`) | Default `0.5` |
| --- | ---: | ---: |
| Accuracy | **98.83%** | 98.96% |
| Precision | **99.51%** | 98.56% |
| Recall | 98.22% | **99.44%** |
| F1 | 98.86% | **99.00%** |
| F0.5 | **99.25%** | 98.74% |
| ROC-AUC | 99.93% | 99.93% |
| PR-AUC | 99.92% | 99.92% |
| Brier score | **0.0089** | 0.0089 |

Confusion matrix at the tuned cut-off:

|  | Predicted ham | Predicted spam |
| --- | ---: | ---: |
| **Actual ham** | 3,666 | 19 |
| **Actual spam** | 70 | 3,858 |

The tuned threshold deliberately trades a little recall for precision: 19 false
positives instead of 57 at the default 0.5. A blocked invoice is usually more
expensive than one junk message reaching an inbox, so F0.5 — not accuracy — is
the tuning objective.

> These scores are specific to `combined_data.csv`. Public spam corpora contain
> repeated campaigns and can be easier than live mail. Validate on your own
> labelled messages before relying on the filter in production.

---

## How it works

```text
                       ┌─────────────────────────────┐
  Message ───────────► │  word TF-IDF (1–2 grams)    │ ─┐
                       │  char TF-IDF (3–5, 4k cap)  │  ├─► linear model ─► threshold ─► spam / ham
                       │  23 text statistics         │ ─┘        │
                       └─────────────────────────────┘           │
                                                                 └─► signed word contributions
```

- **Word TF-IDF** captures phrases such as “click here”, “act now”, and
  “unclaimed prize”.
- **Character TF-IDF** (`char_wb`, 3–5 grams) handles obfuscation such as
  `fr33 m0ney` and `V1AGRA`.
- **Text statistics** add link, email, HTML, caps, digit, punctuation, urgency,
  money, and marketing signals. A `MaxAbsScaler` keeps them non-negative for
  ComplementNB.
- A `FeatureUnion` combines the three views with weights `1.0 / 0.65 / 0.35`.
- Logistic regression (selected on validation PR-AUC) outputs probabilities; the
  decision threshold is tuned on validation to maximise F0.5.
- Exact duplicates and SimHash near-duplicates are removed *before* splitting to
  prevent train/test leakage.

---

## Repository layout

```text
.
├── backend/                  Django 6 + Django REST Framework API
│   ├── classifier/           models, serializers, views, services, tests
│   ├── config/               settings, URLs, WSGI/ASGI
│   ├── Dockerfile            gunicorn image (2 workers)
│   ├── manage.py
│   └── requirements.txt
├── frontend/                 React 19 + Vite 7 single-page app
│   ├── src/                  pages, components, API client, design system
│   ├── Dockerfile            multi-stage Node build → nginx
│   ├── nginx.conf            serves the SPA and proxies /api/
│   ├── package.json
│   └── vite.config.js
├── ml/                       framework-agnostic training package
│   ├── artifacts/
│   │   ├── spam_pipeline.joblib   trained deployable pipeline (~4.2 MB)
│   │   └── metrics.json           metrics, feature list, training metadata
│   ├── notebooks/
│   │   └── spam_classifier.ipynb  documented, executed walkthrough
│   ├── data.py               loading, cleaning, SimHash, splitting
│   ├── features.py           TF-IDF views and TextStats
│   ├── train.py              model search, threshold tuning, export
│   ├── evaluate.py           metrics and reporting helpers
│   ├── predict.py            artifact loading and explanations
│   └── requirements.txt
├── legacy/                   original prototype kept for reference
├── combined_data.csv         training corpus (140 MB, gitignored)
├── docker-compose.yaml
├── requirements.txt          full local dev environment
└── .env.example
```

---

## Quick start with Docker

**Prerequisite:** Docker Desktop (or Docker Engine + Compose v2).

```bash
git clone <your-repository-url>
cd Spam-classifire
docker compose up --build
```

Then open:

| Service | URL |
| --- | --- |
| Web app | <http://localhost:8080> |
| API health | <http://localhost:8000/api/health/> |
| Swagger UI | <http://localhost:8000/api/docs/> |

The compose stack uses the prebuilt artifact, so the 140 MB dataset is **not**
needed for this path. SQLite history is stored in the `sift-data` volume and
survives container restarts.

```bash
# Follow logs
docker compose logs -f backend

# Stop, keeping prediction history
docker compose down

# Stop and delete the database volume
docker compose down -v

# Rebuild after retraining locally
docker compose up --build backend
```

Optional: copy `.env.example` to `.env` and change `DJANGO_SECRET_KEY`,
allowed hosts, or upload limits before starting. Compose has development
defaults, so `.env` is not required for local use.

---

## Manual installation

### Prerequisites

| Tool | Tested version | Notes |
| --- | --- | --- |
| Python | **3.13.3** | Use 3.13 to match the artifact's scikit-learn build. |
| Node.js | **22.16** | Node 20+ should work; npm 10+ is recommended. |
| Git | Any recent version | |
| RAM | ~2 GB free | Full training peaks around 1.5–2 GB. |
| Dataset | `combined_data.csv` | Only needed for retraining, not for serving. |

The artifact was trained with scikit-learn 1.9.1, NumPy 2.5.4, and Python 3.13.
Using a different Python/scikit-learn combination can produce an
`InconsistentVersionWarning` or fail to unpickle; retrain in that case.

### 1. Clone and create a virtual environment

**Windows (PowerShell):**

```powershell
git clone <your-repository-url>
cd Spam-classifire

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, allow it for the current shell only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
git clone <your-repository-url>
cd Spam-classifire

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs the backend, training stack, and notebook tooling in
one go. To install only one part:

```bash
pip install -r backend/requirements.txt   # API only
pip install -r ml/requirements.txt        # training + notebook only
```

### 2. Add the dataset

Place `combined_data.csv` in the repository root:

```text
combined_data.csv
├── text      raw message body
└── label     1 = spam, 0 = ham
```

The file is intentionally gitignored because it is ~140 MB. The trained
artifact is committed, so you can run the API and frontend without it. You need
the dataset only to retrain.

Sanity check:

```bash
python -c "import pandas as pd; f=pd.read_csv('combined_data.csv'); print(f.shape); print(f['label'].value_counts().to_dict())"
# (83448, 2)
# {1: 43910, 0: 39538}
```

If you keep the dataset elsewhere, pass it explicitly:

```bash
python -m ml.train --data D:\datasets\combined_data.csv
```

### 3. Train the model

Skip this step to use the committed artifact.

```bash
python -m ml.train
```

Expected output and runtime:

- ~10–15 minutes on a modern laptop, plus ~2 minutes for feature fitting.
- Raw: 83,448 rows → clean: 76,127 (697 exact + 6,591 near duplicates removed).
- Split: 60,901 train / 7,613 validation / 7,613 test.
- Writes `ml/artifacts/spam_pipeline.joblib` and `ml/artifacts/metrics.json`.
- Prints validation and test metrics and the selected threshold.

For a fast smoke test without the full run:

```bash
python -m ml.train --sample 5000 --output ml/artifacts-smoke
```

Useful flags:

```text
--data PATH          path to combined_data.csv
--output DIR         artifact directory (default: ml/artifacts)
--sample N           row cap for smoke tests (after cleaning)
--no-near-dedup      skip SimHash removal (faster, but risks leakage)
--random-state N     control sampling and splitting
```

Training rewrites the artifact. **Restart the Django process** afterwards so
each worker loads the new model. The frontend model page reads the refreshed
`metrics.json` on request.

### 4. Prepare the database

From the repository root:

```bash
python backend/manage.py migrate
```

This creates `backend/db.sqlite3` with the prediction-history table. The file is
gitignored. You do not need to create a superuser to use the API; do so only if
you want Django admin:

```bash
python backend/manage.py createsuperuser
```

### 5. Run the Django API

```bash
python backend/manage.py runserver 127.0.0.1:8000
```

Verify:

```bash
curl http://127.0.0.1:8000/api/health/
```

A healthy response includes `"status": "ok"`, `"model_loaded": true`, and
`"model_version": "2.0.0"`.

Useful endpoints while the server is running:

- Swagger UI: <http://127.0.0.1:8000/api/docs/>
- OpenAPI schema: <http://127.0.0.1:8000/api/schema/>
- Django admin: <http://127.0.0.1:8000/admin/> (requires a superuser)

### 6. Run the React frontend

In a **second terminal**:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://127.0.0.1:3000>. The Vite dev server proxies `/api` to
`http://127.0.0.1:8000`, so there is no CORS configuration to manage.

If `npm ci` complains about the lockfile, run `npm install` once.

Production build preview:

```bash
npm run build
npm run preview   # http://127.0.0.1:4173
```

The dev server is pinned to `127.0.0.1:3000` in `frontend/vite.config.js`. Change
`server.port` there if 3000 is busy; keep using the Vite proxy so API calls stay
same-origin.

### 7. Open the notebook

```bash
jupyter lab ml/notebooks/spam_classifier.ipynb
```

The notebook is committed **with executed outputs** and covers the dataset
audit, cleaning, leak-free split, feature engineering, candidate comparison,
threshold tuning, test metrics, explainability, and the production artifact. It
uses a 14,000-message stratified sample by default to stay quick; set
`DEMO_ROWS = None` to run the full corpus.

---

## Configuration

Backend settings are environment variables read at process start. Defaults work
for local development.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DJANGO_SECRET_KEY` | dev-only value | Set a long random value outside local dev. |
| `DJANGO_DEBUG` | `true` | Set `0` in production. |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1,backend,0.0.0.0` | Comma-separated hostnames. |
| `CORS_ALLOWED_ORIGINS` | `localhost/127.0.0.1:3000` + `:5173` | Needed only for direct cross-origin calls. |
| `CSRF_TRUSTED_ORIGINS` | `localhost/127.0.0.1:8080` + `:3000` | Trusted origins for unsafe requests. |
| `DB_PATH` | `backend/db.sqlite3` | SQLite path; Docker uses `/app/data/db.sqlite3`. |
| `MODEL_PATH` | `ml/artifacts/spam_pipeline.joblib` | Joblib artifact. |
| `METRICS_PATH` | `ml/artifacts/metrics.json` | Metrics/model-card sidecar. |
| `MAX_TEXT_CHARS` | `20000` | Per-message truncation/validation limit. |
| `MAX_BATCH_ROWS` | `500` | Rows per CSV upload. |
| `MAX_BATCH_FILE_MB` | `5` | CSV upload size limit. |
| `THROTTLE_ANON` | `120/min` | General anonymous request limit. |
| `THROTTLE_PREDICT` | `60/min` | Single-prediction limit. |
| `THROTTLE_BATCH` | `12/min` | Batch-upload limit. |
| `LOG_LEVEL` | `INFO` | Python logging level. |

Frontend:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `/api` | API prefix. Use `/api` behind Vite/nginx; set an absolute URL only when the API is on another origin. |

Docker Compose reads `.env` from the repository root. See `.env.example` for a
starting point. Environment changes require a restart:

```bash
docker compose up -d --force-recreate backend
```

---

## API reference

Base URL: `http://127.0.0.1:8000/api`

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health/` | Database + model readiness. Returns `503` if the model cannot load. |
| `POST` | `/predict/` | Classify one message and persist it. JSON body: `{"text": "..."}`. |
| `POST` | `/predict/batch/` | Classify a CSV. Multipart fields: `file`, optional `text_column`. |
| `GET` | `/history/` | Paginated predictions. Filters: `label`, `source`, `search`. |
| `GET` | `/stats/` | Aggregate counts, spam rate, 7-day trend, sources. |
| `GET` | `/model-info/` | Model card: metrics, features, threshold, dataset stats. |
| `GET` | `/schema/` | OpenAPI 3 schema. |
| `GET` | `/docs/` | Swagger UI. |

### Single prediction

```bash
curl -s http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text":"Congratulations! You have won a free prize. Click here to claim your money now."}'
```

Response (abridged):

```json
{
  "id": 13,
  "label": "spam",
  "spam_probability": 0.993215,
  "ham_probability": 0.006785,
  "threshold": 0.7104,
  "confidence": 0.9932,
  "top_words": [
    {"word": "your", "weight": 0.3415},
    {"word": "money", "weight": 0.3326},
    {"word": "click here", "weight": 0.2448}
  ],
  "signals": {
    "toward_spam": [{"word": "your", "weight": 0.3415}],
    "toward_ham": [{"word": "congratulations", "weight": -0.4326}]
  },
  "model_version": "2.0.0",
  "text_length": 74,
  "created_at": "2026-10-22T19:31:10.090443Z"
}
```

`top_words` lists the largest signed contributions for this message. Positive
weights push toward spam; negative weights pull toward ham. The model is linear,
so these are exact contributions rather than post-hoc approximations.

### Batch prediction

```bash
curl -s http://127.0.0.1:8000/api/predict/batch/ \
  -F "file=@messages.csv" \
  -F "text_column=text"
```

The response reports `count`, `skipped_empty`, `spam_count`, `ham_count`,
`text_column`, the threshold, model version, and a `results` array with `row`,
`text_preview`, `label`, `spam_probability`, and `threshold`. Each non-empty row
is also stored in history with source `batch`.

### History

```bash
curl "http://127.0.0.1:8000/api/history/?page=1&page_size=8&label=spam&source=single&search=prize"
```

Standard page-number pagination: `count`, `next`, `previous`, `results`.
History stores a 280-character preview, not the full message body.

### Model card and stats

```bash
curl http://127.0.0.1:8000/api/model-info/
curl http://127.0.0.1:8000/api/stats/
```

Filesystem paths are deliberately stripped from the model-info response.

### Errors

Errors use one envelope:

```json
{
  "error": {
    "status": 400,
    "message": "Text must not be empty or whitespace only.",
    "details": {}
  }
}
```

Common statuses: `400` invalid input/CSV, `413` oversized upload, `429` throttle,
`503` missing or unloadable model.

---

## Batch CSV format

Only the message column is required. Additional columns (including `label`) are
ignored by the endpoint.

```csv
text,label
"Hi team, the design review moved to Thursday at 10:00.",ham
"URGENT: verify your account and claim your $1,000 reward now!",spam
```

Rules:

- The first row is the header.
- Every comma inside a field must be quoted, or wrap the entire field in double
  quotes and escape embedded quotes as `""`.
- Auto-detection looks for `text`, `email`, `email_content`, `message`,
  `content`, `body`, or `email_text`; otherwise it uses the first column. Pass
  `text_column` to override.
- Empty or whitespace-only rows are skipped and counted.
- Maximum 5 MB and 500 data rows per upload.
- UTF-8 (with or without BOM) and Latin-1 are accepted.

---

## Tests and validation

Backend API tests (13 cases; they use a small local fixture, not the production
artifact):

```bash
python backend/manage.py test classifier
```

Frontend production build:

```bash
cd frontend
npm ci
npm run build
```

Notebook execution check:

```bash
jupyter nbconvert --to notebook --execute ml/notebooks/spam_classifier.ipynb \
  --output /tmp/spam_classifier.executed.ipynb
```

Docker configuration check:

```bash
docker compose config
docker compose build
docker compose up
```

---

## Model card

### Data

| | |
| --- | --- |
| Corpus | `combined_data.csv` |
| Raw rows | 83,448 |
| Clean rows | 76,127 |
| Removed | 697 exact + 6,591 SimHash near-duplicates |
| Labels | 36,850 ham / 39,277 spam |
| Median length | 824 characters |
| Max length | 20,000 characters (truncated) |
| Split | 80 / 10 / 10 stratified, created before fitting |

### Features

| View | Configuration | Weight |
| --- | --- | --- |
| Word TF-IDF | 1–2 grams, 150k features, `min_df=2`, sublinear TF | 1.00 |
| Character TF-IDF | `char_wb` 3–5 grams, 80k features, first 4,000 chars | 0.65 |
| Text statistics | 23 dense features, `MaxAbsScaler` | 0.35 |

Total feature space: **230,023**. Average non-zeros per message: ~1,387.

### Model selection

Three classical candidates were compared on the validation split. The initial
family comparison used logistic regression with `C=4`; `C` was then tuned over
`0.5, 1, 2, 4, 8, 16`. `C=2` was selected as the first maximum on validation
PR-AUC (several values tied at `0.9986`).

| Candidate | Validation PR-AUC | Validation precision | Validation recall | Validation F1 |
| --- | ---: | ---: | ---: | ---: |
| **LogisticRegression, `C=4`** | **0.9986** | 0.9931 | 0.9837 | 0.9884 |
| Calibrated LinearSVC | 0.9983 | 0.9933 | 0.9885 | 0.9909 |
| ComplementNB | 0.9958 | 0.9885 | 0.9603 | 0.9742 |

The final logistic-regression configuration (`C=2`) scored validation PR-AUC
`0.9986` and F0.5 `0.9903`. It was refit on train + validation for the deployed
artifact. The decision threshold (`0.7104`) maximises F0.5 on validation; the
test split was untouched until the final evaluation.

### Artifact

- `ml/artifacts/spam_pipeline.joblib` — ~4.2 MB; pure scikit-learn pipeline plus
  the small `TextStats` class from `ml.features`.
- `ml/artifacts/metrics.json` — dataset stats, candidate comparison, threshold,
  test metrics, feature list, library versions.
- Batch inference: ~1.9 ms per message in the training environment; a single
  request adds explanation overhead and API/database latency.

### Limitations

- Dataset drift: vocabulary and campaigns change; schedule retraining and
  monitor false positives.
- Body-only analysis: no SPF/DKIM, sender reputation, attachments, or link
  reputation.
- Short messages can be over-confident; treat borderline scores as review
  candidates.
- Obfuscation is partially handled by character n-grams, not solved.
- Aggressive near-duplicate removal protects the test score but can discard
  legitimate templates.
- The API is unauthenticated by default. Add authentication and per-user quotas
  before exposing it publicly.

---

## Troubleshooting

### `503 The classifier is not available`

The artifact is missing or cannot be loaded.

```bash
ls ml/artifacts/spam_pipeline.joblib
python -m ml.train          # only if you have combined_data.csv
```

Check `MODEL_PATH` and restart Django. In Docker, confirm the file exists locally
and rebuild the backend image; the build context must include `ml/artifacts/`.

### `InconsistentVersionWarning` or unpickle errors

Train with the same Python/scikit-learn versions as the artifact:

```bash
python --version            # 3.13.x
pip install -r requirements.txt
```

If you intentionally upgraded scikit-learn, retrain the model.

### CORS errors in the browser

Use the Vite dev server (`npm run dev`) or Docker nginx, which proxy `/api` and
keep everything same-origin. For direct cross-origin calls, add the frontend
origin to `CORS_ALLOWED_ORIGINS` and restart Django.

### Vite fails with `EACCES` or the port is unavailable

Windows may reserve ranges that include common Vite ports. Change
`server.port` in `frontend/vite.config.js`, or use the configured port `3000`.
Keep the `/api` proxy target aligned with the Django port.

### `Column '...' was not found` or unquoted-comma CSV errors

Check the header name and quote every field containing a comma. The API rejects
rows with more fields than the header rather than silently truncating messages.

### `npm ci` fails

The lockfile may be out of sync:

```bash
cd frontend
npm install
npm run build
```

### Docker build is slow or runs out of memory

The backend image installs scikit-learn/SciPy and runs two gunicorn workers; each
worker loads its own model copy. Give Docker Desktop at least 4 GB. The dataset
is excluded from the image; only the 4.2 MB artifact is copied.

### Prediction history contains test rows

History is stored in SQLite. For local development, stop the API and delete
`backend/db.sqlite3`, then rerun `python backend/manage.py migrate`. For Docker,
use `docker compose down -v` to remove the volume.

---

## Deployment notes

- Set `DJANGO_DEBUG=0`, a strong `DJANGO_SECRET_KEY`, and explicit
  `DJANGO_ALLOWED_HOSTS` / `CSRF_TRUSTED_ORIGINS` for the public hostname.
- Put nginx or a load balancer in front of both services and enable TLS.
- The included gunicorn command uses two workers. Each worker loads the model
  once; account for that memory when scaling.
- SQLite is fine for a demo or single instance. Use PostgreSQL for concurrent
  production traffic and run migrations during deployment.
- The API has anonymous throttles but no authentication or user accounts. Add
  auth and per-user limits before exposing it to the internet.
- History stores previews only, but previews can still contain personal data;
  define retention and deletion policies.
- Retrain on fresh labelled mail, compare precision/recall before promoting, and
  keep the previous artifact for rollback.

No open-source license file is included in this repository. Add one before
distributing the project.

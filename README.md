# Sift

**A classical-ML spam classifier with an explainable REST API and a React UI.**

Sift turns a message into word and character TF-IDF features, scores it with
logistic regression, and returns a spam probability, a precision-weighted
verdict, and the words that moved the decision. No GPU and no hidden
preprocessing.

The trained artifact is committed, so the API and UI run without the training
data. The full training code is included so the model can be rebuilt.

---

## Contents

- [Results](#results)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Quick start with Docker](#quick-start-with-docker)
- [Manual installation](#manual-installation)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Batch CSV format](#batch-csv-format)
- [Tests](#tests)
- [Model card](#model-card)
- [Troubleshooting](#troubleshooting)
- [Deployment notes](#deployment-notes)

---

## Results

The test split was not used for feature fitting, model selection, or threshold
tuning.

| Metric | Tuned cut-off (`0.7219`) | Default `0.5` |
| --- | ---: | ---: |
| Accuracy | **93.51%** | 97.10% |
| Precision | **99.40%** | 96.16% |
| Recall | 87.32% | **97.99%** |
| F1 | 92.97% | **97.07%** |
| F0.5 | **96.72%** | 96.52% |
| ROC-AUC | 99.64% | 99.64% |
| PR-AUC | 99.62% | 99.62% |
| Brier score | **0.0370** | 0.0370 |

Confusion matrix at the tuned cut-off:

|  | Predicted ham | Predicted spam |
| --- | ---: | ---: |
| **Actual ham** | 4,114 | 21 |
| **Actual spam** | 506 | 3,484 |

The cut-off is chosen to keep false positives low. A blocked invoice is usually
more expensive than one junk message reaching an inbox, so the threshold
maximises F0.5 on validation while keeping validation FPR at or below 0.5%.

### External check

The in-dataset score can be optimistic because the training and test rows share
a source. The model was also tested on the public **SpamAssassin** corpus, which
was not used for selection or tuning.

| Metric | SpamAssassin at `0.7219` |
| --- | ---: |
| Accuracy | 93.72% |
| Precision | 89.95% |
| Recall | 87.02% |
| F1 | 88.46% |
| F0.5 | 89.35% |
| ROC-AUC | 97.38% |
| PR-AUC | 94.10% |
| Confusion matrix | 3,937 TN / 152 FP / 203 FN / 1,361 TP |

The external false-positive rate is 3.72%, higher than the 0.5% validation cap.
That gap is the point of the external check: the model generalises, but not
perfectly.

> These numbers are from public corpora. Validate on your own labelled mail
> before using the filter in production.

---

## How it works

```text
Message
  │
  ├─ normalize: lowercase, strip HTML, mask numbers and URLs
  │
  ├─ word TF-IDF (1–2 grams, 150k features)
  ├─ char TF-IDF (char_wb 3–5 grams, 80k features, weight 0.2)
  │
  ├─ logistic regression (C=0.25)
  ├─ threshold 0.7219
  └─ signed word contributions for the explanation
```

- **Normalization** removes standalone numbers, years, dates, and the old
  corpus's `escapenumber` token. URLs become `urltoken` and email addresses
  become `emailtoken`. HTML tags and repeated whitespace are removed.
- **Word TF-IDF** catches phrases such as "click here", "act now", and
  "unclaimed prize".
- **Character TF-IDF** catches obfuscation such as `fr33 m0ney` and `V1AGRA`.
  It only reads the first 4,000 characters and gets a low weight.
- **No length or style features** are used by the deployed model. Length,
  average word length, unique-word ratio, casing, line count, and punctuation
  counts made short messages look like spam and transferred poorly to another
  corpus. `TextStats` remains in `ml/features.py` for ablations only.
- **Leak protection:** exact duplicates and SimHash near-duplicates are removed
  before the stratified train / validation / test split.

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
│   │   ├── spam_pipeline.joblib   trained pipeline (~4.2 MB)
│   │   └── metrics.json           metrics and model metadata
│   ├── notebooks/
│   │   └── spam_classifier.ipynb  executed walkthrough
│   ├── data.py               loading, cleaning, SimHash, splitting
│   ├── features.py           normalization, TF-IDF, optional TextStats
│   ├── train.py              model search, threshold tuning, export
│   ├── evaluate.py           metrics and reporting helpers
│   ├── external.py           SpamAssassin download and evaluation
│   ├── predict.py            artifact loading and explanations
│   └── requirements.txt
├── legacy/                   original prototype kept for reference
├── data/                     local datasets (gitignored)
├── combined_data.csv         main corpus (gitignored)
├── docker-compose.yaml
├── requirements.txt          full local dev environment
└── .env.example
```

---

## Quick start with Docker

**Prerequisite:** Docker Desktop, or Docker Engine with Compose v2.

Docker Compose now requires a real `DJANGO_SECRET_KEY`. It will refuse to start
without one.

```bash
git clone <your-repository-url>
cd Spam-classifire
cp .env.example .env
```

Generate a secret and put it in `.env`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

On PowerShell:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Then start the stack:

```bash
docker compose up --build
```

Open:

| Service | URL |
| --- | --- |
| Web app | <http://localhost:8080> |
| API health | <http://localhost:8000/api/health/> |
| Swagger UI | <http://localhost:8000/api/docs/> |

The image uses the committed artifact, so the dataset is not needed for this
path. Prediction history lives in the `sift-data` volume.

```bash
# Follow logs
docker compose logs -f backend

# Stop, keeping history
docker compose down

# Stop and delete the database volume
docker compose down -v

# Rebuild after retraining
docker compose up --build backend
```

Django rejects known placeholder secrets and any secret shorter than 32
characters when `DJANGO_DEBUG=0`. Keep `DJANGO_DEBUG=0` in production.

---

## Manual installation

### Prerequisites

| Tool | Tested version | Notes |
| --- | --- | --- |
| Python | **3.13.3** | Match the artifact's scikit-learn build. |
| Node.js | **22.16** | Node 20+ should work; npm 10+ is recommended. |
| Git | Any recent version | |
| RAM | ~2 GB free | Full training peaks around 1.5–2 GB. |
| Dataset | `combined_data.csv` | Needed only for retraining. |

The artifact was trained with scikit-learn 1.9.1, NumPy 2.5.4, and Python 3.13.
A different Python/scikit-learn combination may fail to unpickle; retrain in
that case.

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

If PowerShell blocks activation:

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

To install only one part:

```bash
pip install -r backend/requirements.txt   # API only
pip install -r ml/requirements.txt        # training + notebook only
```

### 2. Add the dataset

Place `combined_data.csv` in the repository root. It needs two columns:

| Column | Meaning |
| --- | --- |
| `text` | message body |
| `label` | `1` = spam, `0` = ham |

The file is gitignored because it is large. The trained artifact is committed,
so you can run the API without it.

Optional short-message data: put the UCI SMS Spam Collection at
`data/sms_spam.zip`. Training picks it up automatically. Use `--sms-data PATH`
to point somewhere else.

```bash
python -m ml.train
```

### 3. Train the model

Skip this step to use the committed artifact.

```bash
python -m ml.train
```

Expected output and runtime:

- About 11 minutes on a modern laptop.
- Raw: 89,022 rows (83,448 main + 5,574 SMS) -> 81,245 clean rows (76,127 main + 5,118 SMS).
- Removed: 1,129 exact duplicates, 6,597 near duplicates, and 51 empty rows.
- Labels: 41,346 ham / 39,899 spam.
- Split: 64,995 train / 8,125 validation / 8,125 test.
- Writes `ml/artifacts/spam_pipeline.joblib` and `ml/artifacts/metrics.json`.

Fast smoke run:

```bash
python -m ml.train --sample 5000 --output ml/artifacts-smoke
```

Useful flags:

```text
--data PATH             path to combined_data.csv
--sms-data PATH         optional SMS zip/TSV
--spamassassin PATH     SpamAssassin directory/archive for external evaluation
--output DIR            artifact directory (default: ml/artifacts)
--sample N              row cap for smoke tests
--no-near-dedup         skip SimHash removal (faster, but risks leakage)
--random-state N        control sampling and splitting
```

Training rewrites the artifact. Restart Django afterwards so each worker loads
the new model. The frontend reads the refreshed `metrics.json` on request.

### 4. Evaluate on SpamAssassin

Download the public corpus and evaluate the saved model:

```bash
python -m ml.external --download
python -m ml.external --evaluate --model ml/artifacts/spam_pipeline.joblib
```

The corpus is stored in `data/spamassassin/` and is gitignored. This evaluation
is separate from model selection and threshold tuning. Training can also run it
automatically with `--spamassassin data/spamassassin`.

### 5. Prepare the database

From the repository root:

```bash
python backend/manage.py migrate
```

This creates `backend/db.sqlite3`. A superuser is only needed for Django admin:

```bash
python backend/manage.py createsuperuser
```

### 6. Run the Django API

```bash
python backend/manage.py runserver 127.0.0.1:8000
```

Verify:

```bash
curl http://127.0.0.1:8000/api/health/
```

A healthy response includes `"status": "ok"`, `"model_loaded": true`, and
`"model_version": "2.1.0"`.

Useful endpoints:

- Swagger UI: <http://127.0.0.1:8000/api/docs/>
- OpenAPI schema: <http://127.0.0.1:8000/api/schema/>
- Django admin: <http://127.0.0.1:8000/admin/>

### 7. Run the React frontend

In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://127.0.0.1:3000>. Vite proxies `/api` to Django, so the browser sees
one origin.

Production build preview:

```bash
npm run build
npm run preview   # http://127.0.0.1:4173
```

### 8. Open the notebook

```bash
jupyter lab ml/notebooks/spam_classifier.ipynb
```

The notebook is committed with executed outputs. It covers data cleaning, the
leak-free split, the two feature views, candidate comparison, the FPR-capped
threshold, test metrics, explainability, external validation, and the production
artifact.

---

## Configuration

Backend settings are environment variables read at process start. Defaults work
for local development.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DJANGO_SECRET_KEY` | dev-only value | Required by Compose. Must be at least 32 characters when `DJANGO_DEBUG=0`. |
| `DJANGO_DEBUG` | `true` | Set `0` in production. |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1,backend,0.0.0.0` | Comma-separated hostnames. |
| `CORS_ALLOWED_ORIGINS` | local Vite origins | Needed only for direct cross-origin calls. |
| `CSRF_TRUSTED_ORIGINS` | local app origins | Trusted origins for unsafe requests. |
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
| `VITE_API_BASE_URL` | `/api` | API prefix. Keep `/api` behind Vite/nginx. |

Docker Compose reads `.env` from the repository root. `.env.example` shows the
required values. Environment changes require a restart:

```bash
docker compose up -d --force-recreate backend
```

---

## API reference

Base URL: `http://127.0.0.1:8000/api`

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health/` | Database + model readiness. Returns `503` if the model cannot load. |
| `POST` | `/predict/` | Classify one message and save it. JSON body: `{"text": "..."}`. |
| `POST` | `/predict/batch/` | Classify a CSV. Multipart fields: `file`, optional `text_column`. |
| `GET` | `/history/` | Paginated predictions. Filters: `label`, `source`, `search`. |
| `GET` | `/stats/` | Counts, spam rate, 7-day trend, sources. |
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
  "spam_probability": 0.992315,
  "ham_probability": 0.007685,
  "threshold": 0.7219,
  "confidence": 0.9923,
  "top_words": [
    {"word": "your", "weight": 0.3415},
    {"word": "money", "weight": 0.3326},
    {"word": "click here", "weight": 0.2448}
  ],
  "signals": {
    "toward_spam": [{"word": "your", "weight": 0.3415}],
    "toward_ham": [{"word": "congratulations", "weight": -0.4326}]
  },
  "model_version": "2.1.0",
  "text_length": 74,
  "created_at": "2026-10-24T09:31:10.090443Z"
}
```

`top_words` lists the largest signed contributions for this message. Positive
weights push toward spam; negative weights pull toward ham. The model is linear,
so these are exact contributions.

### Batch prediction

```bash
curl -s http://127.0.0.1:8000/api/predict/batch/ \
  -F "file=@messages.csv" \
  -F "text_column=text"
```

The response reports `count`, `skipped_empty`, `spam_count`, `ham_count`,
`text_column`, the threshold, model version, and a `results` array with `row`,
`text_preview`, `label`, `spam_probability`, and `threshold`. Each non-empty row
is also stored with source `batch`.

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

Local filesystem paths are removed from the model-info response.

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

Only the message column is required. Extra columns (including `label`) are
ignored.

```csv
text,label
"Hi team, the design review moved to Thursday at 10:00.",ham
"URGENT: verify your account and claim your $1,000 reward now!",spam
```

Rules:

- The first row is the header.
- Quote every field that contains a comma.
- Auto-detection looks for `text`, `email`, `email_content`, `message`,
  `content`, `body`, or `email_text`; otherwise it uses the first column. Pass
  `text_column` to override.
- Empty or whitespace-only rows are skipped and counted.
- Maximum 5 MB and 500 data rows per upload.
- UTF-8 (with or without BOM) and Latin-1 are accepted.

---

## Tests

Backend API tests use a small local fixture, not the production artifact:

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
| Main corpus | `combined_data.csv` (83,448 raw rows; 76,127 after cleaning) |
| Short messages | UCI SMS Spam Collection (5,574 raw rows; 5,118 after cleaning) |
| Raw rows | 89,022 |
| Clean rows | 81,245 |
| Removed | 1,129 exact + 6,597 SimHash near-duplicates; 51 empty rows |
| Labels | 41,346 ham / 39,899 spam |
| Median length | 764 characters |
| Mean length | 1,393 characters |
| Max length | 20,000 characters (truncated) |
| Split | 80 / 10 / 10 stratified, created before fitting |

The SMS source is included only when `data/sms_spam.zip` exists or `--sms-data`
is passed. The dataset summary in `metrics.json` stores source names and counts,
not local paths.

### Features

| View | Configuration | Weight |
| --- | --- | --- |
| Word TF-IDF | 1–2 grams, 150k features, `min_df=2`, sublinear TF | 1.00 |
| Character TF-IDF | `char_wb` 3–5 grams, 80k features, `min_df=5`, first 4,000 chars | 0.20 |

Total feature space: **230,000**. No dense text statistics are used by the
deployed model.

### Model selection

Three classical families were compared on validation:

| Candidate | Validation PR-AUC | Validation F0.5 | Fit time |
| --- | ---: | ---: | ---: |
| Calibrated Linear SVC (`C=0.5`) | 0.9988 | 0.9872 | 9.8s |
| **Logistic regression (`C=0.25`)** | **0.9966** | **0.9646** | **13.9s** |
| Complement NB (`alpha=0.3`) | 0.9946 | 0.9693 | 0.2s |

Calibrated SVC had the highest validation PR-AUC, but the gap was 0.0022, below
the 0.003 selection tolerance. The project therefore chose native logistic
regression: calibrated probabilities, one coefficient per feature, and a more
stable cross-corpus decision. Among tied regularisation values, the strongest
regularisation (`C=0.25`) was preferred. The final artifact was refit on train +
validation.

The threshold (`0.7219`) maximises F0.5 on validation subject to validation FPR
≤ 0.5%. The test split was untouched until the final evaluation.

### Slices

The aggregate score hides weak spots:

| Slice | Accuracy | Precision | Recall |
| --- | ---: | ---: | ---: |
| Long messages (500+ chars) | 95.47% | 99.42% | 90.77% |
| Medium (160–499 chars) | 93.16% | 100.00% | 89.15% |
| Short (<160 chars) | 84.30% | 97.05% | 62.32% |
| SMS source | 92.48% | 90.70% | 51.32% |

Short messages and SMS-style text are the weak point. The model is conservative:
it misses more spam than it blocks real mail, but short spam often scores below
the tuned cut-off.

### Top coefficients

Some strong spam terms are expected (`your`, `urltoken`, `our`, `money`,
`viagra`). Some ham terms are corpus-specific (`enron`, `debian`, `perl`,
`samba`, `louise`). This is a known limitation of training on public email
archives. Treat the explanation as a signal, not a full audit.

### Artifact

- `ml/artifacts/spam_pipeline.joblib` — ~4.2 MB; pure scikit-learn pipeline.
- `ml/artifacts/metrics.json` — dataset stats, candidate comparison, threshold,
  test metrics, external metrics, feature list, library versions.
- Batch inference: ~2.65 ms per message in the training environment. A single
  API request adds explanation and database latency.

### Limitations

- Dataset drift: vocabulary and campaigns change; retrain and monitor false
  positives.
- Body-only analysis: no SPF/DKIM, sender reputation, attachments, or link
  reputation.
- Short messages are hard; treat borderline scores as review candidates.
- Character n-grams help with obfuscation but do not solve it.
- Near-duplicate removal protects the test score but can discard legitimate
  templates.
- External precision and FPR are worse than the in-dataset test. That is
  expected and reported honestly.
- The API is unauthenticated by default. Add auth and per-user quotas before
  exposing it publicly.

---

## Troubleshooting

### `503 The classifier is not available`

The artifact is missing or cannot be loaded.

```bash
ls ml/artifacts/spam_pipeline.joblib
python -m ml.train          # only if you have the dataset
```

Check `MODEL_PATH` and restart Django. In Docker, confirm the artifact exists
locally and rebuild the backend image.

### `InconsistentVersionWarning` or unpickle errors

Train with the same Python/scikit-learn versions as the artifact:

```bash
python --version            # 3.13.x
pip install -r requirements.txt
```

If you upgraded scikit-learn, retrain the model.

### Compose refuses to start

Set a real secret in `.env`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

The placeholder in `.env.example` is rejected on purpose. Compose also fails
when `DJANGO_SECRET_KEY` is unset.

### CORS errors in the browser

Use the Vite dev server (`npm run dev`) or Docker nginx, which proxy `/api` and
keep everything same-origin. For direct cross-origin calls, add the frontend
origin to `CORS_ALLOWED_ORIGINS` and restart Django.

### Vite fails with `EACCES` or the port is unavailable

Windows may reserve ranges that include common Vite ports. Change
`server.port` in `frontend/vite.config.js`. Keep the `/api` proxy target aligned
with the Django port.

### `Column '...' was not found` or unquoted-comma CSV errors

Check the header name and quote every field containing a comma. The API rejects
rows with more fields than the header instead of truncating messages.

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
is excluded from the image; only the artifact is copied.

### Prediction history contains test rows

For local development, stop the API and delete `backend/db.sqlite3`, then rerun
`python backend/manage.py migrate`. For Docker, use `docker compose down -v`.

---

## Deployment notes

- Set `DJANGO_DEBUG=0`, a strong `DJANGO_SECRET_KEY`, and explicit
  `DJANGO_ALLOWED_HOSTS` / `CSRF_TRUSTED_ORIGINS`.
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

No open-source license file is included. Add one before distributing the project.

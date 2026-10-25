import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import {
  AlertIcon,
  ArrowRightIcon,
  CheckIcon,
  ChipIcon,
  DownloadIcon,
  RefreshIcon,
  ShieldIcon,
  SparkIcon,
} from "../components/Icons.jsx";
import { PageHeader, Spinner, EmptyState, StatCard } from "../components/UI.jsx";
import { SignalChips } from "../components/SignalChips.jsx";
import { compactNumber, dateOnly, fixed, integer, percent } from "../lib/format.js";

const FAMILY_LABELS = {
  LogisticRegression: "Logistic regression",
  CalibratedLinearSVC: "Calibrated SVC",
  ComplementNB: "Complement NB",
};

function familyLabel(name = "") {
  return FAMILY_LABELS[name] || name.replace(/([a-z])([A-Z])/g, "$1 $2");
}

function MetricGrid({ metrics }) {
  const items = [
    { label: "Accuracy", value: percent(metrics?.accuracy), hint: "test split" },
    { label: "Precision", value: percent(metrics?.precision), hint: "of spam calls, how many were right", tone: "cool" },
    { label: "Recall", value: percent(metrics?.recall), hint: "spam caught", tone: "warm" },
    { label: "F1", value: fixed(metrics?.f1, 4), hint: "harmonic mean" },
    { label: "PR-AUC", value: fixed(metrics?.average_precision, 4), hint: "threshold independent" },
    { label: "ROC-AUC", value: fixed(metrics?.roc_auc, 4), hint: "ranking quality" },
    { label: "F0.5", value: fixed(metrics?.f0_5, 4), hint: "precision weighted" },
    { label: "Brier", value: fixed(metrics?.brier, 4), hint: "calibration error (lower is better)" },
  ];
  return (
    <div className="metric-grid">
      {items.map((item) => (
        <StatCard key={item.label} {...item} />
      ))}
    </div>
  );
}

function ConfusionMatrix({ matrix }) {
  if (!matrix) return null;
  const { tn, fp, fn, tp } = matrix;
  return (
    <div className="confusion-card">
      <div className="section-heading">
        <h2>Confusion matrix</h2>
        <span>test split · tuned cut-off</span>
      </div>
      <div className="confusion" role="table" aria-label="Confusion matrix">
        <div className="confusion__corner" />
        <div className="confusion__head">Predicted ham</div>
        <div className="confusion__head">Predicted spam</div>

        <div className="confusion__side">Actual ham</div>
        <div className="confusion__cell confusion__cell--good">
          <strong>{integer(tn)}</strong>
          <span>true negative</span>
        </div>
        <div className="confusion__cell confusion__cell--bad">
          <strong>{integer(fp)}</strong>
          <span>false positive · costly</span>
        </div>

        <div className="confusion__side">Actual spam</div>
        <div className="confusion__cell confusion__cell--bad">
          <strong>{integer(fn)}</strong>
          <span>false negative</span>
        </div>
        <div className="confusion__cell confusion__cell--good">
          <strong>{integer(tp)}</strong>
          <span>true positive</span>
        </div>
      </div>
      <p className="confusion__note">
        <AlertIcon size={15} />
        False positives are real mail marked spam. The cut-off is tuned to keep this number low — {integer(fp)} in {integer(tn + fp)} ham messages here.
      </p>
    </div>
  );
}

function ExternalCard({ external }) {
  const metrics = external?.at_tuned_threshold;
  const dataset = external?.dataset;
  if (!metrics) return null;

  return (
    <section className="card table-card">
      <div className="section-heading">
        <h2>External check</h2>
        <span>SpamAssassin · not used for tuning</span>
      </div>
      <div className="dataset-figures">
        <div>
          <span className="dataset-figures__value">{percent(metrics.precision)}</span>
          <span className="dataset-figures__label">precision</span>
        </div>
        <div>
          <span className="dataset-figures__value">{percent(metrics.recall)}</span>
          <span className="dataset-figures__label">recall</span>
        </div>
        <div>
          <span className="dataset-figures__value">{percent(metrics.false_positive_rate, 2)}</span>
          <span className="dataset-figures__label">false-positive rate</span>
        </div>
      </div>
      <p className="table-note">
        {integer(dataset?.label_counts?.spam)} spam and {integer(dataset?.label_counts?.ham)} ham messages from a
        separate public corpus. The external result is reported honestly and was not used for model selection or
        threshold tuning.
      </p>
    </section>
  );
}

function DatasetCard({ dataset, split }) {
  if (!dataset) return null;
  const total = dataset.label_counts?.ham + dataset.label_counts?.spam || 1;
  const hamShare = ((dataset.label_counts?.ham || 0) / total) * 100;
  const splitTotal = (split?.train || 0) + (split?.validation || 0) + (split?.test || 0) || 1;

  return (
    <div className="card dataset-card">
      <div className="section-heading">
        <h2>Training data</h2>
        <span>{dataset.rows_loaded ? `${integer(dataset.rows_loaded)} raw rows` : ""}</span>
      </div>

      <div className="dataset-figures">
        <div>
          <span className="dataset-figures__value">{integer(dataset.rows_clean)}</span>
          <span className="dataset-figures__label">clean messages after de-duplication</span>
        </div>
        <div>
          <span className="dataset-figures__value">{integer((dataset.exact_duplicates_removed || 0) + (dataset.near_duplicates_removed || 0))}</span>
          <span className="dataset-figures__label">duplicates removed</span>
        </div>
        <div>
          <span className="dataset-figures__value">{integer(dataset.text_length?.median)}</span>
          <span className="dataset-figures__label">median characters</span>
        </div>
      </div>

      <div className="split-bar" aria-label="Label balance">
        <span className="split-bar__ham" style={{ width: `${hamShare}%` }} />
        <span className="split-bar__spam" style={{ width: `${100 - hamShare}%` }} />
      </div>
      <div className="split-legend">
        <span><i className="legend-swatch legend-swatch--ham" /> Ham {integer(dataset.label_counts?.ham)}</span>
        <span><i className="legend-swatch legend-swatch--spam" /> Spam {integer(dataset.label_counts?.spam)}</span>
      </div>

      <div className="split-cols">
        {[
          ["Train", split?.train],
          ["Validation", split?.validation],
          ["Test", split?.test],
        ].map(([name, value]) => (
          <div key={name} className="split-col">
            <span>{name}</span>
            <strong>{integer(value)}</strong>
            <span className="split-col__bar">
              <i style={{ width: `${((value || 0) / splitTotal) * 100}%` }} />
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ModelPage({ health }) {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let ignore = false;
    setLoading(true);
    setError("");
    api
      .modelInfo()
      .then((payload) => {
        if (!ignore) setInfo(payload);
      })
      .catch((requestError) => {
        if (!ignore) setError(requestError.message);
      })
      .finally(() => {
        if (!ignore) setLoading(false);
      });
    return () => {
      ignore = true;
    };
  }, [reloadKey]);

  const metrics = info?.test?.at_tuned_threshold || {};
  const matrix = info?.confusion_matrix || metrics?.confusion_matrix;
  const comparison = info?.comparison || [];
  const model = info?.model;

  return (
    <div className="page page--model">
      <PageHeader
        eyebrow="Model card"
        title={<>The brain, <em>in the open.</em></>}
        description="Version, evaluation metrics, the data it learned from, and the words that carry the most weight."
        actions={
          <>
            <button type="button" className="ghost-button" onClick={() => setReloadKey((value) => value + 1)}>
              <RefreshIcon size={16} /> Refresh
            </button>
            <a className="ghost-button" href="/api/docs/" target="_blank" rel="noreferrer">
              API docs <ArrowRightIcon size={15} />
            </a>
          </>
        }
      />

      {loading ? (
        <div className="card loading-card"><Spinner size={26} /> Loading model metadata…</div>
      ) : error ? (
        <div className="card">
          <EmptyState
            icon={<AlertIcon size={24} />}
            title="Could not load the model card"
            description={error}
          >
            <button type="button" className="ghost-button" onClick={() => setReloadKey((value) => value + 1)}>
              Try again
            </button>
          </EmptyState>
        </div>
      ) : !info?.model ? (
        <div className="card">
          <EmptyState
            icon={<ChipIcon size={24} />}
            title="No trained model yet"
            description="Train the classifier, then refresh this page."
          >
            <code className="code-block">python -m ml.train</code>
          </EmptyState>
        </div>
      ) : (
        <>
          <section className="card model-hero">
            <div className="model-hero__main">
              <div className="model-hero__title">
                <span className="model-hero__mark"><ChipIcon size={26} /></span>
                <div>
                  <p className="eyebrow">Classifier</p>
                  <h2>{familyLabel(model.family)}<span>v{info.version}</span></h2>
                </div>
              </div>
              <p className="model-hero__copy">
                {familyLabel(model.family)} over word and character TF-IDF. No length or style statistics.
                Trained {dateOnly(info.trained_at)} in {fixed(info.training_seconds / 60, 1)} minutes.
              </p>

              <div className="model-hero__facts">
                <div><span>Decision cut-off</span><strong>{percent(model.threshold, 1)}</strong></div>
                <div><span>Artifact</span><strong>{fixed(info.artifact?.size_mb, 1)} MB</strong></div>
                <div><span>Latency</span><strong>{fixed(info.latency_ms_per_email, 1)} ms</strong></div>
                <div><span>Features</span><strong>{compactNumber(info.ablation?.full_feature_union?.feature_count)}</strong></div>
              </div>
            </div>

            <div className="model-hero__side">
              <div className={`runtime-card runtime-card--${health?.state || "checking"}`}>
                <div className="runtime-card__head">
                  <span className="status-dot" />
                  <span>{health?.state === "online" ? "Serving live" : health?.state === "offline" ? "API offline" : "Checking…"}</span>
                </div>
                <p>
                  {health?.state === "online"
                    ? `Model ${health.version || info.version} loaded · cut-off ${percent(health.threshold ?? model.threshold, 0)}`
                    : "The web layer is not connected to Django right now."}
                </p>
              </div>
              <div className="architecture">
                <div className="architecture__row">
                  <span>word TF-IDF</span>
                  <span>char TF-IDF</span>
                </div>
                <div className="architecture__row architecture__row--flow">
                  <i className="architecture__arrow">→</i>
                  <strong>{familyLabel(model.family)}</strong>
                  <i className="architecture__arrow">→</i>
                  <em>cut {percent(model.threshold, 0)}</em>
                </div>
              </div>
            </div>
          </section>

          <MetricGrid metrics={metrics} />

          <div className="model-grid">
            <ConfusionMatrix matrix={matrix} />
            <DatasetCard dataset={info.dataset} split={info.split} />
          </div>

          <ExternalCard external={info.external} />

          <div className="model-grid model-grid--features">
            <div className="card feature-card">
              <div className="section-heading">
                <h2>Top spam signals</h2>
                <span>global coefficients</span>
              </div>
              <SignalChips words={(info.top_features?.spam || []).slice(0, 12)} />
              <p className="feature-card__note">Positive weights push a message toward spam. Character patterns are marked <em>char</em>.</p>
            </div>

            <div className="card feature-card">
              <div className="section-heading">
                <h2>Top ham signals</h2>
                <span>global coefficients</span>
              </div>
              <SignalChips words={(info.top_features?.ham || []).slice(0, 12)} />
              <p className="feature-card__note">Negative weights pull a message back toward legitimate mail.</p>
            </div>
          </div>

          <section className="card table-card">
            <div className="section-heading">
              <h2>Model comparison</h2>
              <span>validation split · PR-AUC + parsimony</span>
            </div>
            <div className="table-scroll">
              <table className="data-table comparison-table">
                <thead>
                  <tr>
                    <th>Candidate</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>PR-AUC</th>
                    <th>Fit time</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.map((row) => (
                    <tr key={row.name} className={row.name === model.family ? "is-best" : ""}>
                      <td>
                        <span className="candidate-name">
                          {row.name === model.family ? <CheckIcon size={15} /> : <SparkIcon size={15} />}
                          {familyLabel(row.name)}
                        </span>
                        <span className="candidate-params">
                          {Object.entries(row.params || {}).map(([key, value]) => `${key}=${value}`).join(" · ")}
                        </span>
                      </td>
                      <td>{percent(row.precision, 2)}</td>
                      <td>{percent(row.recall, 2)}</td>
                      <td>{fixed(row.f1, 4)}</td>
                      <td>{fixed(row.average_precision, 4)}</td>
                      <td>{fixed(row.fit_seconds, 1)}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="table-note">
              Initial family comparison shown. The winning family was then tuned on validation,
              then refit on train + validation for the deployed artifact.
            </p>
          </section>

          <div className="model-grid model-grid--notes">
            <div className="card note-card">
              <div className="section-heading">
                <h2>How it was built</h2>
                <span>leak-free pipeline</span>
              </div>
              <ul className="check-list">
                <li><CheckIcon size={15} /> {integer(info.dataset?.exact_duplicates_removed)} exact and {integer(info.dataset?.near_duplicates_removed)} near duplicates removed before splitting.</li>
                <li><CheckIcon size={15} /> Vectorizers stayed on the training fold while tuning; the final artifact refits on train + validation. Test stayed untouched.</li>
                <li><CheckIcon size={15} /> Decision cut-off chosen on validation to maximise F0.5, favouring precision.</li>
                <li><CheckIcon size={15} /> {comparison.length} classical models compared — no deep learning, no GPU.</li>
              </ul>
            </div>

            <div className="card note-card">
              <div className="section-heading">
                <h2>Reproduce</h2>
                <span>one command</span>
              </div>
              <code className="code-block">python -m ml.train</code>
              <p className="note-card__copy">
                The artifact and <code>metrics.json</code> land in <code>ml/artifacts/</code>. Restart or refresh the backend to serve a new model.
              </p>
              <a className="text-link" href="/api/schema/" target="_blank" rel="noreferrer">
                <DownloadIcon size={15} /> OpenAPI schema <ArrowRightIcon size={14} />
              </a>
            </div>
          </div>

          <div className="model-footnote">
            <ShieldIcon size={16} />
            <p>
              Trained on the combined spam corpus ({integer(info.dataset?.rows_clean)} clean messages). Test metrics are from a held-out 10% split;
              real-world mail drifts, so monitor false positives after deployment.
            </p>
          </div>
        </>
      )}
    </div>
  );
}

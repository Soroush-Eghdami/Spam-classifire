import React, { useMemo, useRef, useState } from "react";
import { api } from "../api.js";
import {
  AlertIcon,
  ArrowRightIcon,
  DownloadIcon,
  FileIcon,
  LayersIcon,
  RefreshIcon,
  SearchIcon,
  UploadIcon,
} from "../components/Icons.jsx";
import { PageHeader, LabelPill, Spinner, EmptyState, StatCard } from "../components/UI.jsx";
import { ProbabilityBar } from "../components/Probability.jsx";
import { useToast } from "../components/Toast.jsx";
import { downloadCsv, fileSize, integer, percent } from "../lib/format.js";

const MAX_BYTES = 5 * 1024 * 1024;
const FILTERS = [
  { id: "all", label: "All" },
  { id: "spam", label: "Spam" },
  { id: "ham", label: "Ham" },
];

export default function BatchPage() {
  const [file, setFile] = useState(null);
  const [column, setColumn] = useState("");
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  const inputRef = useRef(null);
  const toast = useToast();

  function chooseFile(candidate) {
    if (!candidate) return;
    if (!candidate.name.toLowerCase().endsWith(".csv")) {
      setError("Choose a .csv file. Export spreadsheets to CSV first.");
      return;
    }
    if (candidate.size > MAX_BYTES) {
      setError(`That file is ${(candidate.size / (1024 * 1024)).toFixed(1)} MB. The limit is 5 MB.`);
      return;
    }
    setError("");
    setResult(null);
    setFile(candidate);
  }

  function onDrop(event) {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files?.[0]);
  }

  async function submit(event) {
    event.preventDefault();
    if (!file) {
      setError("Choose a CSV file first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const data = await api.predictBatch(file, column);
      setResult(data);
      setFilter("all");
      setQuery("");
      toast(`${data.count} message${data.count === 1 ? "" : "s"} classified.`, "success");
    } catch (requestError) {
      setError(requestError.message);
      toast(requestError.message, "error");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFile(null);
    setResult(null);
    setError("");
    setColumn("");
    setQuery("");
    setFilter("all");
    if (inputRef.current) inputRef.current.value = "";
  }

  const visible = useMemo(() => {
    if (!result) return [];
    const needle = query.trim().toLowerCase();
    return result.results.filter((row) => {
      if (filter !== "all" && row.label !== filter) return false;
      if (needle && !row.text_preview.toLowerCase().includes(needle)) return false;
      return true;
    });
  }, [result, filter, query]);

  function download() {
    if (!result) return;
    const rows = [
      ["row", "label", "spam_probability", "text_preview"],
      ...result.results.map((row) => [
        row.row,
        row.label,
        row.spam_probability,
        row.text_preview,
      ]),
    ];
    downloadCsv("sift-batch-results.csv", rows);
  }

  const spamRate = result && result.count ? result.spam_count / result.count : 0;

  return (
    <div className="page page--batch">
      <PageHeader
        eyebrow="Batch triage"
        title={<>One file. <em>Every verdict.</em></>}
        description="Upload a CSV of messages and classify up to 500 rows in a single vectorised pass. Results stay in your history and download as CSV."
      />

      <div className="batch-layout">
        <section className="card upload-card">
          <div className="section-heading">
            <h2>Upload</h2>
            <span>CSV · max 5 MB / 500 rows</span>
          </div>

          <form onSubmit={submit}>
            <div
              className={`dropzone${dragging ? " is-dragging" : ""}${file ? " has-file" : ""}`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  inputRef.current?.click();
                }
              }}
              role="button"
              tabIndex={0}
              aria-label="Choose a CSV file"
            >
              <input
                ref={inputRef}
                type="file"
                accept=".csv,text/csv"
                hidden
                onChange={(event) => {
                  chooseFile(event.target.files?.[0]);
                  event.target.value = "";
                }}
              />
              <span className="dropzone__icon">
                {file ? <FileIcon size={26} /> : <UploadIcon size={26} />}
              </span>
              {file ? (
                <>
                  <strong>{file.name}</strong>
                  <span className="dropzone__meta">
                    {fileSize(file.size)} · click or drop to replace
                  </span>
                </>
              ) : (
                <>
                  <strong>Drop a CSV here</strong>
                  <span className="dropzone__meta">or click to browse · fields with commas must be quoted</span>
                </>
              )}
            </div>

            <label className="field">
              <span className="field__label">Message column <em>optional</em></span>
              <input
                type="text"
                value={column}
                onChange={(event) => setColumn(event.target.value)}
                placeholder="Auto-detect (text, email, message, content…)"
                maxLength={80}
              />
            </label>

            {error ? (
              <p className="form-error" role="alert">
                <AlertIcon size={15} /> {error}
              </p>
            ) : null}

            <div className="upload-card__actions">
              {file ? (
                <button type="button" className="ghost-button" onClick={reset}>
                  Clear
                </button>
              ) : null}
              <button type="submit" className="primary-button" disabled={loading || !file}>
                {loading ? <Spinner size={17} /> : <LayersIcon size={17} />}
                {loading ? "Classifying…" : "Classify file"}
                {!loading ? <ArrowRightIcon size={16} /> : null}
              </button>
            </div>
          </form>

          <p className="upload-card__hint">
            Empty rows are skipped. Each prediction is stored with source <code>batch</code> and can be filtered in History.
          </p>
        </section>

        <section className="card batch-summary">
          <div className="section-heading">
            <h2>Summary</h2>
            <span>{result ? result.text_column : "waiting for a file"}</span>
          </div>

          {result ? (
            <>
              <div className="stat-grid stat-grid--two">
                <StatCard label="Rows" value={integer(result.count)} hint={`${integer(result.skipped_empty)} empty skipped`} />
                <StatCard label="Spam rate" value={percent(spamRate)} hint={`cut-off ${percent(result.threshold, 0)}`} tone="warm" />
                <StatCard label="Spam" value={integer(result.spam_count)} tone="warm" />
                <StatCard label="Ham" value={integer(result.ham_count)} tone="cool" />
              </div>
              <div className="batch-summary__foot">
                <span>model {result.model_version}</span>
                <button type="button" className="ghost-button" onClick={download}>
                  <DownloadIcon size={15} /> Download CSV
                </button>
              </div>
            </>
          ) : (
            <EmptyState
              compact
              icon={<LayersIcon size={24} />}
              title="No batch yet"
              description="The row count, spam rate, and a downloadable result table appear here after you upload a file."
            />
          )}
        </section>
      </div>

      {result ? (
        <section className="card table-card">
          <div className="table-toolbar">
            <div className="segmented" role="group" aria-label="Filter results">
              {FILTERS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={filter === item.id ? "is-active" : ""}
                  onClick={() => setFilter(item.id)}
                >
                  {item.label}
                  <span>
                    {item.id === "all"
                      ? result.count
                      : item.id === "spam"
                        ? result.spam_count
                        : result.ham_count}
                  </span>
                </button>
              ))}
            </div>

            <label className="search-field">
              <SearchIcon size={16} />
              <input
                type="search"
                placeholder="Search previews…"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
            </label>

            <button type="button" className="ghost-button" onClick={download}>
              <DownloadIcon size={15} /> Export
            </button>
          </div>

          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="col-row">Row</th>
                  <th>Message preview</th>
                  <th className="col-label">Verdict</th>
                  <th className="col-prob">Spam probability</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((row) => (
                  <tr key={row.row}>
                    <td className="col-row mono">{row.row}</td>
                    <td><span className="preview-text">{row.text_preview}</span></td>
                    <td className="col-label"><LabelPill label={row.label} /></td>
                    <td className="col-prob">
                      <ProbabilityBar probability={row.spam_probability} threshold={result.threshold} label={row.label} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {visible.length === 0 ? (
            <EmptyState
              compact
              icon={<SearchIcon size={22} />}
              title="No rows match"
              description="Try a different verdict filter or search term."
            />
          ) : null}
        </section>
      ) : null}

      {!result ? (
        <section className="process-strip batch-tips" aria-label="Batch upload tips">
          <div className="process-step">
            <span className="process-step__index">01</span>
            <div>
              <h3>Quote your text</h3>
              <p>Messages containing commas need double quotes around the field, or the CSV becomes ambiguous.</p>
            </div>
          </div>
          <div className="process-step">
            <span className="process-step__index">02</span>
            <div>
              <h3>Name the column</h3>
              <p>Auto-detection looks for text, email, message, content, or body. Override it in the field above.</p>
            </div>
          </div>
          <div className="process-step">
            <span className="process-step__index">03</span>
            <div>
              <h3>Export and audit</h3>
              <p>Every row is stored in History, and the whole result set downloads as a clean CSV.</p>
            </div>
          </div>
        </section>
      ) : null}
    </div>
  );
}

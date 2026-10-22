import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api.js";
import {
  AlertIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  HistoryIcon,
  RefreshIcon,
  SearchIcon,
  XIcon,
} from "../components/Icons.jsx";
import { PageHeader, LabelPill, Spinner, EmptyState, StatCard } from "../components/UI.jsx";
import { ProbabilityBar } from "../components/Probability.jsx";
import { SignalChips } from "../components/SignalChips.jsx";
import { dateTime, integer, percent, timeAgo } from "../lib/format.js";

const LABEL_FILTERS = [
  { id: "", label: "All" },
  { id: "spam", label: "Spam" },
  { id: "ham", label: "Ham" },
];

function Drawer({ prediction, onClose }) {
  const closeRef = useRef(null);

  useEffect(() => {
    const onKey = (event) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    closeRef.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!prediction) return null;

  return (
    <div className="drawer-layer" role="dialog" aria-modal="true" aria-labelledby="drawer-title">
      <button type="button" className="drawer-backdrop" aria-label="Close details" onClick={onClose} />
      <aside className="drawer">
        <div className="drawer__head">
          <div>
            <p className="eyebrow">Prediction #{prediction.id}</p>
            <h2 id="drawer-title"><LabelPill label={prediction.label} size="lg" /></h2>
          </div>
          <button ref={closeRef} type="button" className="icon-button" onClick={onClose} aria-label="Close details">
            <XIcon size={18} />
          </button>
        </div>

        <div className="drawer__body">
          <div className="drawer__probability">
            <div>
              <span className="drawer__big">{percent(prediction.spam_probability)}</span>
              <span className="drawer__caption">spam probability</span>
            </div>
            <ProbabilityBar
              probability={prediction.spam_probability}
              threshold={prediction.threshold}
              label={prediction.label}
            />
            <p>Decision cut-off {percent(prediction.threshold, 1)}</p>
          </div>

          <div className="drawer__preview">
            <h3>Stored preview</h3>
            <p>{prediction.text_preview || "Preview unavailable."}</p>
          </div>

          <div className="drawer__signals">
            <div className="section-heading">
              <h3>Signals</h3>
              <span>signed contribution</span>
            </div>
            <SignalChips
              words={prediction.top_words}
              emptyLabel="This prediction was stored without word-level signals (batch rows keep summaries only)."
            />
          </div>

          <dl className="drawer__meta">
            <div><dt>Source</dt><dd>{prediction.source === "batch" ? "Batch upload" : "Single message"}</dd></div>
            <div><dt>Characters</dt><dd>{integer(prediction.text_length)}</dd></div>
            <div><dt>Model</dt><dd>{prediction.model_version}</dd></div>
            <div><dt>Created</dt><dd>{dateTime(prediction.created_at)}</dd></div>
          </dl>
        </div>
      </aside>
    </div>
  );
}

export default function HistoryPage() {
  const [page, setPage] = useState(1);
  const [label, setLabel] = useState("");
  const [source, setSource] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [data, setData] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(null);
  const [reloadKey, setReloadKey] = useState(0);
  const pageSize = 8;

  useEffect(() => {
    const timer = window.setTimeout(() => setSearch(searchInput.trim()), 350);
    return () => window.clearTimeout(timer);
  }, [searchInput]);

  useEffect(() => {
    let ignore = false;
    setLoading(true);
    setError("");
    api
      .history({ page, pageSize, label, source, search })
      .then((payload) => {
        if (!ignore) setData(payload);
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
  }, [page, label, source, search, reloadKey]);

  useEffect(() => {
    let ignore = false;
    api
      .stats()
      .then((payload) => {
        if (!ignore) setStats(payload);
      })
      .catch(() => {});
    return () => {
      ignore = true;
    };
  }, [reloadKey, data]);

  const pageCount = useMemo(
    () => Math.max(1, Math.ceil((data?.count || 0) / pageSize)),
    [data?.count],
  );

  function changeLabel(next) {
    setLabel(next);
    setPage(1);
  }

  function changeSource(event) {
    setSource(event.target.value);
    setPage(1);
  }

  function refresh() {
    setReloadKey((value) => value + 1);
  }

  const rows = data?.results || [];

  return (
    <div className="page page--history">
      <PageHeader
        eyebrow="Audit trail"
        title={<>Every call, <em>remembered.</em></>}
        description="A paginated record of single and batch predictions. Only short previews are stored — never the full message."
        actions={
          <button type="button" className="ghost-button" onClick={refresh}>
            <RefreshIcon size={16} /> Refresh
          </button>
        }
      />

      <div className="stat-grid stat-grid--four">
        <StatCard label="Total predictions" value={integer(stats?.total ?? data?.count ?? 0)} hint="all time" />
        <StatCard
          label="Spam rate"
          value={percent(stats?.spam_rate ?? 0)}
          hint={`${integer(stats?.spam ?? 0)} spam · ${integer(stats?.ham ?? 0)} ham`}
          tone="warm"
        />
        <StatCard label="Last 24 hours" value={integer(stats?.last_24h ?? 0)} hint={`${integer(stats?.last_24h_spam ?? 0)} marked spam`} tone="cool" />
        <StatCard
          label="Mean spam score"
          value={percent(stats?.average_spam_probability ?? 0)}
          hint="across stored rows"
        />
      </div>

      <section className="card table-card">
        <div className="table-toolbar table-toolbar--history">
          <div className="segmented" role="group" aria-label="Filter by verdict">
            {LABEL_FILTERS.map((item) => (
              <button
                key={item.id || "all"}
                type="button"
                className={label === item.id ? "is-active" : ""}
                onClick={() => changeLabel(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>

          <label className="select-field">
            <span className="sr-only">Filter by source</span>
            <select value={source} onChange={changeSource}>
              <option value="">All sources</option>
              <option value="single">Single</option>
              <option value="batch">Batch</option>
            </select>
          </label>

          <label className="search-field">
            <SearchIcon size={16} />
            <input
              type="search"
              placeholder="Search previews…"
              value={searchInput}
              onChange={(event) => {
                setSearchInput(event.target.value);
                setPage(1);
              }}
            />
          </label>
        </div>

        {error ? (
          <p className="form-error form-error--block" role="alert">
            <AlertIcon size={15} /> {error}
          </p>
        ) : null}

        <div className="table-scroll">
          <table className="data-table history-table">
            <thead>
              <tr>
                <th>Message</th>
                <th className="col-label">Verdict</th>
                <th className="col-prob">Spam probability</th>
                <th className="col-source">Source</th>
                <th className="col-time">When</th>
              </tr>
            </thead>
            <tbody>
              {loading
                ? Array.from({ length: 5 }).map((_, index) => (
                    <tr key={`skeleton-${index}`} className="skeleton-row">
                      <td colSpan={5}><span className="skeleton-line" /></td>
                    </tr>
                  ))
                : rows.map((row) => (
                    <tr
                      key={row.id}
                      className="is-clickable"
                      tabIndex={0}
                      onClick={() => setSelected(row)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          setSelected(row);
                        }
                      }}
                    >
                      <td>
                        <span className="preview-text">{row.text_preview}</span>
                        <span className="preview-meta">#{row.id} · {integer(row.text_length)} chars</span>
                      </td>
                      <td className="col-label"><LabelPill label={row.label} /></td>
                      <td className="col-prob">
                        <ProbabilityBar probability={row.spam_probability} threshold={row.threshold} label={row.label} />
                      </td>
                      <td className="col-source">
                        <span className={`source-tag source-tag--${row.source}`}>{row.source}</span>
                      </td>
                      <td className="col-time" title={dateTime(row.created_at)}>{timeAgo(row.created_at)}</td>
                    </tr>
                  ))}
            </tbody>
          </table>
        </div>

        {!loading && rows.length === 0 && !error ? (
          <EmptyState
            icon={<HistoryIcon size={24} />}
            title="Nothing matches"
            description="No predictions fit these filters yet. Analyze a message or upload a batch to start the trail."
          />
        ) : null}

        <div className="pagination">
          <span className="pagination__meta">
            {data ? `${integer(data.count)} record${data.count === 1 ? "" : "s"} · page ${page} of ${pageCount}` : "—"}
          </span>
          <div className="pagination__buttons">
            <button
              type="button"
              className="icon-button"
              onClick={() => setPage((value) => Math.max(1, value - 1))}
              disabled={!data?.previous || loading}
              aria-label="Previous page"
            >
              <ChevronLeftIcon size={18} />
            </button>
            <button
              type="button"
              className="icon-button"
              onClick={() => setPage((value) => Math.min(pageCount, value + 1))}
              disabled={!data?.next || loading}
              aria-label="Next page"
            >
              <ChevronRightIcon size={18} />
            </button>
          </div>
        </div>
      </section>

      {selected ? <Drawer prediction={selected} onClose={() => setSelected(null)} /> : null}
    </div>
  );
}

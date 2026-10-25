import React, { useRef, useState } from "react";
import { api } from "../api.js";
import { AlertIcon, ArrowRightIcon, ScanIcon, ShieldIcon, SparkIcon } from "../components/Icons.jsx";
import { PageHeader, LabelPill, Spinner } from "../components/UI.jsx";
import { ProbabilityBar, ProbabilityGauge } from "../components/Probability.jsx";
import { SignalChips } from "../components/SignalChips.jsx";
import { useToast } from "../components/Toast.jsx";
import { integer, percent } from "../lib/format.js";

const MAX_CHARS = 20000;

const SAMPLES = {
  spam: `URGENT: Your account has been suspended. Verify your identity within 24 hours by visiting http://secure-account-verify.example and confirming your card details. You have been selected to receive a $1,000 reward — claim it now before this offer expires!`,
  ham: `Hi Sam,

Quick update on the launch: the staging build passed all checks this morning, and the design review is moved to Thursday at 10:00. Could you send the updated copy before then?

Thanks,
Maya`,
};

function EmptyVerdict() {
  return (
    <div className="verdict-empty">
      <div className="orbit" aria-hidden="true">
        <span className="orbit__ring orbit__ring--one" />
        <span className="orbit__ring orbit__ring--two" />
        <span className="orbit__core">
          <ScanIcon size={26} />
        </span>
      </div>
      <h2>Awaiting a message</h2>
      <p>Paste an email, a subject line, or a suspicious DM. The verdict, probability, and contributing words appear here.</p>
      <ul className="verdict-empty__list">
        <li><span>01</span> Verdict against a tuned cut-off</li>
        <li><span>02</span> Calibrated spam probability</li>
        <li><span>03</span> Signed word-level reasons</li>
      </ul>
    </div>
  );
}

function ResultPanel({ result }) {
  const spam = result.label === "spam";
  const cut = percent(result.threshold, 1);
  return (
    <div className={`verdict verdict--${result.label}`} aria-live="polite">
      <div className="verdict__head">
        <div className="verdict__head-left">
          <p className="eyebrow">Verdict</p>
          <LabelPill label={result.label} size="lg" />
        </div>
        <span className="verdict__version">model {result.model_version}</span>
      </div>

      <div className="verdict__hero">
        <ProbabilityGauge probability={result.spam_probability} threshold={result.threshold} label={result.label} />
        <div className="verdict__summary">
          <h2>{spam ? "This reads like spam." : "This looks like legitimate mail."}</h2>
          <p>
            {spam
              ? `The message scored above the ${cut} decision cut-off. The signals below pushed it over the line.`
              : `The message stayed below the ${cut} decision cut-off. Nothing crossed the spam line.`}
          </p>
          <dl className="meta-grid">
            <div>
              <dt>Cut-off</dt>
              <dd>{cut}</dd>
            </div>
            <div>
              <dt>Confidence</dt>
              <dd>{percent(result.confidence, 1)}</dd>
            </div>
            <div>
              <dt>Characters</dt>
              <dd>{integer(result.text_length)}</dd>
            </div>
          </dl>
        </div>
      </div>

      <div className="verdict__meter">
        <div className="verdict__meter-head">
          <span>Ham</span>
          <span>Spam</span>
        </div>
        <ProbabilityBar
          probability={result.spam_probability}
          threshold={result.threshold}
          label={result.label}
        />
        <p className="verdict__meter-note">
          The notch marks the {cut} cut-off. Anything to its right is labelled spam.
        </p>
      </div>

      <div className="signals-block">
        <div className="section-heading">
          <h3>Words that moved the needle</h3>
          <span>signed contribution</span>
        </div>
        <SignalChips words={result.top_words} />
        <div className="signals-legend">
          <span><i className="legend-swatch legend-swatch--spam" /> pushes toward spam</span>
          <span><i className="legend-swatch legend-swatch--ham" /> pulls toward ham</span>
        </div>
      </div>

      <p className="verdict__note">
        <AlertIcon size={15} />
        A statistical signal, not proof. Review borderline messages before acting on them.
      </p>
    </div>
  );
}

function LoadingVerdict() {
  return (
    <div className="verdict-skeleton" aria-busy="true">
      <Spinner size={28} />
      <p>Scoring the message…</p>
      <span>TF-IDF transform · linear model · threshold</span>
    </div>
  );
}

export default function AnalyzePage({ navigate }) {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const resultRef = useRef(null);
  const toast = useToast();

  const tooLong = text.length > MAX_CHARS;

  async function submit(event) {
    event?.preventDefault();
    const value = text.trim();
    if (!value) {
      setError("Paste a message first.");
      return;
    }
    if (tooLong) {
      setError(`Messages are limited to ${integer(MAX_CHARS)} characters.`);
      return;
    }

    setLoading(true);
    setError("");
    try {
      const data = await api.predict(text);
      setResult(data);
      window.setTimeout(() => {
        if (window.innerWidth < 960) {
          resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }, 80);
    } catch (requestError) {
      setError(requestError.message);
      toast(requestError.message, "error");
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(event) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      submit(event);
    }
  }

  return (
    <div className="page page--analyze">
      <PageHeader
        eyebrow="Single message"
        title={<>Read the message. <em>Show the reasoning.</em></>}
        description="A classical machine-learning filter that scores one email at a time and tells you which words moved the verdict."
        actions={
          <button type="button" className="ghost-button" onClick={() => navigate("model")}>
            View model card
            <ArrowRightIcon size={16} />
          </button>
        }
      />

      <div className="analyze-grid">
        <section className="card composer" aria-labelledby="composer-title">
          <div className="composer__head">
            <div>
              <p className="eyebrow">Compose</p>
              <h2 id="composer-title">The message</h2>
            </div>
            <span className={`char-count${tooLong ? " char-count--over" : ""}`}>
              {integer(text.length)} / {integer(MAX_CHARS)}
            </span>
          </div>

          <form onSubmit={submit}>
            <label className="sr-only" htmlFor="message-text">Message to classify</label>
            <textarea
              id="message-text"
              className="composer__textarea"
              value={text}
              onChange={(event) => setText(event.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Paste an email, subject line, or direct message…"
              spellCheck="false"
              maxLength={MAX_CHARS + 500}
            />

            <div className="composer__samples">
              <span>Try an example</span>
              <button type="button" onClick={() => setText(SAMPLES.spam)}>
                <SparkIcon size={14} /> Spam
              </button>
              <button type="button" onClick={() => setText(SAMPLES.ham)}>
                <ShieldIcon size={14} /> Ham
              </button>
              {text ? (
                <button type="button" className="composer__clear" onClick={() => { setText(""); setResult(null); setError(""); }}>
                  Clear
                </button>
              ) : null}
            </div>

            {error ? (
              <p className="form-error" role="alert">
                <AlertIcon size={15} /> {error}
              </p>
            ) : null}

            <div className="composer__foot">
              <span className="shortcut-hint"><kbd>⌘</kbd>/<kbd>Ctrl</kbd> + <kbd>Enter</kbd></span>
              <button type="submit" className="primary-button" disabled={loading || tooLong}>
                {loading ? <Spinner size={17} /> : <ScanIcon size={17} />}
                {loading ? "Analyzing…" : "Analyze message"}
                {!loading ? <ArrowRightIcon size={16} /> : null}
              </button>
            </div>
          </form>
        </section>

        <section className={`card verdict-card${result ? " is-filled" : ""}`} ref={resultRef} aria-labelledby="verdict-title">
          <h2 id="verdict-title" className="sr-only">Classification result</h2>
          {loading ? <LoadingVerdict /> : result ? <ResultPanel result={result} /> : <EmptyVerdict />}
        </section>
      </div>

      <section className="process-strip" aria-label="How the classifier works">
        <div className="process-step">
          <span className="process-step__index">01</span>
          <div>
            <h3>Two views of the text</h3>
            <p>Word phrases and character fragments. Numbers and URLs are normalized before scoring.</p>
          </div>
        </div>
        <div className="process-step">
          <span className="process-step__index">02</span>
          <div>
            <h3>A linear model</h3>
            <p>Logistic regression weighs roughly 230,000 sparse features — no black-box neural net.</p>
          </div>
        </div>
        <div className="process-step">
          <span className="process-step__index">03</span>
          <div>
            <h3>A precision-weighted cut-off</h3>
            <p>The threshold is tuned so a real email is less likely to be thrown away.</p>
          </div>
        </div>
      </section>
    </div>
  );
}

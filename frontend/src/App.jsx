import React, { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import {
  ArrowRightIcon,
  ChipIcon,
  FileIcon,
  HistoryIcon,
  LayersIcon,
  LogoMark,
  MoonIcon,
  ScanIcon,
  SunIcon,
} from "./components/Icons.jsx";
import { Spinner } from "./components/UI.jsx";
import { ToastProvider } from "./components/Toast.jsx";
import AnalyzePage from "./pages/AnalyzePage.jsx";
import BatchPage from "./pages/BatchPage.jsx";
import HistoryPage from "./pages/HistoryPage.jsx";
import ModelPage from "./pages/ModelPage.jsx";

const PAGES = [
  { id: "analyze", label: "Analyze", icon: ScanIcon, component: AnalyzePage },
  { id: "batch", label: "Batch", icon: LayersIcon, component: BatchPage },
  { id: "history", label: "History", icon: HistoryIcon, component: HistoryPage },
  { id: "model", label: "Model", icon: ChipIcon, component: ModelPage },
];

function readPage() {
  const hash = window.location.hash.replace("#", "").split("?")[0];
  return PAGES.some((page) => page.id === hash) ? hash : "analyze";
}

function useHashPage() {
  const [page, setPage] = useState(readPage);

  useEffect(() => {
    const onChange = () => setPage(readPage());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  const navigate = useCallback((next) => {
    if (window.location.hash === `#${next}`) {
      setPage(next);
    } else {
      window.location.hash = next;
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  return [page, navigate];
}

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      const requested = new URLSearchParams(window.location.search).get("theme");
      if (requested === "light" || requested === "dark") return requested;
      return localStorage.getItem("sift-theme") || "dark";
    } catch {
      return "dark";
    }
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", theme === "dark" ? "#172018" : "#F8F6EE");
    try {
      localStorage.setItem("sift-theme", theme);
    } catch {
      /* Storage may be blocked. */
    }
  }, [theme]);

  return [theme, setTheme];
}

function useHealth() {
  const [health, setHealth] = useState({ state: "checking" });
  const check = useCallback(async () => {
    try {
      const data = await api.health();
      setHealth({
        state: "online",
        version: data.model_version,
        threshold: data.threshold,
        responseMs: data.response_ms,
      });
    } catch {
      setHealth({ state: "offline" });
    }
  }, []);

  useEffect(() => {
    check();
    const timer = window.setInterval(check, 30_000);
    return () => window.clearInterval(timer);
  }, [check]);

  return [health, check];
}

function Sidebar({ page, navigate, health, theme, setTheme }) {
  return (
    <aside className="sidebar">
      <a className="brand" href="#analyze" onClick={() => navigate("analyze")} aria-label="Sift home">
        <LogoMark />
        <span className="brand__text">
          <span className="brand__name">Sift</span>
          <span className="brand__tag">Spam triage</span>
        </span>
      </a>

      <nav className="nav" aria-label="Main">
        <p className="nav__label">Workspace</p>
        {PAGES.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              type="button"
              className={`nav__item${page === item.id ? " is-active" : ""}`}
              onClick={() => navigate(item.id)}
              aria-current={page === item.id ? "page" : undefined}
            >
              <Icon size={19} />
              <span>{item.label}</span>
              <span className="nav__arrow">
                <ArrowRightIcon size={15} />
              </span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar__foot">
        <div className={`api-card api-card--${health.state}`}>
          <div className="api-card__row">
            <span className="status-dot" />
            <span className="api-card__title">
              {health.state === "online" ? "API online" : health.state === "offline" ? "API offline" : "Checking API"}
            </span>
            {health.state === "checking" ? <Spinner size={13} /> : null}
          </div>
          <p className="api-card__meta">
            {health.state === "online"
              ? `v${health.version || "2.1.0"} · cut-off ${(Number(health.threshold || 0.5) * 100).toFixed(0)}%`
              : health.state === "offline"
                ? "Start Django on port 8000."
                : "Pinging the classifier…"}
          </p>
        </div>

        <div className="theme-toggle" role="group" aria-label="Colour theme">
          <button
            type="button"
            className={theme === "light" ? "is-active" : ""}
            onClick={() => setTheme("light")}
            aria-pressed={theme === "light"}
          >
            <SunIcon size={16} /> Light
          </button>
          <button
            type="button"
            className={theme === "dark" ? "is-active" : ""}
            onClick={() => setTheme("dark")}
            aria-pressed={theme === "dark"}
          >
            <MoonIcon size={16} /> Dark
          </button>
        </div>

        <a className="sidebar__link" href="/api/docs/" target="_blank" rel="noreferrer">
          <FileIcon size={15} />
          API reference
          <ArrowRightIcon size={14} />
        </a>
      </div>
    </aside>
  );
}

function MobileBar({ page, navigate, health, theme, setTheme }) {
  return (
    <>
      <header className="mobile-bar">
        <a className="brand brand--compact" href="#analyze" onClick={() => navigate("analyze")}>
          <LogoMark size={30} />
          <span className="brand__name">Sift</span>
        </a>
        <div className="mobile-bar__right">
          <span className={`status-pill status-pill--${health.state}`}>
            <span className="status-dot" />
            {health.state === "online" ? "Live" : health.state === "offline" ? "Offline" : "…"}
          </span>
          <button
            type="button"
            className="icon-button"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? <SunIcon size={18} /> : <MoonIcon size={18} />}
          </button>
        </div>
      </header>

      <nav className="mobile-nav" aria-label="Main">
        {PAGES.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              type="button"
              className={page === item.id ? "is-active" : ""}
              onClick={() => navigate(item.id)}
              aria-current={page === item.id ? "page" : undefined}
            >
              <Icon size={20} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>
    </>
  );
}

export default function App() {
  const [page, navigate] = useHashPage();
  const [theme, setTheme] = useTheme();
  const [health, refreshHealth] = useHealth();

  const active = PAGES.find((item) => item.id === page) || PAGES[0];
  const ActivePage = active.component;

  return (
    <ToastProvider>
      <div className="app-shell">
        <Sidebar
          page={page}
          navigate={navigate}
          health={health}
          theme={theme}
          setTheme={setTheme}
        />
        <MobileBar
          page={page}
          navigate={navigate}
          health={health}
          theme={theme}
          setTheme={setTheme}
        />

        <main className="main">
          <div className="main__inner">
            {health.state === "offline" ? (
              <div className="offline-banner" role="alert">
                <span className="status-dot" />
                <p>
                  <strong>The API is offline.</strong> Start the Django backend to classify messages.
                </p>
                <button type="button" className="ghost-button" onClick={refreshHealth}>
                  Retry
                </button>
              </div>
            ) : null}

            <ActivePage navigate={navigate} health={health} refreshHealth={refreshHealth} />

            <footer className="footer">
              <span>Sift · classical ML spam triage</span>
              <span className="footer__meta">Word + character TF-IDF · Django REST · React</span>
            </footer>
          </div>
        </main>
      </div>
    </ToastProvider>
  );
}

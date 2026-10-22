import React from "react";

export function PageHeader({ eyebrow, title, description, actions }) {
  return (
    <header className="page-header">
      <div className="page-header__copy">
        {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
        <h1>{title}</h1>
        {description ? <p className="page-header__description">{description}</p> : null}
      </div>
      {actions ? <div className="page-header__actions">{actions}</div> : null}
    </header>
  );
}

export function Spinner({ size = 18, label = "Loading" }) {
  return (
    <span className="spinner" role="status" aria-label={label} style={{ width: size, height: size }} />
  );
}

export function EmptyState({ icon, title, description, children, compact = false }) {
  return (
    <div className={`empty-state${compact ? " empty-state--compact" : ""}`}>
      {icon ? <div className="empty-state__icon">{icon}</div> : null}
      <h3>{title}</h3>
      {description ? <p>{description}</p> : null}
      {children}
    </div>
  );
}

export function LabelPill({ label, size = "md" }) {
  return (
    <span className={`label-pill label-pill--${label} label-pill--${size}`}>
      <span className="label-pill__dot" />
      {label === "spam" ? "Spam" : "Ham"}
    </span>
  );
}

export function StatCard({ label, value, hint, tone = "neutral", icon }) {
  return (
    <div className={`stat-card stat-card--${tone}`}>
      <div className="stat-card__top">
        <span className="stat-card__label">{label}</span>
        {icon ? <span className="stat-card__icon">{icon}</span> : null}
      </div>
      <span className="stat-card__value">{value}</span>
      {hint ? <span className="stat-card__hint">{hint}</span> : null}
    </div>
  );
}

import React from "react";
import { percent } from "../lib/format.js";

export function ProbabilityGauge({ probability, threshold, label }) {
  const radius = 82;
  const circumference = 2 * Math.PI * radius;
  const safeProbability = Math.min(1, Math.max(0, Number(probability) || 0));
  const dash = circumference * safeProbability;

  const markerAngle = ((Number(threshold) || 0.5) * 360 - 90) * (Math.PI / 180);
  const inner = radius - 12;
  const outer = radius + 10;
  const marker = {
    x1: 100 + Math.cos(markerAngle) * inner,
    y1: 100 + Math.sin(markerAngle) * inner,
    x2: 100 + Math.cos(markerAngle) * outer,
    y2: 100 + Math.sin(markerAngle) * outer,
  };

  return (
    <div className={`gauge gauge--${label}`}>
      <svg viewBox="0 0 200 200" className="gauge__svg" role="img" aria-label={`Spam probability ${percent(probability)}`}>
        <circle className="gauge__track" cx="100" cy="100" r={radius} />
        <circle
          className="gauge__fill"
          cx="100"
          cy="100"
          r={radius}
          strokeDasharray={`${dash} ${circumference}`}
          transform="rotate(-90 100 100)"
        />
        <line className="gauge__marker" {...marker} />
      </svg>
      <div className="gauge__center">
        <span className="gauge__number">{percent(probability, 1)}</span>
        <span className="gauge__caption">spam probability</span>
      </div>
    </div>
  );
}

export function ProbabilityBar({ probability, threshold, label, compact = false }) {
  const value = Math.min(1, Math.max(0, Number(probability) || 0));
  const cut = Math.min(1, Math.max(0, Number(threshold) || 0.5));
  return (
    <div className={`prob-bar prob-bar--${label}${compact ? " prob-bar--compact" : ""}`}>
      <div
        className="prob-bar__track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(value * 100)}
        aria-label="Spam probability"
      >
        <span className="prob-bar__fill" style={{ width: `${value * 100}%` }} />
        <span
          className="prob-bar__cut"
          style={{ left: `${cut * 100}%` }}
          title={`Decision threshold ${percent(cut)}`}
        />
      </div>
      {!compact ? (
        <span className="prob-bar__value">{percent(value)}</span>
      ) : null}
    </div>
  );
}

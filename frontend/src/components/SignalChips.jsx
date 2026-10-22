import React from "react";
import { fixed } from "../lib/format.js";

export function SignalChips({ words, emptyLabel = "No words crossed the explanation cut-off." }) {
  if (!words?.length) {
    return <p className="signals__empty">{emptyLabel}</p>;
  }

  const peak = Math.max(...words.map((word) => Math.abs(Number(word.weight) || 0)), 0.0001);

  return (
    <ul className="signals">
      {words.map((word, index) => {
        const positive = Number(word.weight) >= 0;
        const raw = String(word.word);
        const isChar = raw.startsWith("~");
        const isStat = raw.startsWith("stat:");
        const label = isStat
          ? raw.slice(5).replaceAll("_", " ")
          : isChar
            ? raw.slice(1).replaceAll(" ", "␣") || "(space)"
            : raw;
        return (
          <li key={`${word.word}-${index}`} className={`signal-chip ${positive ? "signal-chip--spam" : "signal-chip--ham"}`}>
            <div className="signal-chip__head">
              <span className="signal-chip__word" title={label}>
                {isChar ? <span className="signal-chip__kind">char</span> : null}
                {isStat ? <span className="signal-chip__kind signal-chip__kind--stat">stat</span> : null}
                {label}
              </span>
              <span className="signal-chip__weight">
                {positive ? "+" : ""}
                {fixed(word.weight, 3)}
              </span>
            </div>
            <span
              className="signal-chip__bar"
              style={{ width: `${(Math.abs(Number(word.weight) || 0) / peak) * 100}%` }}
            />
          </li>
        );
      })}
    </ul>
  );
}

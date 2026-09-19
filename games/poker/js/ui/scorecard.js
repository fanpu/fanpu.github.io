import { CATEGORY_LESSON } from "../core/index.js";
import { esc, fmt, STREETS } from "./html.js";

// The scorecard: how often the coach agreed with you, by kind of decision, worst first.
// scorecardRows is pure, so it is tested in node.
export function scorecardRows(stats) {
  return Object.entries(stats.byCat)
    .map(([category, b]) => ({
      category,
      n: b.n,
      accuracy: b.n ? b.score / b.n : 0,
      mistakes: b.mistakes,
      lessonId: CATEGORY_LESSON[category] || "equity",
    }))
    .sort((x, y) => x.accuracy - y.accuracy || y.n - x.n);
}
export const agreement = (stats) => (stats.decisions ? (stats.correct + 0.5 * stats.acceptable) / stats.decisions : null);

export function scorecardHTML(stats, title) {
  if (!stats.decisions) return "";
  const rows = scorecardRows(stats);
  const recent = stats.recent.slice(0, 12);
  return `<details class="card"><summary><b>${title}</b><span>${Math.round(agreement(stats) * 100)}% with the coach · ${
    stats.decisions
  } decisions · ${stats.net >= 0 ? "+" : "−"}${fmt(Math.abs(stats.net))} in ${stats.hands} hands</span></summary>
    <div class="rows cats"><div class="row head"><span>decision</span><span>spots</span><span>with the coach</span></div>
    ${rows
      .map(
        (r) =>
          `<div class="row"><span>${esc(r.category)}</span><span>${r.n}</span><span class="acc"><i class="meter"><i style="width:${Math.round(
            r.accuracy * 100
          )}%"></i></i>${Math.round(r.accuracy * 100)}%</span></div>`
      )
      .join("")}</div>
    ${
      recent.length
        ? `<h3>Latest decisions</h3><ul class="recent">${recent
            .map(
              (d) =>
                `<li class="${d.grade}"><i></i><span>hand ${d.handNo}, ${STREETS[d.street]} · ${esc(d.key || "")} ${esc(
                  d.pos || ""
                )}</span><span>${esc(d.category)}: you ${d.action}${d.grade === "correct" ? "" : `, coach ${d.recommended}`}</span></li>`
            )
            .join("")}</ul>`
        : ""
    }</details>`;
}

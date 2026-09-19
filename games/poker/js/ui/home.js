import { esc, fmt } from "./html.js";

// The way in: three steps that share one table.
export function createHome(root, { store, onTable, onLesson }) {
  function render() {
    root.hidden = store.progress.mode !== "home";
    if (root.hidden) return;
    const done = Object.keys(store.progress.done).length;
    const score = (level) => {
      const s = store.stats[level];
      return s.decisions
        ? `${Math.round((100 * (s.correct + 0.5 * s.acceptable)) / s.decisions)}% with the coach over ${s.decisions} decisions · ${
            s.net >= 0 ? "+" : "−"
          }${fmt(Math.abs(s.net))} in ${s.hands} hands`
        : "Not played yet";
    };
    const weak = store.weakest("silent") || store.weakest("guided");
    root.innerHTML = `<div class="sheet glass">
      <h1>Poker Trainer</h1>
      <p class="lead">No-limit hold’em, from what beats what to sound decisions. Three steps, one table.</p>
      <ol class="steps">
        <li><button data-go="learn" disabled><b>Learn</b><span>Thirteen short lessons, each drilled at the table until it sticks.</span><small>${
          done ? done + " of 13 lessons done · " : ""
        }arrives with the lessons milestone</small></button></li>
        <li><button data-go="guided"><b>Train</b><span>Play real hands. The coach shows its working: ranges, equity, the value of each action, and why.</span><small>${score(
          "guided"
        )}</small></button></li>
        <li><button data-go="silent"><b>Prove it</b><span>Same table, silent coach. Every decision is graded and shown when the hand is over. This is the honest score.</span><small>${score(
          "silent"
        )}</small></button></li>
      </ol>
      ${
        weak
          ? `<p class="leak">You are leaking most on <b>${esc(weak.category.toLowerCase())}</b>: ${Math.round(weak.accuracy * 100)}% over ${
              weak.n
            } spots.</p>`
          : ""
      }
    </div>`;
  }
  root.addEventListener("click", (e) => {
    const t = e.target.closest("[data-go]");
    if (!t || t.disabled) return;
    if (t.dataset.go === "learn") onLesson?.();
    else onTable(t.dataset.go);
  });
  return { render };
}

import { fmt, renderer } from "./html.js";

// The header: where you are, which coach level, how the session is going, and the way to settings and home.
export function createHud(root, { store, onHome, onLevel, onSettings }) {
  const draw = renderer(root, (el) => {
    const v = store.view,
      st = store.stats[v.level || "guided"];
    const graded = st.decisions ? Math.round((100 * (st.correct + 0.5 * st.acceptable)) / st.decisions) + "%" : "–";
    el.innerHTML = `<button class="home" data-home aria-label="Home">‹ <b>Poker Trainer</b></button>
      <div class="level" role="group" aria-label="Coach level">
        <button data-level="guided" aria-pressed="${v.level === "guided"}" title="The coach shows its working as you play">Train</button>
        <button data-level="silent" aria-pressed="${v.level === "silent"}" title="The coach stays silent until the hand is over">Prove it</button>
      </div>
      <div class="meta"><span>hand <b>${v.handNo || 0}</b></span><span class="${(v.net || 0) > 0 ? "pos" : (v.net || 0) < 0 ? "neg" : ""}"><b>${
        (v.net || 0) > 0 ? "+" : (v.net || 0) < 0 ? "−" : ""
      }${fmt(
        Math.abs(v.net || 0)
      )}</b></span><span title="Share of decisions the coach agreed with, all time at this level">score <b>${graded}</b></span></div>
      <button class="gear" data-settings aria-label="Settings" aria-haspopup="dialog">⚙</button>`;
  });
  root.addEventListener("click", (e) => {
    const t = e.target.closest("[data-home],[data-level],[data-settings]");
    if (!t) return;
    if (t.dataset.level) onLevel(t.dataset.level);
    else if ("home" in t.dataset) onHome();
    else onSettings(t);
  });
  return {
    render() {
      const v = store.view,
        st = store.stats[v.level || "guided"];
      draw([v.level, v.handNo, v.net, st.decisions, st.correct, st.acceptable].join("|"));
    },
  };
}

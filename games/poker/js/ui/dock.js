import { handClass, handKey, evOfRaise } from "../core/index.js";
import { esc, fmt, evFmt, STREETS, actionWord, cardsAttr, mountCards, renderer } from "./html.js";
import { INFO_TABS } from "./info.js";

// The dock: what you hold, what you can do, and what just happened. It keeps the same height through every
// phase of a hand, so the table above it never jumps.
const GRADE_WORD = { correct: "Correct", acceptable: "Acceptable", mistake: "Mistake" };
const PRESETS = [
  ["min", "Min"],
  ["0.5", "½ pot"],
  ["0.66", "⅔ pot"],
  ["1", "Pot"],
  ["max", "All-in"],
];

export function createDock(root, { store, session, openInfo }) {
  const presetValue = (key) => {
    const v = store.view,
      l = v.legal;
    if (key === "coach") return Math.max(l.minTo, Math.min(l.maxTo, v.analysis?.rec.size || l.minTo));
    return key === "min" ? l.minTo : key === "max" ? l.maxTo : session.sizeFor(parseFloat(key));
  };

  // What you hold, in words. Naming a hand is not advice, so it shows at both coach levels.
  function describe(v) {
    if (!v.hero?.cards.length) return "";
    if (v.hero.folded) return "you folded";
    if (!v.board.length) return `${handKey(...v.hero.cards)} <small>${v.hero.pos}</small>`;
    return handClass(v.hero.cards, v.board).label;
  }

  function strip(v) {
    return `<div class="strip"><span class="cards mine" data-cards="${
      v.hero?.cards.length ? cardsAttr(v.hero.cards) : ""
    }"></span><span class="desc">${describe(v)}</span><span class="cards" data-cards="${cardsAttr(v.board || [])}"></span></div>`;
  }

  function actions(v) {
    const l = v.legal,
      a = v.level === "guided" ? v.analysis : null;
    const showPick = a && store.settings.showPick;
    const betting = l.toCall === 0 && v.street > 0;
    const pick = showPick ? (a.rec.action === "raise" ? "raise" : a.rec.action === "fold" ? "fold" : "call") : "";
    const ev = (text) => (a?.ev ? `<small>EV ${text}</small>` : "");
    const callWord = l.canCheck ? "Check" : "Call " + fmt(l.toCall) + (l.toCall >= v.hero.stack ? " <em>all-in</em>" : "");
    const coachSize = showPick && a.rec.action === "raise" && a.rec.size && l.canRaise;
    return `<div class="actions">
        <button data-act="fold" class="${pick === "fold" ? "pick" : ""}"><kbd>F</kbd>Fold${ev("0 bb")}</button>
        <button data-act="call" class="${pick === "call" ? "pick" : ""}"><kbd>C</kbd>${callWord}${ev(a?.ev ? evFmt(a.ev.passive) : "")}</button>
        <button data-act="raise" class="${pick === "raise" ? "pick" : ""}" ${
          l.canRaise ? "" : "disabled"
        }><kbd>R</kbd><span id="raiseLabel"></span></button>
      </div>
      ${
        l.canRaise
          ? `<div class="sizing"><div class="presets">${coachSize ? '<button data-size="coach" class="coach">Coach</button>' : ""}${PRESETS.map(
              ([k, label]) => `<button data-size="${k}">${label}</button>`
            ).join("")}</div>
             <div class="slide"><input type="range" id="raiseSlider" min="${l.minTo}" max="${l.maxTo}" step="1" aria-label="${
               betting ? "Bet" : "Raise"
             } size" /><output id="raiseInfo"></output></div></div>`
          : '<div class="sizing"></div>'
      }`;
  }

  function feedback(v) {
    const d = v.lastGrade,
      r = d.analysis.rec,
      betting = d.analysis.toCall === 0 && d.street > 0;
    let more = "";
    if (d.grade === "mistake")
      more = `The coach preferred to <b>${actionWord(
        r.action,
        betting
      )}</b>. Read why, then decide whether you disagree for a reason (a read, stack sizes) or whether the logic had not occurred to you. Only the second kind is a leak.`;
    else if (d.grade === "acceptable")
      more = `A reasonable alternative. The coach’s first choice was to <b>${actionWord(r.action, betting)}</b>; both lines have merit here.`;
    else more = "That is the coach’s line too.";
    if (d.action === "raise" && r.action === "raise" && r.size) {
      const ratio = d.raiseTo / r.size;
      if (ratio > 1.8)
        more += ` Sizing: ${fmt(d.raiseTo)} was much larger than needed (about ${fmt(
          r.size
        )}). Oversizing folds out the worse hands you want to keep in.`;
      else if (ratio < 0.55)
        more += ` Sizing: ${fmt(d.raiseTo)} was quite small (about ${fmt(
          r.size
        )}). Small bets give draws a cheap price and win less when you are ahead.`;
    }
    if (d.analysis.ev && d.evChosen != null) {
      const gap = d.analysis.ev.bestEV - d.evChosen;
      more += ` Model EV of your choice: <b>${evFmt(d.evChosen)}</b>${
        gap > 2 ? `, about ${evFmt(gap).replace("+", "")} less than the best line.` : "."
      }`;
    }
    const streak = store.stats.guided.streak;
    return `<div class="feedback ${d.grade}"><div><span class="grade">${GRADE_WORD[d.grade]}</span> <span class="ctx">${esc(d.category)}, ${
      STREETS[d.street]
    }${streak >= 3 ? ` · ${streak} correct in a row` : ""}</span><p>${more}</p></div>
      <div class="go"><button data-info="why" class="quiet">Why</button><button data-act="resume" class="primary"><kbd>Space</kbd>Continue</button></div></div>`;
  }

  function handOver(v) {
    const r = v.result,
      won = r.net > 0;
    const headline =
      r.net === 0
        ? "Even."
        : `You ${won ? "won" : "lost"} <b class="${won ? "pos" : "neg"}">${fmt(Math.abs(r.net))}</b>${r.heroFolded ? " (folded)" : ""}.`;
    const shown = r.shown
      .filter((h) => h.seat !== 0)
      .map((h) => `${esc(v.opponents.find((o) => o.seat === h.seat)?.name || "")} showed ${h.desc.toLowerCase()}`);
    const grades = (v.decisions || []).map((d) => `<i class="${d.grade}" title="${STREETS[d.street]}: ${d.action}"></i>`).join("");
    const lesson = (v.decisions || []).find((d) => d.grade === "mistake")?.lesson;
    return `<div class="feedback over"><div><span class="grade plain">${headline}</span> ${grades ? `<span class="pips">${grades}</span>` : ""}<p>${
      shown.length ? shown.join("; ") + ". " : ""
    }${
      lesson ? `Worth a look: <b>${esc(lesson.concept)}</b>. ${esc(lesson.text)}` : v.decisions?.length ? "" : "You had no decisions this hand."
    }</p></div>
      <div class="go">${
        v.decisions?.length ? '<button data-info="ranges" class="quiet">Review</button>' : ""
      }<button data-act="next" class="primary"><kbd>Space</kbd>Next hand</button></div></div>`;
  }

  const draw = renderer(root, (el) => {
    const v = store.view;
    let main;
    if (v.phase === "hero" && v.legal) main = actions(v);
    else if (v.phase === "feedback" && v.lastGrade) main = feedback(v);
    else if (v.phase === "handOver" && v.result) main = handOver(v);
    else
      main = `<div class="waiting">${
        v.phase === "dealing"
          ? "Dealing…"
          : v.hero?.folded
            ? "You are out of this hand. Watch how it plays out: what they show is how you learn their ranges."
            : "Waiting for the others…"
      }</div>`;
    el.innerHTML =
      strip(v) +
      `<div class="main">${main}</div><nav class="sheet-tabs" aria-label="Coach panels">${INFO_TABS.map(
        ([id, label]) => `<button data-info="${id}">${label}</button>`
      ).join("")}</nav>`;
    mountCards(el, store.settings.fourColour);
    syncRaise();
  });

  // The raise button, slider and its read-out follow raiseTo without redrawing the dock (so a drag is never interrupted).
  function syncRaise() {
    const v = store.view,
      l = v.legal;
    const label = root.querySelector("#raiseLabel");
    if (!label || !l) return;
    const betting = l.toCall === 0 && v.street > 0,
      a = v.level === "guided" ? v.analysis : null;
    const live = a?.ev && a.samples ? evOfRaise(a.ctx, a.samples, v.raiseTo) : null;
    label.innerHTML = l.canRaise
      ? `${betting ? "Bet" : "Raise to"} ${fmt(v.raiseTo)}${v.raiseTo >= l.maxTo ? " <em>all-in</em>" : ""}${
          live ? `<small>EV ${evFmt(live.ev)}</small>` : ""
        }`
      : betting
        ? "Bet"
        : "Raise";
    const slider = root.querySelector("#raiseSlider");
    if (slider && document.activeElement !== slider) slider.value = v.raiseTo;
    const info = root.querySelector("#raiseInfo");
    if (info)
      info.innerHTML = live
        ? `all fold ${Math.round(live.foldAll * 100)}%${live.eqCalled == null ? "" : ` · win ${Math.round(live.eqCalled * 100)}% if called`}`
        : "";
    let lit = false;
    for (const b of root.querySelectorAll("[data-size]")) {
      const on = !lit && presetValue(b.dataset.size) === v.raiseTo;
      if (on) lit = true;
      b.classList.toggle("on", on);
    }
  }

  function intent(act) {
    if (act === "resume") session.resume();
    else if (act === "next") session.next();
    else session.act(act);
  }
  root.addEventListener("click", (e) => {
    const t = e.target.closest("[data-act],[data-size],[data-info]");
    if (!t || t.disabled) return;
    if (t.dataset.act) intent(t.dataset.act);
    else if (t.dataset.size) session.setRaise(presetValue(t.dataset.size));
    else openInfo(t.dataset.info);
  });
  root.addEventListener("input", (e) => e.target.id === "raiseSlider" && session.setRaise(+e.target.value));

  // Keys: F fold, C check/call, R bet/raise, 1-5 size presets, Space or Enter to move on.
  addEventListener("keydown", (e) => {
    const tag = e.target.tagName;
    const typing = tag === "TEXTAREA" || tag === "SELECT" || (tag === "INPUT" && e.target.type !== "range");
    if (e.metaKey || e.ctrlKey || e.altKey || typing) return;
    if (store.progress.mode !== "table") return;
    const v = store.view,
      k = e.key.toLowerCase();
    if (k === " " || k === "enter") {
      if (e.target.closest?.("button")) return; // let a focused button handle its own activation
      e.preventDefault();
      if (v.phase === "feedback") session.resume();
      else if (v.phase === "handOver") session.next();
      else session.skip();
    } else if (v.phase === "hero") {
      if (k === "f") session.act("fold");
      else if (k === "c") session.act("call");
      else if (k === "r") session.act("raise");
      else if (/^[1-5]$/.test(k) && v.legal?.canRaise) session.setRaise(presetValue(PRESETS[+k - 1][0]));
    }
  });

  return {
    render() {
      const v = store.view;
      const redrawn = draw(
        [
          v.phase,
          v.handNo,
          v.street,
          v.level,
          !!v.analysis,
          v.legal?.toCall,
          v.legal?.minTo,
          v.legal?.maxTo,
          v.hero?.folded,
          v.board?.length,
          v.lastGrade?.index,
          v.result?.net,
          store.settings.showPick,
          store.settings.fourColour,
          store.stats.guided.streak,
        ].join("|")
      );
      if (!redrawn) syncRaise();
    },
  };
}

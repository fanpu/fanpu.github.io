import { PCT, RANKS, SUITS, openPct } from "../core/index.js";
import { esc, pct, fmt, evFmt, STREETS, actionWord, renderer } from "./html.js";
import { rangeGrid } from "./rangeGrid.js";

// The coach's working, in five tabs: whose range holds what, what each action is worth, what the next card
// does, what you hold, and why the coach would play it this way. When it is not the hero's turn the panel
// shows the reads on each opponent, which is information anybody at the table could gather.
const SHORT_BUCKETS = ["Straight+", "Set or trips", "Two pair", "Top pair+", "Weaker pair", "Draw", "Nothing"];
const BUCKET_COLORS = ["#8f2a1e", "#b8432f", "#d07a5e", "#d8b36a", "#e6d6a4", "#6fa3bd", "#8c8b82"];

function statRow(a, n) {
  const cell = (big, small, cls = "") => `<div><b class="${cls}">${big}</b><span>${small}</span></div>`;
  let h = "";
  if (a.street === 0) {
    const pre = a.rec.pre;
    const raiser = a.opps.find((o) => o.tag === "4bet") || a.opps.find((o) => o.tag === "3bet") || a.opps.find((o) => o.tag === "raise");
    h += cell("top " + Math.max(1, Math.round(a.pct)) + "%", "your hand, " + a.key); // aces are the top 0.45%: never show "top 0%"
    if (pre) {
      const need = Math.max(pre.call, pre.raise);
      h += cell("top " + Math.round(raiser ? raiser.pct : 20) + "%", (raiser ? esc(raiser.name) + "’s" : "their") + " raising range");
      h += cell("top " + need + "%", `range to continue; ${pre.raiseWord} top ${pre.raise}%`, a.pct <= need ? "pos" : "neg");
    } else if (a.category === "Open or fold") {
      const op = openPct(a.posn, n);
      h +=
        cell("top " + op + "%", "range to open from " + a.posn, a.pct <= op ? "pos" : "neg") +
        cell(pct(a.eq), `equity vs ${a.opps.length} random hands`);
    } else h += cell("top 14%", "raise with these, else check", a.pct <= 14 ? "pos" : "") + cell(pct(a.eq), "equity vs their ranges");
    return h + cell(a.posn, a.ip ? "in position" : "out of position");
  }
  h += cell(pct(a.eq), "equity vs " + (a.opps.length > 1 ? "their ranges" : "their range"));
  if (a.toCall > 0) {
    const edge = Math.round((a.eq - a.potOdds) * 100);
    h +=
      cell(pct(a.potOdds), "needed to call " + fmt(a.toCall)) +
      cell((edge > 0 ? "+" : edge < 0 ? "−" : "") + Math.abs(edge) + " pts", "equity minus price", edge >= 0 ? "pos" : "neg");
  } else {
    let ahead = 1;
    for (const o of a.opps) ahead *= o.now.total ? (o.now.ahead + 0.5 * o.now.tie) / o.now.total : 1;
    h += cell(pct(ahead), "ahead of everyone now") + cell(fmt(a.potNow), "pot, nothing to call");
  }
  return h + cell(a.spr >= 10 ? Math.round(a.spr) : a.spr.toFixed(1), "stack to pot, " + (a.ip ? "in position" : "out of position"));
}

function rangesTab(a, ui, n) {
  const narrowed = a.opps.filter((o) => o.pct < 100 || o.filters.length),
    wide = a.opps.filter((o) => !(o.pct < 100 || o.filters.length));
  const canOpen = a.category === "Open or fold";
  if (ui.forAnalysis !== a) {
    ui.forAnalysis = a;
    ui.opp = canOpen ? "open" : narrowed.length ? narrowed.slice().sort((x, y) => x.nCombos - y.nCombos)[0].id : a.opps[0]?.id ?? null;
  }
  const who =
    '<div class="who">' +
    (canOpen ? `<button data-opp="open" class="${ui.opp === "open" ? "on" : ""}">Your opening range</button>` : "") +
    [...narrowed, ...wide]
      .map(
        (o) =>
          `<button data-opp="${o.id}" class="${ui.opp === o.id ? "on" : ""}">${esc(o.name)} <small>${
            o.filters.length ? o.nCombos + " combos" : o.pct < 100 ? "top " + Math.round(o.pct) + "%" : "any two"
          }</small></button>`
      )
      .join("") +
    "</div>";
  if (ui.opp === "open") {
    const op = openPct(a.posn, n);
    return (
      who +
      rangeGrid((k) => (PCT[k] <= op ? 0.85 : 0), a.key) +
      `<p><b>Open from ${a.posn}.</b> Raise the top ${op}% when nobody has entered. Your ${a.key} is around the top ${Math.round(a.pct)}%, so it is ${
        a.pct <= op ? "inside" : "outside"
      } the range.</p><p class="note">Brass outline: your hand. Suited above the diagonal, offsuit below.</p>`
    );
  }
  const o = a.opps.find((x) => x.id === ui.opp) || a.opps[0];
  if (!o) return who;
  const pre =
    {
      raise: "raised pre",
      "3bet": "3-bet pre",
      "4bet": "4-bet pre",
      call: "called a raise",
      call3b: "called a 3-bet",
      limp: "limped",
      blind: "blind, no choice yet",
    }[o.tag] || "yet to act";
  const post = o.filters.map((f) => (f.mode === "big" ? "overbet " : f.mode === "aggr" ? "bet " : "called ") + STREETS[f.len - 2]).join(", ");
  const rd = o.read;
  const seen =
    rd && rd.n
      ? `${rd.n} hand${rd.n > 1 ? "s" : ""} seen: in ${Math.round(rd.rawV)}% of pots, raised ${Math.round(rd.rawR)}%.` +
        (o.style ? ` Reads as ${o.style}.` : " Too few to label.")
      : "No hands seen yet. Assumed average.";
  let h = who + rangeGrid((k, full) => (o.grid[k] || 0 ? Math.max(0.2, Math.min(1, o.grid[k] / full)) * 0.9 : 0), a.key);
  h += `<p><b>${o.pos}</b>, ${pre}${o.pct < 100 ? ` (top ${Math.round(o.pct)}%)` : ""}${post ? ", " + post : ""}. ${
    o.nCombos
  } combos.</p><p class="note">${seen}</p>`;
  if (o.comp)
    h +=
      '<div class="rows"><div class="row head"><span>they hold</span><span>share</span><span>your equity</span></div>' +
      o.comp.buckets
        .filter((b) => b.n > 0)
        .map(
          (b) =>
            `<div class="row"><span><i class="sw" style="background:${BUCKET_COLORS[b.k]}"></i>${SHORT_BUCKETS[b.k]}</span><span>${Math.round(
              (100 * b.n) / o.comp.total
            )}%</span><span class="${b.eq < 0.5 ? "neg" : ""}">${Math.round(b.eq * 100)}%</span></div>`
        )
        .join("") +
      "</div>";
  else h += '<p class="note">Darker: more combos of that hand remain. Brass outline: yours. Suited above the diagonal, offsuit below.</p>';
  return h;
}

function evTab(a) {
  const e = a.ev;
  if (!e) return '<p class="note">EV by action is worked out after the flop.</p>';
  const betting = a.toCall === 0;
  const rows = [];
  if (a.toCall > 0) rows.push({ label: "Fold", ev: 0 });
  rows.push({ label: a.toCall > 0 ? "Call " + fmt(a.toCall) : "Check", ev: e.passive });
  for (const r of e.rows)
    rows.push({ label: (betting ? "Bet " : "Raise to ") + fmt(r.to) + ` <small>${r.label}</small>`, ev: r.ev, fold: r.foldAll, eqc: r.eqCalled });
  const best = Math.max(...rows.map((r) => r.ev));
  let h = '<table class="evt"><tr><th>Action</th><th>EV</th><th>all fold</th><th>win if called</th></tr>';
  h += rows
    .map(
      (r) =>
        `<tr class="${r.ev === best ? "best" : ""}"><td>${r.label}</td><td class="${r.ev < -0.05 ? "neg" : ""}">${evFmt(r.ev)}</td><td>${
          r.fold === undefined ? "" : Math.round(r.fold * 100) + "%"
        }</td><td>${r.eqc == null ? "" : Math.round(r.eqc * 100) + "%"}</td></tr>`
    )
    .join("");
  h += "</table>";
  if (e.disagree)
    h += `<p class="note">${
      e.pureBluff
        ? `The model prices a ${
            betting ? "bet" : "raise"
          } above the coach’s pick even though you would be behind when called. That profit rests on its guesses about who folds and who calls, the shakiest numbers on this page, so it does not change your grade.`
        : `The EV model prefers to ${actionWord(e.best, betting)} while the coach’s rule of thumb says ${actionWord(
            a.rec.action,
            betting
          )}. Treat it as a close spot. Both are graded as fine.`
    }</p>`;
  return (
    h +
    '<p class="note">Big blinds, assuming no betting after this street. Betting EV = chance all fold × pot + chance called × (your win share × final pot, minus your bet). Opponents continue when their equity against the range your bet implies beats the price, which is how these bots really decide. Strong draws are undervalued; fold estimates are the shakiest input.</p>'
  );
}

function nextTab(a) {
  const cells = a.nextCards;
  if (!cells) return '<p class="note">On the flop and turn, this shows your equity after each card that could come next.</p>';
  const live = cells.filter((c) => c.eq !== null);
  const good = live.filter((c) => c.eq - a.eq >= 0.1).length,
    bad = live.filter((c) => a.eq - c.eq >= 0.1).length;
  let h = '<div class="ncards"><span></span>';
  for (let r = 14; r >= 2; r--) h += "<span>" + RANKS[r - 2] + "</span>";
  for (let s = 0; s < 4; s++) {
    h += `<span class="${s === 1 || s === 2 ? "red" : ""}">${SUITS[s]}</span>`;
    for (const c of cells.filter((x) => x.s === s)) {
      if (c.eq === null) {
        h += '<i class="gone">·</i>';
        continue;
      }
      const d = c.eq - a.eq,
        al = Math.min(0.85, Math.abs(d) / 0.35);
      h += `<i${Math.abs(d) < 0.04 ? "" : ` style="background:rgba(${d > 0 ? "63,185,132" : "226,96,79"},${al.toFixed(2)})"`}>${Math.round(
        c.eq * 100
      )}</i>`;
    }
  }
  return (
    h +
    `</div><p>You are at <b>${pct(a.eq)}</b> now. <b>${good}</b> of ${
      live.length
    } cards lift you by 10 points or more and <b>${bad}</b> drop you by 10 or more.${
      bad > good * 2 && a.eq >= 0.55 ? " Many bad cards with a good hand means bet now rather than later." : ""
    }</p><p class="note">Your equity after each possible ${
      a.street === 1 ? "turn" : "river"
    }. Green is better than now, red is worse. Each cell is a small simulation, so read it to within about 4 points.</p>`
  );
}

function handTab(a) {
  if (!a.hc)
    return `<p>You hold <b>${a.key}</b>, around the top ${Math.round(a.pct)}% of starting hands, in <b>${a.posn}</b> (${
      a.ip ? "in position" : "out of position"
    }).</p><p class="note">After the flop this tab names your hand, counts your outs and shows what beats you.</p>`;
  let h = `<p>You have <b>${a.hc.label}</b>.${a.hc.note ? " " + a.hc.note : ""}</p>`;
  if (a.oa) {
    if (a.oa.bigOuts > 0)
      h += `<p><b>${a.oa.bigOuts} outs</b> to a straight or flush, about ${Math.min(95, a.oa.bigOuts * (a.oa.cardsToCome === 2 ? 4 : 2))}% ${
        a.oa.cardsToCome === 2 ? "by the river" : "on the river"
      }.${a.oa.draws.length ? " " + a.oa.draws.join("; ") + "." : ""}</p>`;
    else if (a.oa.draws.length) h += "<p>" + a.oa.draws.join("; ").replace(/^./, (m) => m.toUpperCase()) + ".</p>";
  }
  const cb = a.combos;
  const by = Object.entries(cb.beatBy)
    .sort((x, y) => y[1] - x[1])
    .map(([k, v]) => v + " " + k.toLowerCase())
    .join(", ");
  h += cb.isNuts
    ? "<p><b>You have the nuts.</b> Nothing beats you right now.</p>"
    : `<p>The nuts is ${cb.nuts.toLowerCase()}. <b>${cb.beatTotal}</b> of ${cb.total} possible holdings beat you now${
        by ? " (" + by + ")" : ""
      }.</p>`;
  h +=
    '<div class="rows">' +
    a.opps
      .map((o) => {
        const ap = o.now.total ? Math.round((100 * (o.now.ahead + 0.5 * o.now.tie)) / o.now.total) : 0;
        return `<div class="row bar"><span>Ahead of ${esc(
          o.name
        )}’s range</span><span class="meter"><i style="width:${ap}%"></i></span><span>${ap}%</span></div>`;
      })
      .join("") +
    "</div>";
  return h + `<p>Board is <b>${a.tex.wetness}</b>. ${a.tex.who}</p><p class="note">${a.tex.advice}</p>`;
}

function whyTab(a) {
  const r = a.rec,
    betting = a.toCall === 0 && a.street > 0;
  let h = `<div class="verdict"><b>${actionWord(r.action, betting).toUpperCase()}${r.size && r.action === "raise" ? " to " + fmt(r.size) : ""}</b>${
    r.alts.length ? ` <span>also fine: ${r.alts.map((x) => actionWord(x, betting)).join(", ")}</span>` : ""
  }</div><ul>${r.reasons.map((x) => "<li>" + x + "</li>").join("")}</ul>`;
  if (a.toCall > 0 && a.street > 0 && a.opps.length === 1) {
    const mdf = Math.round(100 * (1 - a.toCall / a.potNow));
    h += `<p class="note">Minimum defence frequency at this bet size is <b>${mdf}%</b>. Against a balanced opponent you would continue with the best ${mdf}% of the hands you could hold here, or they could profit by bluffing with anything. These bots bluff less than a balanced player, so the coach goes by your equity against their range instead, which means folding more.</p>`;
  }
  return h;
}

function readsList(v) {
  if (!v.opponents?.length) return "";
  return (
    '<div class="rows reads"><div class="row head"><span>player</span><span>in pots</span><span>raises</span><span>reads as</span></div>' +
    v.opponents
      .map(
        (o) =>
          `<div class="row${o.folded ? " out" : ""}"><span><b>${esc(o.name)}</b> <small>${o.pos}</small></span><span>${
            o.read.n ? Math.round(o.read.rawV) + "%" : "–"
          }</span><span>${o.read.n ? Math.round(o.read.rawR) + "%" : "–"}</span><span>${
            o.label || (o.read.n ? o.read.n + " of 15 seen" : "new")
          }</span></div>`
      )
      .join("") +
    '</div><p class="note">How often each player has put chips in voluntarily and raised before the flop, from the hands you have watched. Fifteen hands earn a tentative label; forty make it firm.</p>'
  );
}

// The hand, replayed: your equity street by street, then every action in order. Choosing one rewinds the table to it.
function replayer(v) {
  const trail = v.equityTrail || [];
  let h = "";
  if (trail.length > 1) {
    const eqs = trail.map((t) => t.eq),
      swing = Math.max(...eqs) - Math.min(...eqs);
    h += `<div class="trail" role="img" aria-label="Your equity by street">${trail
      .map(
        (t) =>
          `<div><b>${Math.round(t.eq * 100)}%</b><i style="height:${Math.max(3, Math.round(t.eq * 46))}px"></i><span>${
            STREETS[t.street]
          }</span></div>`
      )
      .join("")}</div>`;
    h += `<p class="note">Your equity against the ranges still in, street by street.${
      swing > 0.35 ? ` A swing of ${Math.round(swing * 100)} points. Big swings are normal; what you control is the price you paid at each step.` : ""
    }</p>`;
  }
  const word = (t) => {
    const s = t.hero ? "" : "s"; // "You call", "Ava calls"
    const w =
      t.kind === "fold"
        ? "fold" + s
        : t.kind === "check"
          ? "check" + s
          : t.kind === "call"
            ? `call${s} ${fmt(t.added)}`
            : t.kind === "bet"
              ? `bet${s} ${fmt(t.to)}`
              : `raise${s} to ${fmt(t.to)}`;
    return w + (t.allIn ? " all-in" : "");
  };
  let street = -1;
  h += '<div class="timeline">';
  for (const t of v.timeline || []) {
    if (t.street !== street) h += `${street >= 0 ? "</div>" : ""}<div class="st"><h3>${STREETS[(street = t.street)]}</h3>`;
    h += `<button data-k="${t.k}" class="${t.hero ? "hero " + (t.grade || "") : ""}${v.at === t.k ? " on" : ""}">${t.hero ? "<i></i>" : ""}<b>${esc(
      t.name
    )}</b> ${word(t)}</button>`;
  }
  return h + `${street >= 0 ? "</div>" : ""}<div class="st"><button data-k="end" class="end${v.at == null ? " on" : ""}">Result</button></div></div>`;
}

const TABS = [
  ["ranges", "Ranges"],
  ["ev", "EV"],
  ["next", "Next card"],
  ["hand", "Hand"],
  ["why", "Why"],
];
export const INFO_TABS = TABS;

export function createInfo(root, { store, onClose, onRewind }) {
  const ui = { tab: "ranges", opp: null, forAnalysis: null };
  const draw = renderer(root, (el) => {
    const v = store.view,
      st = store.settings;
    // What is there to explain? The spot in front of the hero, or, once a hand is over, whichever moment of it is chosen.
    const over = v.phase === "handOver";
    const picked = over && v.at != null ? (v.decisions || []).find((d) => d.index === v.at) : null;
    const a = over ? picked?.analysis : v.level === "guided" ? v.analysis : null;
    const tabs = TABS.filter(([id]) => id !== "why" || st.showPick || over);
    if (!tabs.some(([id]) => id === ui.tab)) ui.tab = "ranges";

    let body;
    if (a) {
      const pane = { ranges: () => rangesTab(a, ui, v.n), ev: () => evTab(a), next: () => nextTab(a), hand: () => handTab(a), why: () => whyTab(a) }[
        ui.tab
      ]();
      body = `<div class="stats">${statRow(a, v.n)}</div><nav class="tabs" role="tablist">${tabs
        .map(([id, label]) => `<button role="tab" data-tab="${id}" aria-selected="${id === ui.tab}">${label}</button>`)
        .join("")}</nav><div class="pane" role="tabpanel">${pane}</div>`;
    } else if (v.level === "guided" && v.analysing) body = '<div class="thinking"><i></i>The coach is working it out…</div>' + readsList(v);
    else
      body =
        `<p class="lead">${
          v.level === "silent" && !over
            ? "The coach is silent. Play your own game; every decision is graded and shown when the hand is over."
            : over
              ? v.at == null
                ? "Choose any action above to put the table back to that moment. Your own decisions carry the coach’s full working."
                : "This was not your decision, so there is nothing to grade. The table shows the spot as it stood."
              : "The coach speaks when it is your turn. Meanwhile, watch the table:"
        }</p>` + readsList(v);

    const review = over && v.timeline?.length ? replayer(v) : "";
    const betting = picked && picked.analysis.toCall === 0 && picked.street > 0;
    const verdict = picked
      ? `<p class="graded ${picked.grade}"><span class="grade">${
          { correct: "Correct", acceptable: "Acceptable", mistake: "Mistake" }[picked.grade]
        }</span> You chose to <b>${actionWord(picked.action, betting)}${picked.raiseTo ? " " + fmt(picked.raiseTo) : ""}</b>${
          picked.grade === "correct" ? "." : `; the coach preferred to <b>${actionWord(picked.analysis.rec.action, betting)}</b>.`
        }</p>`
      : "";
    el.innerHTML = `<button class="close" data-close aria-label="Close">×</button><h2>${
      over ? "Replay" : v.level === "silent" ? "Table reads" : "Coach"
    }</h2>${review}${verdict}${body}`;
  });

  root.addEventListener("click", (e) => {
    const t = e.target.closest("[data-tab],[data-opp],[data-k],[data-close]");
    if (!t) return;
    if (t.dataset.k) return onRewind?.(t.dataset.k === "end" ? null : +t.dataset.k);
    if (t.dataset.tab) ui.tab = t.dataset.tab;
    else if (t.dataset.opp) ui.opp = t.dataset.opp === "open" ? "open" : +t.dataset.opp;
    else return onClose?.();
    api.render(true);
  });

  let bump = 0;
  const api = {
    get tab() {
      return ui.tab;
    },
    setTab(id) {
      ui.tab = id;
      api.render(true);
    },
    render(force) {
      const v = store.view;
      if (force) bump++;
      draw(
        [
          bump,
          v.phase,
          v.level,
          v.handNo,
          v.analysing,
          !!v.analysis,
          v.analysis?.category,
          v.street,
          v.decisions?.length,
          v.at,
          v.timeline?.length,
          v.opponents?.map((o) => o.read.n + o.label + o.folded).join(),
          store.settings.showPick,
        ].join("|")
      );
    },
  };
  return api;
}

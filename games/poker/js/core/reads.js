import { STREET_NAMES } from "./engine.js";

// What an observer can work out from the action: how often each player enters pots (VPIP) and raises
// preflop (PFR), and from that, the range of hands their line in this hand represents. Reads are built
// only from engine events, never from a bot's hidden settings, so they are exactly what a human at the
// table could know. The coach reasons from these ranges; the bots use them to judge each other.
export const READ_PRIOR = { vpip: 28, pfr: 18, weight: 8 };

const freshHand = () => ({ pct: 100, filters: [], tag: "", story: [], seen: false, vp: false, pf: false });
export function createReads(n) {
  const players = [];
  for (let i = 0; i < n; i++) players.push({ obs: { hands: 0, vpip: 0, pfr: 0 }, ...freshHand() });
  return { players };
}

// The estimate starts at a table-average prior and moves toward the observed share as hands accumulate.
export function readOf(reads, seat) {
  const o = reads.players[seat].obs,
    n = o.hands,
    w = READ_PRIOR.weight;
  const vpip = (100 * (o.vpip + (READ_PRIOR.vpip / 100) * w)) / (n + w);
  const pfr = Math.min(vpip, (100 * (o.pfr + (READ_PRIOR.pfr / 100) * w)) / (n + w));
  let label = "";
  if (n >= 15)
    label = vpip < 15.5 ? "Nit" : vpip >= 45 && pfr >= 29 ? "Maniac" : vpip >= 36 && pfr < 16 ? "Station" : vpip >= 21 && pfr >= 15 ? "LAG" : "TAG";
  return { n, vpip, pfr, rawV: n ? (100 * o.vpip) / n : 0, rawR: n ? (100 * o.pfr) / n : 0, label, sure: n >= 40 };
}
export function readLabel(reads, seat) {
  const r = readOf(reads, seat);
  return r.label ? r.label + (r.sure ? "" : "?") : "";
}
export const rangeOf = (reads, seat) => ({ pct: reads.players[seat].pct, filters: reads.players[seat].filters });

export function observe(reads, e) {
  if (e.type === "handStart") for (const p of reads.players) Object.assign(p, freshHand());
  else if (e.type === "post") reads.players[e.seat].tag = "blind";
  else if (e.type === "action") observeAction(reads, e);
}
export function observeAll(reads, events) {
  for (const e of events) observe(reads, e);
  return events;
}

function observeAction(reads, e) {
  const p = reads.players[e.seat];
  const st = e.seat === 0 ? null : readOf(reads, e.seat); // no stats are assumed about the hero: fixed defaults instead
  const aggressive = e.kind === "bet" || e.kind === "raise";
  const street = STREET_NAMES[e.street];
  if (e.street === 0) {
    const freeCheck = e.kind === "check";
    if (!p.seen && !freeCheck) {
      p.seen = true;
      p.obs.hands++;
    }
    if ((aggressive || e.kind === "call") && !p.vp) {
      p.vp = true;
      p.obs.vpip++;
    }
    if (aggressive && !p.pf) {
      p.pf = true;
      p.obs.pfr++;
    }
  }
  if (e.kind === "fold") p.story.push("folded " + street);
  else if (e.kind === "check") p.story.push(e.street === 0 ? "checked in the big blind" : "checked the " + street);
  else if (e.kind === "call") {
    if (e.street > 0) {
      p.filters.push({ len: e.boardLen, mode: "call" });
      p.story.push("called on the " + street);
    } else if (e.raisesBefore === 0) {
      Object.assign(p, { tag: "limp", pct: st ? st.vpip : 55 });
      p.story.push("limped in from " + e.pos);
    } else if (e.raisesBefore === 1) {
      Object.assign(p, { tag: "call", pct: Math.min(p.pct, st ? Math.max(8, st.vpip * 0.75) : 25) });
      p.story.push("called the raise from " + e.pos);
    } else {
      Object.assign(p, { tag: "call3b", pct: Math.min(p.pct, st ? Math.max(4, st.vpip * 0.35) : 9) });
      p.story.push("called a 3-bet");
    }
  } else if (e.street > 0) {
    const big = e.to - e.currentBetBefore > 1.5 * (e.potBefore + e.toCallBefore); // overbets read as a much stronger range
    p.filters.push({ len: e.boardLen, mode: big ? "big" : "aggr" });
    p.story.push((e.kind === "bet" ? "bet the " : "raised on the ") + street);
  } else if (e.raisesBefore === 0) {
    Object.assign(p, { tag: "raise", pct: st ? st.pfr : 20 });
    p.story.push("raised from " + e.pos);
  } else if (e.raisesBefore === 1) {
    Object.assign(p, { tag: "3bet", pct: st ? Math.max(3, st.pfr * 0.4) : 7 });
    p.story.push("3-bet from " + e.pos);
  } else {
    Object.assign(p, { tag: "4bet", pct: st ? Math.max(2, st.pfr * 0.15) : 3 });
    p.story.push("4-bet");
  }
}

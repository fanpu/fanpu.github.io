import {
  RANKS,
  SUITS,
  CAT_NAMES,
  newDeck,
  shuffle,
  cardKey,
  parseCards,
  eval7,
  evalCat,
  describeScore,
  PCT,
  handPct,
  handKey,
  openPct,
  flushDrawSuit,
  straightOuts,
  pickRunout,
  simulate,
  handClass,
  boardTexture,
  cbetPlan,
} from "../core/index.js";
import { rangeGrid } from "../ui/rangeGrid.js";

// The thirteen lessons and sixteen drills. The prose and the generators come from the original single-file
// trainer; what is new is that they are seedable (one rng for every random choice, so a drill can be tested
// and reproduced) and that every row of cards they draw is also remembered, so the lesson view can lay the
// same cards out on the 3D table.
let rng = Math.random;
export const seedLessons = (fn) => {
  rng = fn;
  lastTypeQ = -1; // the one piece of memory a drill keeps (so it does not ask the same question twice running)
};
const shuf = (a) => shuffle(a, rng);

// Rows of cards drawn since the last takeStaged(): [[{ label, cards, pick?: Set<cardKey> }]].
let staged = [];
export function takeStaged() {
  const s = staged;
  staged = [];
  return s;
}

// A card in the panel: an empty slot that the view fills with the shared card painting.
const CODE = (c) => RANKS[c.r - 2] + "shdc"[c.s];
function cardHTML(c, extra) {
  return '<span class="lcard' + (extra ? " " + extra : "") + '" data-cards="' + CODE(c) + '"></span>';
}
const legacyRangeGrid = (threshold, highlight) => rangeGrid((k) => (PCT[k] <= threshold ? 0.85 : 0), highlight);

const rint = (a, b) => a + Math.floor(rng() * (b - a + 1));
const pick = (arr) => arr[Math.floor(rng() * arr.length)];
const pc = (v) => Math.round(v * 100) + "%";
const pc1 = (v) => (v * 100).toFixed(1) + "%";
function frac(n, d) {
  return '<span class="fr"><span>' + n + "</span><span>" + d + "</span></span>";
}
function eqn(h) {
  return '<div class="eqn">' + h + "</div>";
}
function term(t) {
  return '<span class="term">' + t + "</span>";
}
function cardsRow(groups) {
  // groups: [{label, cards, pick:Set, dim:bool}]
  staged.push(groups);
  return groups
    .map(
      (gp) =>
        '<div class="tcards">' +
        (gp.label ? '<span class="lbl">' + gp.label + "</span>" : "") +
        gp.cards.map((c) => cardHTML(c, gp.pick ? (gp.pick.has(cardKey(c)) ? "plays" : "dim") : "")).join("") +
        "</div>"
    )
    .join("");
}
function bestFive(cards) {
  let best = -1,
    keep = null;
  const n = cards.length;
  for (let a = 0; a < n; a++)
    for (let b = a + 1; b < n; b++) {
      const five = cards.filter((_, i) => i !== a && i !== b);
      const s = eval7(five);
      if (s > best) {
        best = s;
        keep = five;
      }
    }
  return new Set(keep.map(cardKey));
}
function deckGrid(seen, outs) {
  let h = '<div class="nc"><div class="hd"></div>';
  for (let r = 14; r >= 2; r--) h += '<div class="hd">' + RANKS[r - 2] + "</div>";
  for (let s = 0; s < 4; s++) {
    h += '<div class="hd' + (s === 1 || s === 2 ? " red" : "") + '">' + SUITS[s] + "</div>";
    for (let r = 14; r >= 2; r--) {
      const k = cardKey({ r, s });
      if (seen.has(k)) h += '<div class="gone">\u00b7</div>';
      else if (outs.has(k)) h += '<div style="background:rgba(84,160,110,.85);font-weight:700">' + RANKS[r - 2] + "</div>";
      else h += '<div style="color:#b8ad92">' + RANKS[r - 2] + "</div>";
    }
  }
  return h + "</div>";
}
function equityHeadsUp(h1, h2, board, iters) {
  const dead = new Set([...h1, ...h2, ...board].map(cardKey));
  const deck = newDeck().filter((c) => !dead.has(cardKey(c)));
  let w = 0;
  for (let i = 0; i < iters; i++) {
    const full = board.concat(pickRunout(deck, 5 - board.length, [], rng));
    const a = eval7([...h1, ...full]),
      b = eval7([...h2, ...full]);
    w += a > b ? 1 : a === b ? 0.5 : 0;
  }
  return w / iters;
}
function spacedOptions(correct, candidates, minGap, lo, hi) {
  const out = [correct];
  for (const v of candidates) {
    if (out.length >= 4) break;
    if (v < lo || v > hi) continue;
    if (out.every((x) => Math.abs(x - v) >= minGap)) out.push(v);
  }
  let k = 1;
  while (out.length < 4 && k < 40) {
    for (const v of [correct + minGap * k, correct - minGap * k]) {
      if (out.length < 4 && v >= lo && v <= hi && out.every((x) => Math.abs(x - v) >= minGap)) out.push(v);
    }
    k++;
  }
  return out.sort((a, b) => a - b);
}

// ---------- Drill generators: each returns {q, visual, options, answer, explain} ----------
export const DRILLS = {};
DRILLS.nameHand = function () {
  const target = pick([0, 1, 1, 2, 2, 3, 4, 5, 6]);
  let c;
  for (let t = 0; t < 4000; t++) {
    c = shuf(newDeck()).slice(0, 7);
    if (evalCat(eval7(c)) === target) break;
  }
  const score = eval7(c),
    cat = evalCat(score);
  const near = [];
  for (let k = Math.max(0, cat - 3); k <= Math.min(8, cat + 3); k++) if (k !== cat) near.push(k);
  const opts = [cat, ...shuf(near).slice(0, 3)].sort((a, b) => a - b);
  const best = bestFive(c);
  const tips = [
    "No two cards match and there is no straight or flush, so the five highest cards play.",
    "Two cards of the same rank, plus the three highest other cards.",
    "Two different pairs, plus the highest remaining card. If three pairs are available, only the best two count, because a hand is exactly five cards.",
    "Three cards of one rank.",
    "Five ranks in a row. Suits do not matter for a straight.",
    "Five cards of the same suit. They do not need to be in a row.",
    "Three of one rank together with two of another.",
    "All four cards of one rank.",
    "Five cards in a row, all of the same suit.",
  ];
  return {
    q: "What is your best five-card hand?",
    visual: cardsRow([
      { label: "Your cards", cards: c.slice(0, 2) },
      { label: "Board", cards: c.slice(2) },
    ]),
    options: opts.map((k) => CAT_NAMES[k]),
    answer: opts.indexOf(cat),
    explain:
      "<b>" +
      describeScore(score) +
      ".</b> " +
      tips[cat] +
      " The five cards that play are outlined." +
      cardsRow([
        { label: "Your cards", cards: c.slice(0, 2), pick: best },
        { label: "Board", cards: c.slice(2), pick: best },
      ]),
  };
};
DRILLS.whoWins = function () {
  let c, sa, sb;
  for (let t = 0; t < 400; t++) {
    c = shuf(newDeck()).slice(0, 9);
    sa = eval7([...c.slice(0, 2), ...c.slice(4)]);
    sb = eval7([...c.slice(2, 4), ...c.slice(4)]);
    if (evalCat(sa) === evalCat(sb) || rng() < 0.25) break;
  }
  const ans = sa > sb ? 0 : sb > sa ? 1 : 2;
  const board = c.slice(4);
  let why = "Ana has <b>" + describeScore(sa).toLowerCase() + "</b>. Ben has <b>" + describeScore(sb).toLowerCase() + "</b>. ";
  if (ans === 2) why += "Both players end up with exactly the same best five cards, so they split the pot. Suits never break a tie.";
  else if (evalCat(sa) !== evalCat(sb)) why += "Different categories, so the higher category wins and nothing else matters.";
  else if (describeScore(sa) === describeScore(sb))
    why +=
      "Same category and same main ranks, so the side cards decide. The first side card that differs is called the kicker, and " +
      (ans === 0 ? "Ana" : "Ben") +
      " has the higher one.";
  else why += "Same category, so compare the ranks that make the hand. " + (ans === 0 ? "Ana" : "Ben") + " has the higher ones.";
  const ba = bestFive([...c.slice(0, 2), ...board]),
    bb = bestFive([...c.slice(2, 4), ...board]);
  return {
    q: "Who wins at showdown?",
    visual: cardsRow([
      { label: "Ana", cards: c.slice(0, 2) },
      { label: "Ben", cards: c.slice(2, 4) },
      { label: "Board", cards: board },
    ]),
    options: ["Ana", "Ben", "Split pot"],
    answer: ans,
    explain:
      why +
      '<div style="margin-top:8px">' +
      cardsRow([
        { label: "Ana plays", cards: [...c.slice(0, 2), ...board].filter((x) => ba.has(cardKey(x))) },
        { label: "Ben plays", cards: [...c.slice(2, 4), ...board].filter((x) => bb.has(cardKey(x))) },
      ]) +
      "</div>",
  };
};
DRILLS.position = function () {
  const post = ["SB", "BB", "UTG", "HJ", "CO", "BTN"],
    pre = ["UTG", "HJ", "CO", "BTN", "SB", "BB"];
  if (rng() < 0.3) {
    const first = rng() < 0.5;
    const ans = first ? "UTG" : "BB";
    const opts = shuf([ans, ...shuf(pre.filter((x) => x !== ans)).slice(0, 3)]);
    return {
      q: "Six players, cards just dealt, nobody has acted. Who acts " + (first ? "first" : "last") + " before the flop?",
      visual: "",
      options: opts,
      answer: opts.indexOf(ans),
      explain:
        "Before the flop the order is " +
        pre.join(", ") +
        '. The blinds have already been forced to put chips in, so they act last in this round only. UTG (short for "under the gun") sits just left of the big blind and must act first.',
    };
  }
  const n = rint(2, 4);
  const inHand = shuf(post.slice())
    .slice(0, n)
    .sort((a, b) => post.indexOf(a) - post.indexOf(b));
  const first = rng() < 0.5;
  const ans = first ? inHand[0] : inHand[n - 1];
  const opts = shuf(inHand.slice());
  const st = pick(["flop", "turn", "river"]);
  return {
    q: "On the " + st + ", these players are still in: " + shuf(inHand.slice()).join(", ") + ". Who acts " + (first ? "first" : "last") + "?",
    visual: "",
    options: opts,
    answer: opts.indexOf(ans),
    explain:
      "After the flop the order always starts left of the button: " +
      post.join(", ") +
      ". Skip anyone who has folded. Among these players that makes " +
      inHand[0] +
      " first and <b>" +
      inHand[n - 1] +
      "</b> last. Acting last is called being in position, and it is an advantage on every street.",
  };
};
DRILLS.outs = function () {
  const type = pick(["fd", "oesd", "gut", "fd+gut", "fd+oesd"]);
  const nb = rng() < 0.3 ? 4 : 3;
  let hero, board;
  for (let t = 0; t < 30000; t++) {
    const c = shuf(newDeck());
    hero = c.slice(0, 2);
    board = c.slice(2, 2 + nb);
    if (evalCat(eval7([...hero, ...board])) > 1) continue;
    const fs = flushDrawSuit(hero, board) >= 0,
      so = straightOuts(hero, board);
    const bs = [0, 0, 0, 0];
    for (const x of board) bs[x.s]++;
    if (Math.max(...bs) >= 3) continue;
    if (type === "fd" && fs && so === 0) break;
    if (type === "oesd" && !fs && so === 2) break;
    if (type === "gut" && !fs && so === 1) break;
    if (type === "fd+gut" && fs && so === 1) break;
    if (type === "fd+oesd" && fs && so === 2) break;
  }
  const all = [...hero, ...board];
  const seen = new Set(all.map(cardKey));
  const outs = new Set();
  for (const c of newDeck()) {
    if (seen.has(cardKey(c))) continue;
    if (evalCat(eval7([...all, c])) >= 4) outs.add(cardKey(c));
  }
  const n = outs.size;
  const opts = spacedOptions(n, shuf([4, 8, 9, 12, 15, 17, 13, 6, 3, 11, 7]), 1, 1, 21);
  const fs = flushDrawSuit(hero, board) >= 0,
    so = straightOuts(hero, board);
  let why = "<b>" + n + " outs.</b> ";
  if (fs) why += "Flush: a suit has 13 cards and you can see 4 of them, so 13 \u2212 4 = 9 are left. ";
  if (so) why += "Straight: " + so + " rank" + (so > 1 ? "s complete it, 4 cards each, so " + so * 4 : " completes it, so 4 cards") + ". ";
  if (fs && so)
    why +=
      "The two lists share " +
      (9 + so * 4 - n) +
      " card" +
      (9 + so * 4 - n === 1 ? "" : "s") +
      " (the ones that make both), and each card can only arrive once, so 9 + " +
      so * 4 +
      " \u2212 " +
      (9 + so * 4 - n) +
      " = " +
      n +
      ". ";
  why += "The outs are green below. Dots are the cards you can already see." + deckGrid(seen, outs);
  return {
    q: "How many unseen cards give you a straight or a flush on the next card?",
    visual: cardsRow([
      { label: "Your cards", cards: hero },
      { label: "Board", cards: board },
    ]),
    options: opts.map((v) => String(v)),
    answer: opts.indexOf(n),
    explain: why,
  };
};
function hitExact(n, toCome) {
  return toCome === 1 ? n / 46 : 1 - ((47 - n) * (46 - n)) / (47 * 46);
}
DRILLS.hitChance = function () {
  const n = pick([4, 6, 8, 9, 12, 15]),
    toCome = pick([1, 2]);
  const ex = Math.round(hitExact(n, toCome) * 100);
  const opts = spacedOptions(ex, [toCome === 2 ? n * 2 : n * 4, n, Math.round((n / 52) * 100), ex + 12, ex - 9, ex + 20], 4, 1, 95);
  const why =
    toCome === 1
      ? "On the turn you can see 6 cards, so 46 are unseen, and " +
        n +
        " of them help. " +
        eqn(frac(n, 46) + " = " + pc1(n / 46)) +
        "The shortcut is outs \u00d7 2 = " +
        n * 2 +
        "%, which is close. Each unseen card is about 2% of the deck, so each out is worth about 2%."
      : "It is easier to count missing. You miss the turn with " +
        (47 - n) +
        " of 47 cards, then miss the river with " +
        (46 - n) +
        " of the 46 left. " +
        eqn("1 \u2212 " + frac(47 - n, 47) + " \u00d7 " + frac(46 - n, 46) + " = " + pc1(hitExact(n, 2))) +
        "The shortcut is outs \u00d7 4 = " +
        n * 4 +
        "%. " +
        (n >= 12
          ? 'With this many outs the shortcut runs a little high, because it counts "hit on both cards" twice.'
          : "Close enough to use at the table.");
  return {
    q:
      "You have " +
      n +
      " outs " +
      (toCome === 1 ? "on the turn, with one card to come" : "on the flop, with two cards to come") +
      ". About how often do you hit?",
    visual: "",
    options: opts.map((v) => v + "%"),
    answer: opts.indexOf(ex),
    explain: why,
  };
};
function potSpot() {
  const P0 = pick([6, 8, 10, 12, 16, 20, 24, 30, 40]);
  const f = pick([1 / 3, 1 / 2, 2 / 3, 3 / 4, 1, 1.5]);
  const B = Math.max(1, Math.round(P0 * f));
  return { P0, B, f };
}
DRILLS.potOdds = function () {
  const { P0, B } = potSpot();
  const need = Math.round((100 * B) / (P0 + 2 * B));
  const opts = spacedOptions(
    need,
    [Math.round((100 * B) / (P0 + B)), Math.min(95, Math.round((100 * B) / P0)), Math.round((100 * P0) / (P0 + 2 * B)), need + 10, need - 10],
    4,
    2,
    95
  );
  return {
    q: "The pot is " + P0 + " bb. Your opponent bets " + B + " bb. What equity do you need for a call to break even?",
    visual: "",
    options: opts.map((v) => v + "%"),
    answer: opts.indexOf(need),
    explain:
      "After the bet the pot holds " +
      P0 +
      " + " +
      B +
      " = " +
      (P0 + B) +
      " bb, and calling costs " +
      B +
      " bb. " +
      eqn("<i>e</i> = " + frac("call", "pot + call") + " = " + frac(B, P0 + B + " + " + B) + " = " + frac(B, P0 + 2 * B) + " = " + need + "%") +
      "The most common slip is leaving your own call out of the bottom. You are paying " +
      B +
      " to win a final pot of " +
      (P0 + 2 * B) +
      ", and " +
      B +
      " of that is your own money coming back.",
  };
};
DRILLS.callOrFold = function () {
  const names = {
    4: "a gutshot straight draw",
    8: "an open-ended straight draw",
    9: "a flush draw",
    12: "a flush draw plus a gutshot",
    15: "a flush draw plus an open-ended straight draw",
  };
  let n, P0, B, hit, need;
  for (let t = 0; t < 200; t++) {
    n = pick([4, 8, 9, 12, 15]);
    ({ P0, B } = potSpot());
    hit = n / 46;
    need = B / (P0 + 2 * B);
    if (Math.abs(hit - need) > 0.035) break;
  }
  const call = hit > need;
  const ev = hit * (P0 + B) - (1 - hit) * B;
  return {
    q:
      "Turn. You hold " +
      names[n] +
      " (" +
      n +
      " outs). The pot is " +
      P0 +
      " bb and your opponent bets " +
      B +
      " bb. Assume you win when you hit, lose when you miss, and nobody bets on the river. Call or fold?",
    visual: "",
    options: ["Call", "Fold"],
    answer: call ? 0 : 1,
    explain:
      "Chance to hit: " +
      frac(n, 46) +
      " = " +
      pc1(hit) +
      ". Equity needed: " +
      frac(B, P0 + 2 * B) +
      " = " +
      pc1(need) +
      ". You have " +
      (call ? "more than you need, so calling makes money" : "less than you need, so calling loses money") +
      ". " +
      eqn(
        "EV = " +
          pc1(hit) +
          " \u00d7 " +
          (P0 + B) +
          " \u2212 " +
          pc1(1 - hit) +
          " \u00d7 " +
          B +
          " = " +
          (ev >= 0 ? "+" : "\u2212") +
          Math.abs(ev).toFixed(1) +
          " bb"
      ) +
      (call
        ? ""
        : "In real play you might still call a close one if you expect to win extra chips on the river when you hit. That extra is called implied odds."),
  };
};
DRILLS.equityGuess = function () {
  const R = () => rint(2, 14);
  let h1, h2, label, rule;
  const mk = (r1, r2, r3, r4) => {
    const su = shuf([0, 1, 2, 3]);
    h1 = [
      { r: r1, s: su[0] },
      { r: r2, s: su[1] },
    ];
    h2 = [
      { r: r3, s: su[2] },
      { r: r4, s: su[3] },
    ];
  };
  const type = pick(["pp", "overs", "dom", "hilo", "mixed"]);
  if (type === "pp") {
    const a = rint(4, 14),
      b = rint(2, a - 1);
    mk(a, a, b, b);
    label = "a higher pair against a lower pair";
    rule = "The lower pair almost always needs one of its two remaining cards, so it wins only about 1 time in 5.";
  } else if (type === "overs") {
    const p = rint(2, 11),
      x = rint(p + 2, 14),
      y = rint(p + 1, x - 1);
    mk(p, p, x, y);
    label = "a pair against two higher cards";
    rule =
      'This is the classic "coin flip". The pair is ahead now, but the two overcards have six cards to pair up plus straight and flush chances, so it lands near 55 to 45.';
  } else if (type === "dom") {
    const top = rint(10, 14),
      k1 = rint(4, top - 1),
      k2 = rint(2, k1 - 1);
    mk(top, k1, top, k2);
    label = "a dominated hand: both share a card, one has the better side card";
    rule =
      "When both pair the shared card, the better kicker wins. The weaker hand really has only three cards that help it, so it sits near 25 to 30%.";
  } else if (type === "hilo") {
    const a = rint(9, 14),
      b = rint(8, a - 1),
      c = rint(3, b - 1),
      d = rint(2, c - 1);
    mk(a, b, c, d);
    label = "two higher cards against two lower cards";
    rule =
      "Neither hand has a pair yet, and the lower cards win whenever they pair and the higher ones do not. The higher cards are usually a 60 to 65% favourite, less than most people expect.";
  } else {
    const p = rint(5, 12),
      x = rint(p + 1, 14),
      y = rint(2, p - 1);
    mk(p, p, x, y);
    label = "a pair against one higher and one lower card";
    rule = "Only the one overcard really helps, three cards instead of six, so the pair is about a 70 to 30 favourite.";
  }
  const swap = rng() < 0.5;
  if (swap) {
    const t = h1;
    h1 = h2;
    h2 = t;
  }
  const eq = equityHeadsUp(h1, h2, [], 5000);
  const ex = Math.round(eq * 20) * 5;
  const opts = spacedOptions(ex, shuf([ex + 15, ex - 15, ex + 30, ex - 30, ex + 45, ex - 45]), 15, 5, 95);
  return {
    q: "Both players are all-in before the flop. Roughly how often do you win?",
    visual: cardsRow([
      { label: "You", cards: h1 },
      { label: "Opponent", cards: h2 },
    ]),
    options: opts.map((v) => v + "%"),
    answer: opts.indexOf(ex),
    explain: "A simulation of 5,000 deals gives you <b>" + pc(eq) + "</b>. This is " + label + ". " + rule,
  };
};
DRILLS.combos = function () {
  const kind = pick(["any", "any", "pair", "suited"]);
  const x = rint(9, 14);
  let y = rint(8, 13);
  if (y >= x) y = x - 1;
  const X = RANKS[x - 2],
    Y = RANKS[y - 2];
  let known;
  for (let t = 0; t < 500; t++) {
    known = shuf(newDeck()).slice(0, pick([2, 5]));
    const has = known.filter((c) => c.r === x || c.r === y).length;
    if (has >= 1 && has <= 3) break;
  }
  const seen = new Set(known.map(cardKey));
  const left = (r) => [0, 1, 2, 3].filter((s) => !seen.has(cardKey({ r, s })));
  const lx = left(x),
    ly = left(y);
  let n, name, why;
  if (kind === "pair") {
    n = (lx.length * (lx.length - 1)) / 2;
    name = X + X;
    why =
      "You can see " +
      (4 - lx.length) +
      " of the four " +
      X +
      "s, so " +
      lx.length +
      " remain. A pair needs two of them, and the number of ways to choose 2 from " +
      lx.length +
      " is " +
      eqn(frac(lx.length + " \u00d7 " + (lx.length - 1), 2) + " = " + n) +
      "With none visible it would be 6.";
  } else if (kind === "suited") {
    n = lx.filter((s) => ly.includes(s)).length;
    name = X + Y + " suited";
    why =
      "A suited combo needs both cards in the same suit. Go suit by suit and check that neither card is already visible. That leaves <b>" +
      n +
      "</b> of the usual 4.";
  } else {
    n = lx.length * ly.length;
    name = X + Y + " (suited or not)";
    why =
      "Any remaining " +
      X +
      " can go with any remaining " +
      Y +
      ". You can see " +
      (4 - lx.length) +
      " of the " +
      X +
      "s and " +
      (4 - ly.length) +
      " of the " +
      Y +
      "s. " +
      eqn(lx.length + " \u00d7 " + ly.length + " = " + n) +
      "With none visible it would be 4 \u00d7 4 = 16.";
  }
  const opts = spacedOptions(n, shuf([16, 12, 9, 8, 6, 4, 3, 2, 1, 0]), 1, 0, 16);
  return {
    q: "How many combos of " + name + " can one opponent still hold?",
    visual: cardsRow([{ label: "Your cards", cards: known.slice(0, 2) }].concat(known.length > 2 ? [{ label: "Flop", cards: known.slice(2) }] : [])),
    options: opts.map(String),
    answer: opts.indexOf(n),
    explain: "<b>" + n + ".</b> " + why + " Cards you can see are called blockers, because they block hands your opponent would otherwise have.",
  };
};
DRILLS.openOrFold = function () {
  const seat = pick(["UTG", "HJ", "CO", "BTN", "SB"]);
  const thr = openPct(seat, 6);
  const want = pick(["raise", "fold"]);
  let c, p;
  for (let t = 0; t < 3000; t++) {
    c = shuf(newDeck()).slice(0, 2);
    p = handPct(c[0], c[1]);
    if (Math.abs(p - thr) < 4) continue;
    if (want === "raise" ? p <= thr : p > thr && p <= thr + 40) break;
  }
  const raise = p <= thr;
  const key = handKey(c[0], c[1]);
  return {
    q: "Six players. Everyone before you has folded and you are in the " + seat + " seat. Raise or fold?",
    visual: cardsRow([{ label: "Your cards", cards: c }]),
    options: ["Raise", "Fold"],
    answer: raise ? 0 : 1,
    explain:
      "<b>" +
      key +
      "</b> ranks around the top " +
      Math.round(p) +
      "% of starting hands. From " +
      seat +
      " at a six-player table the chart opens about the top " +
      thr +
      "%, so this is a <b>" +
      (raise ? "raise" : "fold") +
      "</b>. Green cells below are the opening range for this seat, and your hand is outlined." +
      rangeGrid((k) => (PCT[k] <= thr ? 0.85 : 0), key) +
      '<span class="gridnote">Upper right of the diagonal is suited, lower left is offsuit, the diagonal is pairs.</span>',
  };
};
function simpleClass(hero, board) {
  const cat = evalCat(eval7([...hero, ...board]));
  const br = [...new Set(board.map((c) => c.r))].sort((a, b) => b - a);
  const pocket = hero[0].r === hero[1].r;
  if (cat >= 4) return "Straight or better";
  if (cat === 3) return pocket ? "Set" : "Trips";
  if (cat === 2) return "Two pair";
  if (cat === 1) {
    if (pocket) return hero[0].r > br[0] ? "Overpair" : "Pocket pair below the top card";
    const pr = hero.find((c) => br.includes(c.r)).r;
    const i = br.indexOf(pr);
    return i === 0 ? "Top pair" : i === 1 ? "Second pair" : "Bottom pair";
  }
  return "No pair";
}
DRILLS.handClass = function () {
  const classes = ["Top pair", "Second pair", "Bottom pair", "Overpair", "Pocket pair below the top card", "Set", "Two pair", "No pair"];
  const target = pick(classes);
  let hero, board;
  for (let t = 0; t < 40000; t++) {
    const c = shuf(newDeck());
    hero = c.slice(0, 2);
    board = c.slice(2, 5);
    if (new Set(board.map((x) => x.r)).size < 3) continue;
    if (simpleClass(hero, board) === target) break;
  }
  const cls = simpleClass(hero, board);
  const opts = shuf([cls, ...shuf(classes.filter((x) => x !== cls)).slice(0, 3)]);
  const hc = handClass(hero, board);
  return {
    q: "What do you have on this flop?",
    visual: cardsRow([
      { label: "Your cards", cards: hero },
      { label: "Flop", cards: board },
    ]),
    options: opts,
    answer: opts.indexOf(cls),
    explain: "<b>" + cls + ".</b> In full: " + hc.label + ". " + (hc.note || "With no pair you are relying on a draw or on your opponent folding."),
  };
};
DRILLS.texture = function () {
  const want = pick(["dry", "medium", "wet"]);
  let b, t;
  for (let k = 0; k < 2000; k++) {
    b = shuf(newDeck()).slice(0, 3);
    t = boardTexture(b);
    if (t.wetness === want) break;
  }
  const opts = ["Dry", "Medium", "Wet"];
  return {
    q: "How wet is this flop? Wet means many straight and flush draws are possible.",
    visual: cardsRow([{ label: "Flop", cards: b }]),
    options: opts,
    answer: opts.indexOf(t.wetness[0].toUpperCase() + t.wetness.slice(1)),
    explain: t.text + " " + t.advice,
  };
};
DRILLS.betPrice = function () {
  const { P0, B } = potSpot();
  const caller = Math.round((100 * B) / (P0 + 2 * B)),
    bluff = Math.round((100 * B) / (P0 + B));
  if (rng() < 0.5) {
    const opts = spacedOptions(caller, [bluff, Math.min(95, Math.round((100 * B) / P0)), caller + 12, caller - 8], 4, 2, 95);
    return {
      q: "You bet " + B + " bb into a pot of " + P0 + " bb. What equity does your opponent need to call?",
      visual: "",
      options: opts.map((v) => v + "%"),
      answer: opts.indexOf(caller),
      explain:
        "They pay " +
        B +
        " to win a final pot of " +
        P0 +
        " + " +
        B +
        " + " +
        B +
        " = " +
        (P0 + 2 * B) +
        ". " +
        eqn(frac(B, P0 + 2 * B) + " = " + caller + "%") +
        "So a bet this size is a mistake for them to call with any draw that arrives less often than " +
        caller +
        "%.",
    };
  }
  const opts = spacedOptions(bluff, [caller, Math.min(95, Math.round((100 * B) / P0)), bluff + 12, bluff - 10], 4, 2, 95);
  return {
    q:
      "You bluff " +
      B +
      " bb into a pot of " +
      P0 +
      " bb with a hand that cannot win if called. How often must your opponent fold for the bluff to break even?",
    visual: "",
    options: opts.map((v) => v + "%"),
    answer: opts.indexOf(bluff),
    explain:
      "When they fold you win " +
      P0 +
      ". When they call you lose " +
      B +
      ". Setting the average to zero, <i>f</i> \u00d7 " +
      P0 +
      " \u2212 (1 \u2212 <i>f</i>) \u00d7 " +
      B +
      " = 0, gives " +
      eqn("<i>f</i> = " + frac("bet", "pot + bet") + " = " + frac(B, P0 + B) + " = " + bluff + "%") +
      "Bigger bluffs must work more often.",
  };
};
DRILLS.mdf = function () {
  const { P0, B } = potSpot();
  const mdf = Math.round((100 * P0) / (P0 + B)),
    alpha = Math.round((100 * B) / (P0 + B)),
    bluffs = Math.round((100 * B) / (P0 + 2 * B));
  if (rng() < 0.6) {
    const opts = spacedOptions(mdf, [alpha, bluffs, 100 - bluffs, mdf - 12, mdf + 10], 4, 2, 98);
    return {
      q:
        "The pot is " +
        P0 +
        " bb and your opponent bets " +
        B +
        " bb. If you fold too often, they profit by betting any two cards. What share of your range must continue so that a pure bluff does not make money?",
      visual: "",
      options: opts.map((v) => v + "%"),
      answer: opts.indexOf(mdf),
      explain:
        "A pure bluff risks " +
        B +
        " to win " +
        P0 +
        ", so it profits when you fold more than " +
        frac(B, P0 + " + " + B) +
        " = " +
        alpha +
        "% of the time. You must therefore continue with at least the rest. " +
        eqn("MDF = " + frac("pot", "pot + bet") + " = " + frac(P0, P0 + B) + " = " + mdf + "%") +
        "Notice the pot here is the pot <b>before</b> the bet. Smaller bets force you to defend more, bigger bets let you fold more.",
    };
  }
  const opts = spacedOptions(bluffs, [alpha, mdf, bluffs + 12, bluffs - 8], 4, 2, 98);
  return {
    q:
      "River. You bet " +
      B +
      " bb into " +
      P0 +
      " bb with a range of strong hands and pure bluffs. What share of your betting range should be bluffs, so that your opponent gains nothing by always calling or always folding?",
    visual: "",
    options: opts.map((v) => v + "%"),
    answer: opts.indexOf(bluffs),
    explain:
      "Your opponent pays " +
      B +
      " for a final pot of " +
      (P0 + 2 * B) +
      ", so a call needs to win " +
      frac(B, P0 + 2 * B) +
      " = " +
      bluffs +
      "% of the time. If exactly that share of your bets are bluffs, calling and folding earn them the same, and they cannot exploit you. " +
      eqn("bluff share = " + frac("bet", "pot + 2 \u00d7 bet") + " = " + bluffs + "%") +
      "It is the same number as the caller's pot odds. Bigger bets are allowed more bluffs.",
  };
};
DRILLS.cbet = function () {
  const want = pick(["small", "check", "big"]);
  let b, pl;
  for (let t = 0; t < 6000; t++) {
    b = shuf(newDeck()).slice(0, 3);
    pl = cbetPlan(b);
    if (!pl || pl.kind !== want) continue;
    const tx = boardTexture(b);
    if (want === "small" && !(tx.hi >= 13 && tx.connected === 0 && tx.maxSuit === 1)) continue;
    if (want === "check" && !(tx.hi <= 8 && tx.connected === 2)) continue;
    if (want === "big" && !(tx.hi >= 11 && tx.connected === 2 && tx.maxSuit === 2 && !tx.paired)) continue;
    break;
  }
  const opts = ["Bet small with most hands", "Bet big with strong hands and draws, check the rest", "Check most hands"];
  const idx = { small: 0, big: 1, check: 2 }[pl.kind];
  return {
    q: "You raised before the flop from the cutoff and only the big blind called. They check to you. What is the default plan on this flop?",
    visual: cardsRow([{ label: "Flop", cards: b }]),
    options: opts,
    answer: idx,
    explain: pl.why,
  };
};
const TYPE_QS = [
  {
    q: "A Station (calls far too much) checks to you on the river. You missed your draw and have nothing. What now?",
    o: ["Bluff big", "Check and give up"],
    a: 1,
    e: "A bluff only earns when the other player folds, and a Station does not fold. Against this type, save your chips when you miss and bet bigger when you hit.",
  },
  {
    q: "A Nit (plays very few hands, rarely bluffs) suddenly raises your flop bet. You hold top pair with a medium kicker.",
    o: ["Usually fold", "Re-raise all-in"],
    a: 0,
    e: "A Nit raises with strong hands only. Top pair is rarely ahead of that range, so folding loses the least. The same raise from a Maniac would mean far less.",
  },
  {
    q: "A Maniac (raises with almost anything) keeps betting into you. You hold a strong made hand.",
    o: ["Raise at once to end it", "Mostly call and let them keep betting"],
    a: 1,
    e: "Their betting range is full of weak hands that fold to a raise. Calling keeps those bluffs coming on later streets. Raise eventually, but there is no hurry.",
  },
  {
    q: "You have a strong but not unbeatable hand against a Station. How should you size your value bet?",
    o: ["Small, so they do not fold", "Large, because they call anyway"],
    a: 1,
    e: "A value bet should be as large as worse hands will still call. A Station calls big bets with weak pairs, so a small bet just leaves chips behind.",
  },
  {
    q: "What do VPIP 45 and PFR 9 tell you about a player?",
    o: ["Loose and passive: plays many hands, mostly by calling", "Tight and aggressive"],
    a: 0,
    e: "VPIP is the share of hands they put chips in with by choice. PFR is the share they raise before the flop. 45 against 9 means lots of hands entered by calling. That is the Station profile.",
  },
  {
    q: "A TAG (tight and aggressive) opens from UTG. Compared with the same player opening from the button, their range is",
    o: ["Much stronger", "About the same"],
    a: 0,
    e: "Good players open few hands from early seats, around the top 17% at a six-player table, and many from the button, around 43%. Same player, same action, very different range. Always read the seat along with the player.",
  },
];
let lastTypeQ = -1;
DRILLS.vsType = function () {
  let i;
  do {
    i = rint(0, TYPE_QS.length - 1);
  } while (i === lastTypeQ);
  lastTypeQ = i;
  const t = TYPE_QS[i];
  return { q: t.q, visual: "", options: t.o, answer: t.a, explain: t.e };
};

// ---------- Lessons ----------
function rankTable() {
  const rows = [
    ["Straight flush", "Ts 9s 8s 7s 6s", "five in a row, all one suit", "0.0015%"],
    ["Four of a kind", "9s 9h 9d 9c Kd", "all four of one rank", "0.024%"],
    ["Full house", "Qs Qh Qd 5c 5d", "three of one rank plus two of another", "0.14%"],
    ["Flush", "Ah Jh 8h 5h 2h", "five of one suit, any ranks", "0.20%"],
    ["Straight", "9s 8d 7d 6c 5h", "five ranks in a row, any suits", "0.39%"],
    ["Three of a kind", "7s 7h 7d Kc 2s", "three of one rank", "2.1%"],
    ["Two pair", "Js Jd 4h 4c As", "two different pairs", "4.8%"],
    ["One pair", "Ts Td Ac 8h 3d", "two of one rank", "42%"],
    ["High card", "As Jd 9h 6c 3s", "none of the above", "50%"],
  ];
  return (
    '<table class="ttable"><tr><th>Hand</th><th>Example</th><th>How often in 5 random cards</th></tr>' +
    rows
      .map(
        (r) =>
          "<tr><td><b>" +
          r[0] +
          '</b><br><span style="color:var(--ink-soft);font-size:12px">' +
          r[2] +
          '</span></td><td style="white-space:nowrap">' +
          parseCards(r[1])
            .map((c) => cardHTML(c))
            .join("") +
          "</td><td>" +
          r[3] +
          "</td></tr>"
      )
      .join("") +
    "</table>"
  );
}
function hitTable() {
  return (
    '<table class="ttable"><tr><th>Draw</th><th>Outs</th><th>One card (exact)</th><th>\u00d72</th><th>Two cards (exact)</th><th>\u00d74</th></tr>' +
    [
      ["Gutshot", 4],
      ["Open-ended straight draw", 8],
      ["Flush draw", 9],
      ["Flush draw + gutshot", 12],
      ["Flush draw + open-ended", 15],
    ]
      .map(
        ([nm, n]) =>
          "<tr><td>" +
          nm +
          "</td><td>" +
          n +
          "</td><td>" +
          pc1(n / 46) +
          "</td><td>" +
          n * 2 +
          "%</td><td>" +
          pc1(hitExact(n, 2)) +
          "</td><td>" +
          n * 4 +
          "%</td></tr>"
      )
      .join("") +
    "</table>"
  );
}
function priceTable(bluff) {
  return (
    '<table class="ttable"><tr><th>Bet size</th><th>Caller needs</th>' +
    (bluff ? "<th>Pure bluff must work</th>" : "") +
    "</tr>" +
    [
      ["A third of the pot", 1 / 3],
      ["Half the pot", 1 / 2],
      ["Two-thirds of the pot", 2 / 3],
      ["The whole pot", 1],
      ["Twice the pot", 2],
    ]
      .map(([nm, f]) => "<tr><td>" + nm + "</td><td>" + pc(f / (1 + 2 * f)) + "</td>" + (bluff ? "<td>" + pc(f / (1 + f)) + "</td>" : "") + "</tr>")
      .join("") +
    "</table>"
  );
}
const MATCHUPS = [
  ["Qs Qh", "7d 7c", "Higher pair against lower pair"],
  ["8s 8h", "Ad Kc", "Pair against two higher cards"],
  ["Ts Th", "Ad 6c", "Pair against one higher, one lower"],
  ["As Kh", "Ad Qc", "Dominated: same top card, worse side card"],
  ["As Kh", "8d 7d", "Two high cards against two lower suited cards"],
  ["As Ah", "Kd Kc", "Aces against kings"],
];

export const LESSONS = [
  {
    id: "cards",
    title: "The cards and the goal",
    blurb: "What is in the deck, what you are dealt, and the two ways to win a pot.",
    pages: [
      {
        t: "A deck of 52 cards",
        h: () =>
          "<p>Poker uses an ordinary deck. Every card has a " +
          term("rank") +
          " and a " +
          term("suit") +
          '.</p><p>There are 13 ranks. From lowest to highest they are 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, Ace. Poker shorthand gives every rank a single character, so ten is written T: 2 3 4 5 6 7 8 9 T J Q K A. That is why a hand is called "ATs" or "T9o". The cards themselves still show 10.</p><p>There are 4 suits: spades \u2660, hearts \u2665, diamonds \u2666 and clubs \u2663. No suit is stronger than another. Suits only matter for one kind of hand, the flush, which you will meet in the next lesson.</p>' +
          cardsRow([{ cards: parseCards("As Kh Qd Jc Ts") }]) +
          "<p>13 ranks times 4 suits gives 52 cards, and every card appears exactly once. That single fact is what makes the maths of poker possible: if you can see a card, nobody else can be holding it.</p>",
      },
      {
        t: "Two cards for you, five for everyone",
        h: () => {
          const hole = parseCards("Ah Jh"),
            board = parseCards("Js 7h 4h 2c Kh");
          const best = bestFive([...hole, ...board]);
          return (
            "<p>The game here is Texas hold\u2019em. Each player is dealt two private cards, face down. These are your " +
            term("hole cards") +
            ". Only you can see them.</p><p>Over the course of the hand, five more cards are dealt face up in the middle of the table. They are called the " +
            term("board") +
            ", and they belong to everyone.</p><p>Your hand is the <b>best five cards</b> you can make out of those seven. You may use both hole cards, one, or even none.</p>" +
            cardsRow([
              { label: "Your cards", cards: hole, pick: best },
              { label: "Board", cards: board, pick: best },
            ]) +
            "<p>Here the best five are the five hearts, a flush. The pair of jacks is also there, but a hand is always exactly five cards, so you play whichever five are strongest and the other two are ignored.</p>"
          );
        },
      },
      {
        t: "Two ways to win",
        h: () =>
          "<p>Players bet chips into a shared pile called the " +
          term("pot") +
          ". You win the pot in one of two ways.</p><ol><li><b>Everyone else gives up.</b> Giving up is called folding. If all your opponents fold, the pot is yours and you never have to show your cards.</li><li><b>You have the best hand at the end.</b> If two or more players are still in after the last bet, they turn their cards over. This is the " +
          term("showdown") +
          ', and the best five-card hand takes the pot.</li></ol><div class="key">Because of the first way, you can win with bad cards. Because every bet costs chips, you can lose a lot with good cards. Poker skill is deciding how many chips to risk when you cannot see the other hands.</div><p>The rest of this tutorial builds the tools for that decision, one at a time: which hands beat which, how a hand is played, how to count your chances, how to compare those chances with the price of a bet, and how to think about what your opponents might hold.</p>',
      },
    ],
  },
  {
    id: "ranks",
    title: "Which hand beats which",
    blurb: "The nine hand types, why they are ordered that way, and how ties are broken.",
    pages: [
      {
        t: "Nine kinds of hand",
        h: () =>
          "<p>Every five-card hand falls into one of nine types. The order is not arbitrary. <b>The rarer a hand is, the stronger it is.</b> The last column shows how often five random cards make each type.</p>" +
          rankTable() +
          "<p>Half of all five-card hands are nothing at all, and another 42% are just one pair. Everything above one pair is rare, which is why it wins.</p><p>In hold\u2019em you choose five from seven, so good hands appear more often than this table says. You will make at least a pair about 83% of the time by the end of the hand. The ranking order stays the same.</p>",
      },
      {
        t: "Breaking ties",
        h: () => {
          const b = parseCards("Ad 7c 4s 2h Jd");
          return (
            "<p>When two players have the same type of hand, compare the ranks that make it. A pair of kings beats a pair of tens. A flush with an ace beats a flush with a queen on top.</p><p>If those ranks match too, look at the side cards, highest first. The side card that settles it is called the " +
            term("kicker") +
            ".</p>" +
            cardsRow([
              { label: "Ana", cards: parseCards("As Kd") },
              { label: "Ben", cards: parseCards("Ah 9c") },
              { label: "Board", cards: b },
            ]) +
            '<p>Both have a pair of aces. Ana\u2019s five cards are A A K J 7. Ben\u2019s are A A J 9 7. The first side card differs, king against jack, so Ana wins. People say Ben was "out-kicked".</p><p>Remember that only five cards count. If the board were A A K K Q and neither player could improve on it, both would play those same five cards and split the pot.</p>'
          );
        },
      },
      {
        t: "Four traps for beginners",
        h: () =>
          "<ul><li><b>There is no such thing as three pair.</b> A hand is five cards. With three pairs available you play the two highest pairs and one side card.</li><li><b>Suits never break ties.</b> If the best five cards have the same ranks, the pot is split.</li><li><b>A flush does not need to be in a row,</b> and a straight does not need matching suits. Between two flushes, the highest card decides.</li><li><b>The ace is high, with one exception.</b> It can act as a 1 in the lowest straight, A 2 3 4 5. Straights do not wrap around, so Q K A 2 3 is nothing.</li></ul><p>Next, two drills. The questions are dealt at random, so you can repeat them as often as you like.</p>",
      },
      { drill: "nameHand", need: 5, t: "Drill: name the hand" },
      { drill: "whoWins", need: 5, t: "Drill: who wins?" },
    ],
  },
  {
    id: "flow",
    title: "How a hand is played",
    blurb: "Blinds, the button, four rounds of betting, your five possible actions, and seat names.",
    pages: [
      {
        t: "The button and the blinds",
        h: () =>
          "<p>One seat holds a disc called the " +
          term("button") +
          ". It marks the dealer position, and it moves one seat to the left after every hand, so everyone takes turns in each position.</p><p>Before any cards are dealt, the two players to the left of the button must put chips in. The first posts the " +
          term("small blind") +
          " and the next posts the " +
          term("big blind") +
          ', which is twice as much. They are called blinds because you pay them without seeing your cards.</p><p>Blinds exist so that there is always something in the pot to fight over. Without them, the best strategy would be to fold everything except aces, and the game would die.</p><div class="key">Poker players measure everything in big blinds, written <b>bb</b>. A stack of 100 bb means the same whether the big blind is 2 cents or 200 dollars. In this app everyone starts each hand with about 100 bb.</div>',
      },
      {
        t: "Four rounds of betting",
        h: () =>
          "<p>A hand has up to four betting rounds. Each round is called a " +
          term("street") +
          '.</p><table class="ttable"><tr><th>Street</th><th>What is dealt</th><th>Cards you can see</th></tr><tr><td><b>Preflop</b></td><td>two hole cards each</td><td>2</td></tr><tr><td><b>Flop</b></td><td>three board cards at once</td><td>5</td></tr><tr><td><b>Turn</b></td><td>a fourth board card</td><td>6</td></tr><tr><td><b>River</b></td><td>the fifth and last board card</td><td>7</td></tr></table><p>After each deal there is a round of betting that goes clockwise around the table. If more than one player is left after the river betting, there is a showdown.</p><p>Most hands end early. Someone bets, everyone else folds, and the next hand begins.</p>',
      },
      {
        t: "Your five actions",
        h: () =>
          "<p>When it is your turn, what you can do depends on whether someone has already bet in this round.</p><p><b>If nobody has bet:</b></p><ul><li>" +
          term("Check") +
          ": pass the turn without putting chips in. You stay in the hand.</li><li>" +
          term("Bet") +
          ": put chips in. Everyone else must now at least match it to stay.</li></ul><p><b>If someone has bet:</b></p><ul><li>" +
          term("Fold") +
          ": give up. You lose whatever you already put in, and nothing more.</li><li>" +
          term("Call") +
          ": match the bet exactly.</li><li>" +
          term("Raise") +
          ": put in more than the bet, so that everyone else must match your larger amount.</li></ul><p>A round ends when every player still in has put in the same amount, or all but one have folded.</p><p>Before the flop the big blind counts as a bet, so you cannot check unless you are the big blind and nobody raised. Just calling the big blind is called a " +
          term("limp") +
          ". The first raise is called an " +
          term("open") +
          ". A raise over that is a " +
          term("3-bet") +
          ", because the blind was the first bet, the open the second, and this is the third.</p>",
      },
      {
        t: "Seat names and who acts when",
        h: () =>
          '<p>Seats are named by where they sit relative to the button. With six players, going clockwise from the button:</p><table class="ttable"><tr><th>Seat</th><th>Name</th><th>Note</th></tr><tr><td><b>BTN</b></td><td>button</td><td>acts last on the flop, turn and river</td></tr><tr><td><b>SB</b></td><td>small blind</td><td>acts first on the flop, turn and river</td></tr><tr><td><b>BB</b></td><td>big blind</td><td>acts last before the flop</td></tr><tr><td><b>UTG</b></td><td>under the gun</td><td>acts first before the flop</td></tr><tr><td><b>HJ</b></td><td>hijack</td><td></td></tr><tr><td><b>CO</b></td><td>cutoff</td><td>just before the button</td></tr></table><p><b>Before the flop</b> the order is UTG, HJ, CO, BTN, SB, BB. The blinds act last because they have already paid.</p><p><b>On every later street</b> the order is SB, BB, UTG, HJ, CO, BTN, skipping anyone who folded.</p><div class="key">Acting after your opponent is called being ' +
          term("in position") +
          ". You see what they do before you decide, on every street. That information is worth real money, and it is why the button is the best seat and the blinds are the worst.</div>",
      },
      { drill: "position", need: 5, t: "Drill: who acts when?" },
    ],
  },
  {
    id: "outs",
    title: "Counting your chances",
    blurb: "Unseen cards, outs, exact probabilities, and the shortcut called the rule of 2 and 4.",
    pages: [
      {
        t: "Probability from counting",
        h: () => {
          const hero = parseCards("Ah 9h"),
            board = parseCards("Kh 6h 2c");
          const seen = new Set([...hero, ...board].map(cardKey));
          const outs = new Set();
          for (const c of newDeck()) if (c.s === 1 && !seen.has(cardKey(c))) outs.add(cardKey(c));
          return (
            "<p>When every outcome is equally likely, probability is just counting:</p>" +
            eqn("chance = " + frac("outcomes you want", "all possible outcomes")) +
            "<p>On the flop you can see 5 cards, your 2 plus 3 on the board. The other 52 \u2212 5 = 47 are " +
            term("unseen") +
            ". Some of them are in other players\u2019 hands, but you do not know which, so from where you sit every unseen card is equally likely to come next.</p><p>An " +
            term("out") +
            " is an unseen card that turns your hand into the likely winner.</p>" +
            cardsRow([
              { label: "Your cards", cards: hero },
              { label: "Flop", cards: board },
            ]) +
            "<p>You have four hearts and need a fifth. This is a " +
            term("flush draw") +
            ". There are 13 hearts in the deck and you can see 4, so 13 \u2212 4 = <b>9 outs</b>.</p>" +
            deckGrid(seen, outs) +
            eqn("chance on the next card = " + frac(9, 47) + " = 19.1%")
          );
        },
      },
      {
        t: "Two cards to come",
        h: () =>
          "<p>On the flop there are two cards still to come, the turn and the river. What is the chance that <b>at least one</b> of them is a heart?</p><p>The clean way is to work out the chance of missing both, then subtract from 1.</p><ul><li>Miss the turn: 38 of the 47 unseen cards are not hearts.</li><li>Then miss the river: one non-heart is gone, so 37 of the remaining 46.</li></ul>" +
          eqn("miss both = " + frac(38, 47) + " \u00d7 " + frac(37, 46) + " = 65.0%") +
          eqn("hit at least once = 1 \u2212 65.0% = 35.0%") +
          "<p>Why multiply? Out of all the times you miss the turn, you then miss the river in 37 of every 46. A fraction of a fraction is a product.</p><p>So a flush draw on the flop arrives about one time in three by the river, and about one time in five on any single card. Those two numbers are worth memorising.</p>",
      },
      {
        t: "The rule of 2 and 4",
        h: () =>
          '<p>Nobody multiplies fractions at the table. There is a shortcut.</p><div class="key"><b>One card to come:</b> chance \u2248 outs \u00d7 2.<br><b>Two cards to come:</b> chance \u2248 outs \u00d7 4.</div><p>It works because one unseen card is about 1 in 47, which is close to 2%. So each out is worth about 2% per card dealt.</p>' +
          hitTable() +
          "<p>The shortcut is within a point or two until the outs get large. With 15 outs the \u00d74 rule says 60% and the truth is 54%, because the rule counts the times you hit on both cards twice.</p>",
      },
      {
        t: "The common draws, and a warning",
        h: () =>
          '<ul><li><b>Flush draw:</b> four of a suit, 9 outs.</li><li><b>Open-ended straight draw:</b> four in a row such as 8 9 T J, where a 7 or a Q completes it. Two ranks, four cards each, 8 outs.</li><li><b>Gutshot:</b> a gap in the middle such as 8 9 J Q, where only a T completes it. 4 outs.</li><li><b>Flush draw plus open-ended straight draw:</b> 9 + 8 would be 17, but two of the straight cards are also flush cards. Each card can only arrive once, so the answer is 15.</li></ul><div class="key">An out only counts if it makes you the <b>winner</b>. If the card that completes your straight also puts a third heart on the board, someone else may make a flush. Be honest when you count, and discount outs that could help an opponent even more.</div>',
      },
      { drill: "outs", need: 5, t: "Drill: count the outs" },
      { drill: "hitChance", need: 4, t: "Drill: how often do you hit?" },
    ],
  },
  {
    id: "odds",
    title: "Expected value and pot odds",
    blurb: "What a good decision means, the break-even formula derived step by step, and when to call.",
    pages: [
      {
        t: "Expected value",
        h: () =>
          "<p>Suppose someone offers you a coin flip. Heads you win 3 dollars, tails you lose 2. Should you play?</p><p>On any one flip you cannot know. Over many flips, half are heads and half are tails, so the average result per flip is</p>" +
          eqn("0.5 \u00d7 3 \u2212 0.5 \u00d7 2 = +0.50") +
          "<p>That average is the " +
          term("expected value") +
          ', or EV. In general you multiply each possible result by its probability and add them up.</p><div class="key">A good decision is one with the highest EV. It is still a good decision when it loses this time, and a bad decision is still bad when it gets lucky. You cannot control the cards. You can only control the average.</div><p>Every poker decision is a gamble of this kind. The job is to estimate the probabilities and the payoffs, then choose the option with the best average.</p>',
      },
      {
        t: "The EV of a call",
        h: () =>
          '<p>Someone bets and you must call or fold. Use three numbers.</p><ul><li><i class="v">P</i> is the pot right now, including the bet you are facing.</li><li><i class="v">c</i> is what it costs to call.</li><li><i class="v">e</i> is the chance you end up winning.</li></ul><p>If you call and win, you gain <i class="v">P</i>. If you call and lose, you lose <i class="v">c</i>.</p>' +
          eqn("EV of calling = <i>e</i> \u00d7 <i>P</i> \u2212 (1 \u2212 <i>e</i>) \u00d7 <i>c</i>") +
          "<p>And folding?</p>" +
          eqn("EV of folding = 0") +
          '<p>This surprises people. The chips you put in earlier are <b>already gone</b>. They belong to the pot, not to you. Folding loses nothing further from this point, so it scores zero. Feeling that you must "protect" chips already in the pot is one of the most expensive mistakes in poker.</p>',
      },
      {
        t: "Pot odds: the break-even point",
        h: () =>
          '<p>Calling is right when its EV is above zero. Find the value of <i class="v">e</i> where it is exactly zero.</p>' +
          eqn("<i>e</i> \u00d7 <i>P</i> \u2212 (1 \u2212 <i>e</i>) \u00d7 <i>c</i> = 0") +
          eqn("<i>e</i> \u00d7 <i>P</i> + <i>e</i> \u00d7 <i>c</i> = <i>c</i>") +
          eqn("<i>e</i> = " + frac("<i>c</i>", "<i>P</i> + <i>c</i>")) +
          "<p>This is the most useful formula in poker. It is called your " +
          term("pot odds") +
          '. The price of the call, divided by the final pot with your call included, is the share of the time you need to win.</p><p><b>Example.</b> The pot is 10 bb and your opponent bets 5. Now <i class="v">P</i> = 15 and <i class="v">c</i> = 5, so you need 5 / 20 = 25%.</p>' +
          priceTable(false) +
          "<p>Look at how low these numbers are. Small bets need very little equity to call. Even a bet of twice the pot needs only 40%.</p>",
      },
      {
        t: "Putting the pieces together",
        h: () =>
          "<p>You now have both halves: the chance of winning from the last lesson, and the price from this one.</p><p><b>Example.</b> On the turn you have a flush draw, 9 outs, and one card to come, so you hit 9 / 46 = 19.6% of the time. Your opponent bets half the pot, which needs 25%. You have less than you need, so if nothing else happens, calling loses money.</p><p>Against a bet of one quarter of the pot you would need 1 / 6 = 16.7%, and now the same draw is a profitable call. Same cards, different price, different answer.</p><h3>One refinement</h3><p>If you hit your draw, you can often win more chips on the next street. That expected extra is called " +
          term("implied odds") +
          ", and it lets you call when you are slightly short of the price, provided there are plenty of chips left to win. On the river there are no more streets, so implied odds are zero.</p>",
      },
      { drill: "potOdds", need: 5, t: "Drill: what equity do you need?" },
      { drill: "callOrFold", need: 5, t: "Drill: call or fold?" },
    ],
  },
  {
    id: "equity",
    title: "Equity",
    blurb: "Your share of the pot, the classic matchups with live simulations, and why more opponents shrink it.",
    pages: [
      {
        t: "Your share of the pot",
        h: () =>
          '<p>In the last lesson <i class="v">e</i> was "the chance you hit your draw". That is a simplification. Sometimes you win without improving. Sometimes you hit and still lose. Sometimes the pot is split.</p><p>The precise version of <i class="v">e</i> is ' +
          term("equity") +
          ": your average share of the pot if all the remaining cards were dealt out now, with no more betting.</p><p>If you have 60% equity in a 10 bb pot, that pot is worth 6 bb to you on average. Nobody hands you 6 bb. You get all 10 or nothing. But over many repeats it averages out to 6.</p><p>Equity cannot be calculated in your head. This app measures it by " +
          term("simulation") +
          ": it deals the rest of the hand thousands of times at random and counts how often you win. What you can do is learn the common situations by heart.</p>",
      },
      {
        t: "Matchup lab",
        h: () =>
          '<p>Tap a matchup. The app deals out 6,000 random boards and shows how often each hand wins when both players are all-in before the flop.</p><div class="lab" id="mlab"><div class="chips">' +
          MATCHUPS.map((m, i) => '<button data-i="' + i + '">' + m[2] + "</button>").join("") +
          '</div><div id="mout" style="color:var(--ink-soft)">Pick a matchup above.</div></div><p>Things to notice:</p><ul><li>A higher pair against a lower pair is about 80 to 20. This is the most lopsided common matchup.</li><li>A pair against two higher cards is close to even. Poker players call it a coin flip, though the pair is a slight favourite.</li><li>Dominated hands do badly, around 25 to 30%. This is why A6 is much weaker than it looks: when another ace is out there, it usually has a better side card.</li><li>Two high cards against two low cards is only about 60 to 40. Preflop, no unpaired hand is a big favourite over another.</li></ul>',
        m: (root) => {
          root.querySelectorAll("#mlab .chips button").forEach(
            (b) =>
              (b.onclick = () => {
                root.querySelectorAll("#mlab .chips button").forEach((x) => x.classList.remove("on"));
                b.classList.add("on");
                const m = MATCHUPS[+b.dataset.i];
                const h1 = parseCards(m[0]),
                  h2 = parseCards(m[1]);
                const eq = equityHeadsUp(h1, h2, [], 6000);
                root.querySelector("#mout").innerHTML =
                  cardsRow([
                    { label: "Hand A", cards: h1 },
                    { label: "Hand B", cards: h2 },
                  ]) +
                  '<div class="eqbar"><span class="a" style="width:' +
                  eq * 100 +
                  '%">A ' +
                  pc(eq) +
                  '</span><span class="b" style="width:' +
                  (1 - eq) * 100 +
                  '%">B ' +
                  pc(1 - eq) +
                  "</span></div>";
              })
          );
        },
      },
      {
        t: "More opponents, less equity",
        h: () =>
          '<p>With several players in the pot, you have to beat all of them at once. Even the best starting hand shrinks quickly.</p><div class="lab" id="alab"><div style="margin-bottom:6px">Pocket aces against this many random hands:</div><div class="chips">' +
          [1, 2, 3, 5, 8].map((n) => '<button data-n="' + n + '">' + n + "</button>").join("") +
          '</div><div id="aout" style="color:var(--ink-soft)">Pick a number.</div></div><p>Against one random hand, aces win about 85% of the time. Against five they win about half. Aces are still by far the best hand, since a fair share among six players would be 17%, but they are no longer close to certain.</p><div class="key">Two lessons follow. Strong hands want <b>fewer</b> opponents, which is one reason to raise rather than just call. And the equity you need to continue should be compared with the price you are offered, never with 50%.</div>',
        m: (root) => {
          root.querySelectorAll("#alab .chips button").forEach(
            (b) =>
              (b.onclick = () => {
                root.querySelectorAll("#alab .chips button").forEach((x) => x.classList.remove("on"));
                b.classList.add("on");
                const n = +b.dataset.n;
                const eq = simulate(parseCards("As Ah"), [], new Array(n).fill({ pct: 100, filters: [] }), 3000, rng);
                root.querySelector("#aout").innerHTML =
                  '<div class="eqbar"><span class="a" style="width:' +
                  eq * 100 +
                  '%">Aces ' +
                  pc(eq) +
                  '</span><span class="b" style="width:' +
                  (1 - eq) * 100 +
                  '%">' +
                  pc(1 - eq) +
                  '</span></div><div style="margin-top:6px;color:var(--ink-soft)">A fair share among ' +
                  (n + 1) +
                  " players would be " +
                  pc(1 / (n + 1)) +
                  ".</div>";
              })
          );
        },
      },
      { drill: "equityGuess", need: 5, t: "Drill: estimate the equity" },
    ],
  },
  {
    id: "start",
    title: "Starting hands and position",
    blurb: "The 169 starting hands, what makes one good, and why your seat decides how many you play.",
    pages: [
      {
        t: "1,326 combos, 169 hands",
        h: () =>
          "<p>There are 52 \u00d7 51 / 2 = 1,326 different two-card holdings. Each specific one, like A\u2660 K\u2665, is called a " +
          term("combo") +
          '.</p><p>Before the flop, suits only matter in one way: whether your two cards share a suit or not. That collapses the 1,326 combos into 169 kinds of hand.</p><table class="ttable"><tr><th>Kind</th><th>Written</th><th>How many kinds</th><th>Combos each</th></tr><tr><td>Pair</td><td>QQ</td><td>13</td><td>6</td></tr><tr><td>Suited (same suit)</td><td>AKs</td><td>78</td><td>4</td></tr><tr><td>Offsuit (different suits)</td><td>AKo</td><td>78</td><td>12</td></tr></table><p>Check: 13 \u00d7 6 + 78 \u00d7 4 + 78 \u00d7 12 = 78 + 312 + 936 = 1,326.</p><p>Poker players draw all 169 on a 13 by 13 grid. Pairs run down the diagonal, suited hands sit above it, and offsuit hands below. Here the strongest fifth of hands is shaded.</p>' +
          legacyRangeGrid(20, null) +
          "<p>Notice that an offsuit hand is three times as common as its suited twin. When you picture what an opponent holds, there is much more AKo around than AKs.</p>",
      },
      {
        t: "What makes a starting hand good",
        h: () =>
          "<ul><li><b>High cards.</b> When you pair one, you have the top pair, and it is more likely to be the best pair.</li><li><b>A pair.</b> You already have a made hand, and about one flop in eight gives you three of a kind.</li><li><b>Suited.</b> It adds a few points of equity through flushes. Helpful, but much less than beginners think. A bad hand is still bad when it is suited.</li><li><b>Connected.</b> Cards close in rank, like 9 8, can make straights.</li></ul><h3>Domination</h3><p>The quiet killer is being " +
          term("dominated") +
          ". Suppose you hold K9 and your opponent holds KQ. When a king comes you both have a pair of kings, but their queen beats your nine. You will put in a lot of chips with the second-best hand, and the matchup lab showed you win only about a quarter of the time.</p><p>This is why weak aces and weak kings are worse than they look. The hands that are happy to play a big pot against you are exactly the ones that dominate you.</p>",
      },
      {
        t: "Why your seat matters so much",
        h: () =>
          '<p>Imagine you are first to act with five players still to come. Call the top 5% of hands "very strong". What is the chance that at least one of those five players holds one?</p>' +
          eqn("1 \u2212 0.95<sup>5</sup> = 22.6%") +
          "<p>On the button there are only the two blinds behind you.</p>" +
          eqn("1 \u2212 0.95<sup>2</sup> = 9.8%") +
          '<p>So an early raise runs into a big hand more than twice as often. On top of that, the button acts last on every later street, and early seats usually do not.</p><p>Both effects point the same way: <b>play few hands from early seats and many from late seats.</b> A simple guide for six players, when nobody has entered the pot before you:</p><table class="ttable"><tr><th>Seat</th><th>Raise with roughly</th></tr>' +
          ["UTG", "HJ", "CO", "BTN", "SB"].map((x) => "<tr><td>" + x + "</td><td>the top " + openPct(x, 6) + "% of hands</td></tr>").join("") +
          '</table><p>These come from published charts that simplify computer-solver output for six players with 100 bb stacks. At a bigger table the first seats have more players behind them and open tighter still, around 10 to 13%. The shape is what matters: tight early, wide late.</p><p>The chart is not simply "the prettiest hands first". It likes suited kings and suited aces more than beginners expect, and offsuit connected cards like T9o less. The next page explains why.</p>',
      },
      {
        t: "Raw equity is not the whole story",
        h: () => {
          const buttonRange = [{ pct: openPct("BTN", 6), filters: [] }];
          const e1 = simulate(parseCards("Kc 9d"), [], buttonRange, 4000, rng),
            e2 = simulate(parseCards("8s 7s"), [], buttonRange, 4000, rng);
          return (
            "<p>Equity assumes you always see all five cards. In real play you often do not. You miss the flop, your opponent bets, and you fold a hand that would have won some of the time. The share of your equity that you actually collect is called " +
            term("equity realisation") +
            '.</p><p>Here are two hands against a button raise, simulated just now:</p><table class="ttable"><tr><th>Hand</th><th>Raw equity</th><th>What happens after the flop</th></tr><tr><td><b>K9 offsuit</b></td><td>' +
            pc(e1) +
            "</td><td>Usually one weak pair or nothing. Hard to continue against bets, and when the king pairs you are often out-kicked.</td></tr><tr><td><b>87 suited</b></td><td>" +
            pc(e2) +
            '</td><td>Often flops a straight draw, flush draw or pair plus draw. Easy to continue, and its big hands are well hidden.</td></tr></table><p>K9 offsuit has the higher raw number, yet charts prefer 87 suited, because 87 suited collects more of what it is owed. Three things decide how much you realise:</p><ul><li><b>Position.</b> This is the biggest factor. Acting last, you take free cards when you want them and you are harder to push out.</li><li><b>Suited and connected cards.</b> They flop more hands that can stand a bet.</li><li><b>How strong your range is on this board.</b> The player with the stronger range can bet more, and the other player folds more.</li></ul><div class="key">This is why the Trainer shows no "equity against price" number before the flop. Raw equity would tell you to call far too often. Before the flop, trust the chart. After the flop, equity against their range is a much better guide.</div>'
          );
        },
      },
      {
        t: "Raise or fold, and facing a raise",
        h: () =>
          '<h3>When nobody has entered: raise or fold</h3><p>Beginners like to limp, which means just calling the big blind. A raise is almost always better, for three reasons. You can win the blinds immediately. You thin the field, and the last lesson showed that strong hands want fewer opponents. And you build a bigger pot for the times you are ahead.</p><p>A normal opening raise is 2.5 bb, plus about 1 bb for each player who limped in before you.</p><h3>When someone has already raised: tighten up</h3><p>A raise tells you that player has a good hand. To continue you need a hand that does well against <b>good hands</b>, which is a much higher bar than doing well against random cards. A rough guide:</p><ul><li>Re-raise (3-bet) with about the best 6%.</li><li>Call with roughly the next 12%, a little more on the button, a little less in the blinds.</li><li>Fold everything else, including pretty-looking dominated hands like KJ offsuit against an early raise.</li></ul><h3>The big blind is different</h3><p>In the big blind you have already paid 1 bb and nobody can raise behind you, so your price is much better and you continue far wider. How wide depends on who raised, because a late raiser has a weaker range:</p><table class="ttable"><tr><th>Raise comes from</th><th>Big blind continues with about</th></tr><tr><td>UTG or HJ</td><td>the top 30 to 35%</td></tr><tr><td>CO</td><td>top 40%</td></tr><tr><td>BTN</td><td>top 48%</td></tr><tr><td>SB</td><td>top 65%</td></tr></table><p>Folding the big blind too much is one of the most common leaks. Some solver charts defend even wider than this against late raises.</p><h3>One thing this trainer simplifies</h3><p>Here your re-raising range is just your very best hands. Strong players also re-raise some hands as bluffs, typically small suited aces like A5s, which block the opponent\'s aces and still make flushes. That keeps them from being easy to read. It is a refinement for later.</p>',
      },
      { drill: "openOrFold", need: 6, t: "Drill: raise or fold?" },
    ],
  },
  {
    id: "ranges",
    title: "Ranges, combos and blockers",
    blurb: "Stop guessing one hand. Count all the hands they could have, and let visible cards remove some.",
    pages: [
      {
        t: "Think in ranges",
        h: () =>
          '<p>Beginners ask "what does he have?" and pick one hand. It is the wrong question, because nobody can answer it.</p><p>The right question is "what are <b>all</b> the hands he would play this way?" That whole set is called his ' +
          term("range") +
          '.</p><p>A range gets narrower with every action.</p><ol><li>A tight player raises from UTG. The range is about the top 12%: big pairs, big aces, the best suited high cards.</li><li>The flop comes 9 6 2 with three different suits, and he bets. Most of his range missed this flop, but he might bet anyway. Little changes.</li><li>The turn is a 3. He bets large again. Now the hands that missed mostly drop away, and what is left is weighted towards big pairs.</li></ol><p>You never reach certainty. You end with a list of possible hands and a rough idea of how likely each is. Your equity against that list is what you compare with the price.</p><p>In Trainer mode, the "Their ranges" panel does this narrowing for you on the 13 by 13 grid, so you can watch it happen.</p>',
      },
      {
        t: "Counting combos",
        h: () =>
          "<p>To weigh the hands in a range you need to know how many ways each can be dealt.</p><ul><li>An unpaired hand like AK has 4 aces \u00d7 4 kings = <b>16 combos</b>. Four are suited and twelve are offsuit.</li><li>A pair like QQ has <b>6 combos</b>, the number of ways to choose 2 queens from 4.</li></ul><h3>Blockers</h3><p>Any card you can see cannot be in your opponent\u2019s hand. Seeing a card therefore removes combos from their range. Such a card is called a " +
          term("blocker") +
          ".</p><ul><li>You hold an ace. There are 3 aces left, so AK drops from 16 to 3 \u00d7 4 = <b>12</b>, and AA drops from 6 to <b>3</b>.</li><li>The flop has a king on it. Pocket kings, which would now be three of a kind, drop from 6 combos to 3.</li></ul><h3>Why this matters</h3><p>On a flop of K 7 2 you may fear three of a kind. But KK, 77 and 22 add up to only 3 + 3 + 3 = 9 combos. Hands like AK, KQ and KJ, which are just one pair of kings, add up to 12 + 12 + 12 = 36. Any given bet is four times as likely to be one pair as three of a kind. Counting replaces fear with a number.</p>",
      },
      { drill: "combos", need: 5, t: "Drill: count the combos" },
    ],
  },
  {
    id: "texture",
    title: "Your hand on this board",
    blurb: "The vocabulary for made hands, and how dry and wet boards change what they are worth.",
    pages: [
      {
        t: "A pair is not just a pair",
        h: () =>
          '<p>"I have a pair" says almost nothing. What matters is how your pair relates to the board. These names are used constantly.</p>' +
          cardsRow([
            { label: "Overpair", cards: parseCards("Qs Qh") },
            { label: "Flop", cards: parseCards("9d 6c 2s") },
          ]) +
          "<p>A pocket pair higher than every board card. It beats any hand that paired the board.</p>" +
          cardsRow([
            { label: "Top pair", cards: parseCards("As 9h") },
            { label: "Flop", cards: parseCards("9d 6c 2s") },
          ]) +
          "<p>You paired the highest board card. The other hole card is your kicker, and a strong kicker matters a great deal.</p>" +
          cardsRow([
            { label: "Second pair", cards: parseCards("As 6h") },
            { label: "Flop", cards: parseCards("9d 6c 2s") },
          ]) +
          "<p>You paired the middle card. It is a medium hand, good for showing down cheaply, poor for building a big pot.</p>" +
          cardsRow([
            { label: "Set", cards: parseCards("6s 6h") },
            { label: "Flop", cards: parseCards("9d 6c 2s") },
          ]) +
          "<p>Three of a kind using a pocket pair. It is very strong and very well hidden. When the three of a kind uses a pair on the board plus one of your cards, it is called " +
          term("trips") +
          ", and everyone can see that it is possible.</p>",
      },
      {
        t: "Dry boards and wet boards",
        h: () =>
          "<p>The " +
          term("texture") +
          " of the board tells you how many strong hands and draws are possible.</p>" +
          cardsRow([{ label: "Dry", cards: parseCards("Ks 7d 2c") }]) +
          "<p>Three different suits, which players call a " +
          term("rainbow") +
          " flop, and ranks far apart. No flush draw and almost no straight draw exists. Whoever is ahead now will usually still be ahead on the river.</p>" +
          cardsRow([{ label: "Wet", cards: parseCards("9h 8h 7c") }]) +
          "<p>Two hearts and three ranks in a row. Straights are already possible, and many hands have a draw. The best hand now is often not the best hand two cards later.</p><h3>What changes</h3><ul><li><b>On dry boards</b> you can bet small. Nobody has a draw to charge, and small bets get called by weaker hands.</li><li><b>On wet boards</b> bet bigger with your good hands, so that draws pay a bad price. Go back to the price table: a flush draw on the turn hits 19.6% of the time, so any bet of half the pot or more makes calling a mistake.</li><li><b>Paired boards,</b> such as 8 8 3, make full houses possible and make two pair much weaker, since everyone shares the pair of eights.</li></ul>",
      },
      {
        t: "Who does the board favour?",
        h: () =>
          "<p>Think back to ranges. The player who raised before the flop holds a lot of high cards and big pairs. The player who called holds more medium pairs and suited connected cards.</p><ul><li>A board like <b>A K 5</b> hits the raiser\u2019s range hard. They can bet often, and you should believe them more.</li><li>A board like <b>7 6 5</b> hits the caller\u2019s range. The raiser\u2019s ace-king has nothing, and should slow down.</li></ul><p>So the same bet means different things on different boards. Before reacting to a bet, ask whether this board is good for the person betting.</p>",
      },
      { drill: "handClass", need: 5, t: "Drill: what do you have?" },
      { drill: "texture", need: 4, t: "Drill: wet or dry?" },
    ],
  },
  {
    id: "betting",
    title: "Betting: why and how much",
    blurb: "The only two reasons to bet, the price your bet sets, bluff maths, and stack-to-pot ratio.",
    pages: [
      {
        t: "Two reasons to bet",
        h: () =>
          "<p>Before every bet, ask what you want to happen next. There are only two good answers.</p><ol><li><b>Value.</b> You want a <b>worse</b> hand to call. You are probably ahead, and you are building the pot.</li><li><b>Bluff.</b> You want a <b>better</b> hand to fold. You are probably behind, and folding is the only way you win.</li></ol><p>If neither is likely, do not bet. This happens all the time with medium hands like second pair. Bet, and the worse hands fold while the better hands call. You win nothing extra when ahead and lose more when behind. With these hands, checking is usually best.</p><h3>The semi-bluff</h3><p>A bet made with a draw is called a " +
          term("semi-bluff") +
          ". It has two ways to win: everyone may fold now, or you may be called and then hit. That second chance makes semi-bluffs far safer than pure bluffs. If you are going to bluff, bluff with hands that can improve.</p>",
      },
      {
        t: "Your bet sets their price",
        h: () =>
          '<p>You already know pot odds from the caller\u2019s side. Now you are the one naming the price.</p><p>If you bet <i class="v">B</i> into a pot of <i class="v">P</i>, the caller pays <i class="v">B</i> for a final pot of <i class="v">P</i> + 2<i class="v">B</i>.</p>' +
          eqn("equity the caller needs = " + frac("<i>B</i>", "<i>P</i> + 2<i>B</i>")) +
          priceTable(false) +
          "<p><b>For value, bet as much as worse hands will still call.</b> Bigger bets win more when called, but fewer hands call them. Two-thirds of the pot is a sound default on boards with draws. On dry boards smaller bets do the same job, as lesson 12 explains.</p><p><b>Against draws, make the price wrong.</b> A flush draw on the turn has 19.6%. A half-pot bet asks for 25%. If they call, they lose money on average, which means you gain it.</p>",
      },
      {
        t: "When does a bluff make money?",
        h: () =>
          '<p>Suppose you bet <i class="v">B</i> as a pure bluff, with no chance of winning if called, and your opponent folds a fraction <i class="v">f</i> of the time.</p>' +
          eqn("EV = <i>f</i> \u00d7 <i>P</i> \u2212 (1 \u2212 <i>f</i>) \u00d7 <i>B</i>") +
          "<p>Set that to zero and solve, exactly as you did for pot odds.</p>" +
          eqn("<i>f</i> = " + frac("<i>B</i>", "<i>P</i> + <i>B</i>")) +
          priceTable(true) +
          "<p>The right-hand column is new. A half-pot bluff has to work one time in three. A pot-sized bluff has to work half the time.</p><p>The value gained from the chance that everyone folds is called " +
          term("fold equity") +
          '. Three things reduce it: more opponents, opponents who love to call, and boards that fit their range.</p><div class="key">Beginners bluff too often, against too many players, and against the wrong players. Until you can name the better hands you expect to fold, do not bluff.</div>',
      },
      {
        t: "Stack-to-pot ratio",
        h: () =>
          "<p>How strong a hand you need depends on how much is left to bet. The " +
          term("stack-to-pot ratio") +
          ", or SPR, measures it.</p>" +
          eqn("SPR = " + frac("smaller of the two stacks", "pot at the start of the street")) +
          "<ul><li><b>SPR below 3.</b> There is little left behind. One good pair is usually enough to put all your chips in, because folding gives up a large pot to save a small amount.</li><li><b>SPR from 3 to 8.</b> Top pair or an overpair is fine against one opponent. Be careful against several.</li><li><b>SPR above 8.</b> Stacks are deep. Playing for everything needs a very strong hand, two pair or better. Draws gain value because there is a lot to win when they arrive.</li></ul><p>Trainer mode shows the SPR on every decision after the flop.</p>",
      },
      { drill: "betPrice", need: 5, t: "Drill: prices and bluffs" },
    ],
  },
  {
    id: "defend",
    title: "Defending against bets",
    blurb: "Why you cannot just fold, minimum defence frequency, bluff-to-value ratios, and when to ignore both.",
    pages: [
      {
        t: "The problem with folding too much",
        h: () =>
          '<p>So far, facing a bet, you compared your equity with the price. That works when you know your opponent\u2019s range. But it hides a danger.</p><p>Suppose you fold to a pot-sized bet 70% of the time. Your opponent can now bet <b>any two cards</b> and profit. They risk one pot to win one pot, so they break even if you fold half the time, and you fold far more than that. Their cards do not matter.</p><p>Go back to the bluff formula from lesson 10. A pure bluff of <i class="v">B</i> into a pot of <i class="v">P</i> makes money when you fold more often than</p>' +
          eqn(frac("<i>B</i>", "<i>P</i> + <i>B</i>")) +
          "<p>To stop that, you must continue with at least the rest of your range:</p>" +
          eqn("MDF = 1 \u2212 " + frac("<i>B</i>", "<i>P</i> + <i>B</i>") + " = " + frac("<i>P</i>", "<i>P</i> + <i>B</i>")) +
          "<p>This is the " +
          term("minimum defence frequency") +
          ', MDF. Here <i class="v">P</i> is the pot <b>before</b> the bet.</p><table class="ttable"><tr><th>Bet size</th><th>You must continue with at least</th></tr><tr><td>A third of the pot</td><td>75% of your range</td></tr><tr><td>Half the pot</td><td>67%</td></tr><tr><td>Two-thirds of the pot</td><td>60%</td></tr><tr><td>The whole pot</td><td>50%</td></tr><tr><td>Twice the pot</td><td>33%</td></tr></table><p>Against small bets you should fold very little.</p>',
      },
      {
        t: "Which hands continue",
        h: () =>
          "<p>MDF is about your whole <b>range</b>, not the two cards in front of you. The method is:</p><ol><li>List every hand you could hold here, given how you have played.</li><li>Rank them from best to worst on this board.</li><li>Continue with the top share that MDF demands. Raise some of the very best, call with the rest, fold the bottom.</li></ol><p>A hand that beats only bluffs is called a " +
          term("bluff-catcher") +
          ". Second pair on the river against a big bet is the classic example. It loses to every value bet and beats every bluff, so its whole value depends on how often your opponent bluffs.</p><p>When you must pick between bluff-catchers, blockers break the tie. Prefer calling with hands that hold a card your opponent\u2019s <b>value</b> hands need, and avoid hands that hold a card their <b>bluffs</b> need. If the obvious bluffs are missed heart draws, holding a heart yourself makes a bluff less likely, which makes your call worse.</p>",
      },
      {
        t: "The mirror image: how much to bluff",
        h: () =>
          '<p>Turn it around. On the river you bet with some strong hands and some bluffs. What mix leaves your opponent with no good answer?</p><p>They pay <i class="v">B</i> for a final pot of <i class="v">P</i> + 2<i class="v">B</i>, so a call needs to win</p>' +
          eqn(frac("<i>B</i>", "<i>P</i> + 2<i>B</i>")) +
          '<p>of the time. If exactly that share of your bets are bluffs, calling and folding earn them the same. Bluff more and they profit by always calling. Bluff less and they profit by always folding.</p><table class="ttable"><tr><th>River bet</th><th>Bluffs in your betting range</th><th>Value hands per bluff</th></tr><tr><td>Half the pot</td><td>25%</td><td>3</td></tr><tr><td>The whole pot</td><td>33%</td><td>2</td></tr><tr><td>Twice the pot</td><td>40%</td><td>1.5</td></tr></table><p>A range built like this, strong hands plus bluffs with little in between, is called ' +
          term("polarised") +
          ". Notice that <b>bigger bets are allowed more bluffs</b>. That is one reason strong players bet big when their range is polarised.</p><p>On earlier streets your bluffs are usually draws, which still have equity, so you can bluff more often than these river numbers.</p>",
      },
      {
        t: "When to ignore all of this",
        h: () =>
          "<p>MDF and balanced bluffing describe play that <b>cannot be exploited</b>. Poker players call this game theory optimal, or GTO. It is the right default against strong, unknown opponents.</p><p>It is not the most profitable play against opponents with obvious habits.</p><ul><li>Against someone who almost never bluffs, MDF makes you pay off their value bets. Fold more than MDF says.</li><li>Against someone who bluffs constantly, call more than MDF says.</li><li>Against someone who never folds, your balanced bluffs just burn chips. Stop bluffing and bet your good hands bigger.</li></ul><p>Adjusting like this is called playing " +
          term("exploitatively") +
          '. It wins more, and it opens you up to being exploited in turn if you have misjudged them.</p><div class="key">MDF also has a built-in simplification: it assumes a bluff has no chance of winning when called. Before the river most bluffs have some equity, and solvers defend a little less than MDF when out of position. Treat it as a guide to "am I folding far too much?", not as a law.</div><p><b>About this trainer.</b> The bots here bluff less than a balanced player. So the coach decides by your equity against their range, which means folding more than MDF. That is the right adjustment against them and against most casual players. Against strong players it would make you too easy to push around. The review after each bet you face shows the MDF number so you can see the gap.</p>',
      },
      { drill: "mdf", need: 5, t: "Drill: defend and bluff in the right proportions" },
    ],
  },
  {
    id: "cbet",
    title: "Range advantage and the flop bet",
    blurb: "Why the player who raised often bets small with everything, and when they should not.",
    pages: [
      {
        t: "Whose flop is it?",
        h: () =>
          "<p>When you raise before the flop and then bet the flop, that bet is called a " +
          term("continuation bet") +
          ", or c-bet. How often should you make it? The modern answer starts with ranges, not with your own two cards.</p><p>Compare the two ranges from lesson 7. The raiser holds more big pairs and big aces. The big blind, who called, holds more middling suited and connected cards, and would have re-raised their very best hands.</p><ul><li>" +
          term("Range advantage") +
          ": whose whole range has more equity on this board.</li><li>" +
          term("Nut advantage") +
          ": who holds more of the very strongest hands, such as sets, two pair and overpairs.</li></ul>" +
          cardsRow([{ label: "Raiser\u2019s flop", cards: parseCards("Ks 7d 2c") }]) +
          "<p>The raiser has every AK, KQ, AA and KK. The caller has fewer strong kings and almost no big pairs, because they would have re-raised those. The raiser has both advantages.</p>" +
          cardsRow([{ label: "Caller\u2019s flop", cards: parseCards("7h 6s 5d") }]) +
          "<p>The caller\u2019s suited connectors and small pairs have made straights, sets and two pair. The raiser\u2019s AK and AQ have nothing. The advantages have moved to the caller.</p>",
      },
      {
        t: "Frequency and size",
        h: () =>
          '<p>Studying solver output gives a rule worth memorising:</p><div class="key"><b>Range advantage decides how often you bet.<br>Nut advantage decides how big you bet.</b></div><h3>Dry, high-card flops: bet small, bet often</h3><p>On K 7 2 you have both advantages. The standard play is a small bet, about a third of the pot, with <b>most of your range</b>, strong and weak hands alike. From the lesson 10 table, a third-pot bluff only needs to work 25% of the time, the caller has missed often, and even your weakest hands have some equity.</p><p>Betting your whole range the same way also hides your hand. If you bet big only when you are strong, observant players will simply fold.</p><h3>Low or connected flops: check a lot</h3><p>On 7 6 5 you have neither advantage. Check most hands. Bet only your strong hands and good draws, and bet bigger, because those hands need protection from the many draws out there.</p><h3>Wet, high flops: split your range</h3><p>On Q J 9 with two of a suit, both ranges connect. Bet big with strong hands and good draws, check the medium hands such as a weak pair of jacks.</p>',
      },
      {
        t: "Three adjustments",
        h: () =>
          '<ul><li><b>Out of position, check more.</b> When you must act first on every later street, a bet that gets called puts you in a hard spot. Solvers c-bet noticeably less often out of position.</li><li><b>Against several opponents, tighten up.</b> With two or more callers, somebody has usually connected. Bet your good hands and give up most bluffs. The small whole-range bet is a heads-up play.</li><li><b>Against players who never fold, drop the bluffs.</b> The small bet with weak hands relies on folds. Against a Station, bet when you have something and check when you do not.</li></ul><p>In Trainer mode, when you were the preflop raiser and one opponent called, the coach now follows this plan on the flop and says which of the three boards you are on. Later streets still go by your equity.</p><div class="key">Turn and river play in strong games continues the same logic: who still has the strong hands after the action so far, and therefore who is allowed to bet big. That is beyond this tutorial, and it is the natural next thing to study with a solver-based trainer.</div>',
      },
      { drill: "cbet", need: 5, t: "Drill: pick the flop plan" },
    ],
  },
  {
    id: "types",
    title: "Opponents and the checklist",
    blurb: "Five player types, how to adjust to each, and the routine to run on every decision.",
    pages: [
      {
        t: "Two numbers that describe a player",
        h: () =>
          "<p>Two statistics describe how someone plays before the flop.</p><ul><li>" +
          term("VPIP") +
          ": the share of hands they choose to put chips in with. It measures how <b>loose</b> they are.</li><li>" +
          term("PFR") +
          ': the share of hands they raise before the flop. It measures how <b>aggressive</b> they are.</li></ul><p>The five bots at this table are one of each type below, but their seats do not say which. Each seat shows the two numbers as observed so far, for example 55/45. In Trainer mode the coach adds its guess at the type once it has seen about 15 hands. In Coaching mode you only get the numbers.</p><table class="ttable"><tr><th>Type</th><th>VPIP / PFR</th><th>What to do</th></tr><tr><td><b>Nit</b></td><td>14 / 11</td><td>Believe their bets. Take their blinds.</td></tr><tr><td><b>TAG</b></td><td>22 / 18</td><td>Solid. Give them normal respect.</td></tr><tr><td><b>LAG</b></td><td>34 / 27</td><td>They bluff more, so call them down with weaker hands.</td></tr><tr><td><b>Station</b></td><td>45 / 9</td><td>Never bluff them. Bet your good hands big.</td></tr><tr><td><b>Maniac</b></td><td>55 / 45</td><td>Let them bet into your strong hands.</td></tr></table><p>Real opponents do not wear labels. You work out their type by watching how often they enter pots and what they show at the end. The Calibration section after each showdown in Trainer and Coaching mode is practice for exactly that.</p>',
      },
      {
        t: "The checklist",
        h: () =>
          "<p>Everything in this tutorial folds into one routine. Run it on every decision until it becomes automatic.</p><ol><li><b>Ranges.</b> Who has shown strength? What seat and what type are they? What hands does that leave?</li><li><b>My hand.</b> What do I have on this board, in the proper vocabulary? Roughly what is my equity against that range?</li><li><b>Price.</b> If I face a bet, what equity does a call need? Call over pot plus call.</li><li><b>Plan.</b> If I raised before the flop, whose board is this? Am I betting for value, bluffing, or checking? If I bet, which worse hands call and which better hands fold?</li><li><b>Size.</b> What size does that job best?</li></ol><h3>Where to go from here</h3><p><b>Trainer mode</b> shows all of these numbers live: your equity, the price, every opponent\u2019s range, the EV of each button, and the coach\u2019s pick. Use it to build intuition. Guess the number first, then look.</p><p><b>Coaching mode</b> hides everything. You decide on your own, and the coach then reviews the decision in full. Your score there is the honest one.</p>",
      },
      { drill: "vsType", need: 4, t: "Drill: adjusting to opponents" },
    ],
  },
];

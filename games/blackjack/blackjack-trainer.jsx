import { useState, useEffect, useRef, useMemo, useCallback } from "react";

/* ----------------------------------------------------------------------
   Card and shoe primitives
---------------------------------------------------------------------- */
const SUITS = ["♠", "♥", "♦", "♣"];
const RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];
const NUM_DECKS = 6;
const PENETRATION = 0.75;
const START_BANKROLL = 1000;
const CHIPS = [5, 25, 100, 500];

function cardValue(r) {
  if (r === "A") return 11;
  if (r === "10" || r === "J" || r === "Q" || r === "K") return 10;
  return Number(r);
}
function tenRank(r) {
  return cardValue(r) === 10 ? "10" : r;
}
function hiLo(r) {
  const v = cardValue(r);
  if (v >= 2 && v <= 6) return 1;
  if (v >= 10) return -1;
  return 0;
}
function handValue(cards) {
  let total = 0;
  let aces = 0;
  for (const c of cards) {
    total += cardValue(c.r);
    if (c.r === "A") aces += 1;
  }
  while (total > 21 && aces > 0) {
    total -= 10;
    aces -= 1;
  }
  return { total, soft: aces > 0 };
}
function isBlackjack(cards) {
  return cards.length === 2 && handValue(cards).total === 21;
}
function buildShoe() {
  const shoe = [];
  let id = 0;
  for (let d = 0; d < NUM_DECKS; d++) {
    for (const s of SUITS) {
      for (const r of RANKS) shoe.push({ r, s, id: id++ });
    }
  }
  for (let i = shoe.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shoe[i], shoe[j]] = [shoe[j], shoe[i]];
  }
  return shoe;
}

/* ----------------------------------------------------------------------
   Expected value engine (infinite deck approximation)
   Every EV is expressed as a fraction of the original bet.
---------------------------------------------------------------------- */
const EV_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"];
const EV_P = { A: 1 / 13, 2: 1 / 13, 3: 1 / 13, 4: 1 / 13, 5: 1 / 13, 6: 1 / 13, 7: 1 / 13, 8: 1 / 13, 9: 1 / 13, 10: 4 / 13 };

function addRank(t, s, r) {
  let nt, ns;
  if (r === "A") {
    if (t + 11 <= 21) {
      nt = t + 11;
      ns = true;
    } else {
      nt = t + 1;
      ns = s;
    }
  } else {
    nt = t + cardValue(r);
    ns = s;
  }
  if (nt > 21 && ns) {
    nt -= 10;
    ns = false;
  }
  return [nt, ns];
}

function dealerDist(up, h17) {
  const memo = {};
  const rec = (t, s) => {
    if (t > 21) return [0, 0, 0, 0, 0, 1];
    if (t > 17 || (t === 17 && !(h17 && s))) {
      const d = [0, 0, 0, 0, 0, 0];
      d[t - 17] = 1;
      return d;
    }
    const key = t + (s ? "s" : "h");
    if (memo[key]) return memo[key];
    const d = [0, 0, 0, 0, 0, 0];
    for (const r of EV_RANKS) {
      const [nt, ns] = addRank(t, s, r);
      const sub = rec(nt, ns);
      for (let i = 0; i < 6; i++) d[i] += EV_P[r] * sub[i];
    }
    memo[key] = d;
    return d;
  };
  const [t0, s0] = addRank(0, false, up);
  const blocked = (r) => (up === "A" && r === "10") || (up === "10" && r === "A");
  let norm = 0;
  for (const r of EV_RANKS) if (!blocked(r)) norm += EV_P[r];
  const d = [0, 0, 0, 0, 0, 0];
  for (const r of EV_RANKS) {
    if (blocked(r)) continue;
    const [nt, ns] = addRank(t0, s0, r);
    const sub = rec(nt, ns);
    for (let i = 0; i < 6; i++) d[i] += (EV_P[r] / norm) * sub[i];
  }
  return d; // probabilities of dealer ending on 17, 18, 19, 20, 21, bust
}

const engineCache = new Map();
function getEngine(up, rules) {
  const key = `${up}|${rules.h17 ? 1 : 0}|${rules.das ? 1 : 0}`;
  if (engineCache.has(key)) return engineCache.get(key);
  const dd = dealerDist(up, rules.h17);
  const stand = (t) => {
    if (t > 21) return -1;
    let ev = dd[5];
    for (let i = 0; i < 5; i++) {
      const d = 17 + i;
      if (d < t) ev += dd[i];
      else if (d > t) ev -= dd[i];
    }
    return ev;
  };
  const hitMemo = {};
  const hit = (t, s) => {
    const k = t + (s ? "s" : "h");
    if (hitMemo[k] !== undefined) return hitMemo[k];
    let ev = 0;
    for (const r of EV_RANKS) {
      const [nt, ns] = addRank(t, s, r);
      ev += EV_P[r] * (nt > 21 ? -1 : Math.max(stand(nt), hit(nt, ns)));
    }
    hitMemo[k] = ev;
    return ev;
  };
  const dbl = (t, s) => {
    let ev = 0;
    for (const r of EV_RANKS) {
      const [nt] = addRank(t, s, r);
      ev += EV_P[r] * (nt > 21 ? -1 : stand(nt));
    }
    return 2 * ev;
  };
  const split = (r) => {
    let ev = 0;
    if (r === "A") {
      for (const c of EV_RANKS) {
        const [nt] = addRank(11, true, c);
        ev += EV_P[c] * stand(nt);
      }
      return 2 * ev;
    }
    const [t0, s0] = addRank(0, false, r);
    for (const c of EV_RANKS) {
      const [nt, ns] = addRank(t0, s0, c);
      let best = Math.max(stand(nt), hit(nt, ns));
      if (rules.das) best = Math.max(best, dbl(nt, ns));
      ev += EV_P[c] * best;
    }
    return 2 * ev;
  };
  const bustIfHit = (t, s) => {
    if (s) return 0;
    let p = 0;
    for (const r of EV_RANKS) {
      const [nt] = addRank(t, s, r);
      if (nt > 21) p += EV_P[r];
    }
    return p;
  };
  const eng = { dd, stand, hit, dbl, split, bustIfHit };
  engineCache.set(key, eng);
  return eng;
}

const ACTION_LABEL = { stand: "Stand", hit: "Hit", double: "Double", split: "Split", surrender: "Surrender" };

function analyze(cards, upRank, rules, opts) {
  const { total, soft } = handValue(cards);
  const eng = getEngine(tenRank(upRank), rules);
  const acts = [
    { a: "stand", ev: eng.stand(total) },
    { a: "hit", ev: eng.hit(total, soft) },
  ];
  if (opts.canDouble) acts.push({ a: "double", ev: eng.dbl(total, soft) });
  if (opts.canSplit) acts.push({ a: "split", ev: eng.split(tenRank(cards[0].r)) });
  if (opts.canSurrender) acts.push({ a: "surrender", ev: -0.5 });
  acts.sort((x, y) => y.ev - x.ev);
  return { total, soft, acts, best: acts[0], dd: eng.dd, bustIfHit: eng.bustIfHit(total, soft), isPair: opts.canSplit };
}

const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const signedPct = (x, d = 1) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(d)}%`;
const money = (x) => `$${Math.abs(x).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;

function describeHand(an, cards) {
  if (an.isPair) return `a pair of ${tenRank(cards[0].r) === "10" ? "tens" : cards[0].r === "A" ? "aces" : cards[0].r + "s"}`;
  return `${an.soft ? "soft" : "hard"} ${an.total}`;
}

function explain(an, cards, upRank) {
  const up = tenRank(upRank);
  const upName = up === "A" ? "an ace" : `a ${up}`;
  const bust = an.dd[5];
  const strongUp = up === "A" || up === "10" || up === "9";
  const weakUp = ["4", "5", "6"].includes(up);
  const evOf = (a) => an.acts.find((x) => x.a === a)?.ev;
  const best = an.best.a;
  const second = an.acts[1];
  const gap = an.best.ev - second.ev;
  const paras = [];

  paras.push(
    `Dealer shows ${upName}. From that upcard the dealer busts ${pct(bust)} of the time and finishes on 17 or more ${pct(1 - bust)} of the time` +
      (weakUp ? ". This is a weak upcard: the dealer has to draw from a stiff total and often breaks." : strongUp ? ". This is a strong upcard: the dealer usually makes a pat hand, so you have to make one too." : ".")
  );

  if (best === "stand") {
    if (an.total >= 17) {
      paras.push(`You hold ${describeHand(an, cards)}. Standing on 17 or higher is close to automatic: hitting would bust you ${pct(an.bustIfHit)} of the time, and the few cards that help are outweighed by the ones that break you.`);
    } else if (an.soft) {
      paras.push(`You hold ${describeHand(an, cards)}. A soft ${an.total} against this upcard already beats the dealer more often than not; drawing risks turning a good hand into a mediocre one.`);
    } else {
      paras.push(`You hold ${describeHand(an, cards)}, a stiff hand: it cannot win on its own and hitting busts you ${pct(an.bustIfHit)} of the time. Against a weak upcard you let the dealer take the bust risk instead. Standing here wins whenever the dealer breaks (${pct(bust)}).`);
    }
  } else if (best === "hit") {
    if (an.soft) {
      paras.push(`You hold ${describeHand(an, cards)}. A soft hand cannot bust on one card, so drawing is free improvement. Your current total is not good enough to stand on against ${upName}.`);
    } else if (an.total <= 11) {
      paras.push(`You hold ${describeHand(an, cards)}. No single card can bust you, so you always take another. Doubling was not better here${evOf("double") === undefined ? " (not available on this hand)" : ` (double EV ${signedPct(evOf("double"))} vs hit ${signedPct(evOf("hit"))})`}.`);
    } else {
      paras.push(`You hold ${describeHand(an, cards)}. Hitting busts you ${pct(an.bustIfHit)} of the time, which feels bad, but standing only wins when the dealer busts (${pct(bust)}). Against a strong upcard the dealer makes a hand too often, so you must take the risk to reach a competitive total.`);
    }
  } else if (best === "double") {
    paras.push(`You hold ${describeHand(an, cards)}. Doubling means you put a second bet down and receive exactly one more card. It is right when a single card lands you on a strong total often enough that doubling your stake beats keeping the option to hit again. Here the extra money is working with an edge: doubling earns ${signedPct(evOf("double"))} of your bet versus ${signedPct(evOf("hit"))} for hitting.`);
  } else if (best === "split") {
    const r = tenRank(cards[0].r);
    if (r === "A") paras.push(`Always split aces. As one hand they are a soft 12; as two hands each starts with the most valuable card in the deck and has a ${pct(4 / 13)} chance of drawing a ten for 21 on the next card.`);
    else if (r === "8") paras.push(`Always split eights. Together they make 16, the worst total in the game. Apart, each hand starts from 8 and can build toward 18. Even against a strong upcard, two 8s lose less than one 16 (split EV ${signedPct(evOf("split"))} vs hit ${signedPct(evOf("hit"))}).`);
    else paras.push(`Splitting these ${r}s is right against ${upName}. Two hands starting from ${r} beat one hand of ${an.total}, especially when the dealer is likely to bust and you can double after the split.`);
  } else if (best === "surrender") {
    paras.push(`You hold ${describeHand(an, cards)} against ${upName}. Surrender gives up half your bet immediately. It is correct only when playing on loses more than half a bet on average, which means you would win under 25% of the time. Here the best playing option earns ${signedPct(second.ev)}, worse than the flat -50% of surrendering.`);
  }

  if (best !== "stand" && an.total >= 17 && !an.soft) {
    // nothing extra
  }
  paras.push(
    gap < 0.01
      ? `The margin between ${ACTION_LABEL[best].toLowerCase()} and ${ACTION_LABEL[second.a].toLowerCase()} is only ${pct(gap, 2)} of your bet, so this is a coin flip in practice. Either choice is defensible; basic strategy just picks the marginal winner.`
      : gap < 0.05
      ? `${ACTION_LABEL[best]} beats ${ACTION_LABEL[second.a].toLowerCase()} by ${pct(gap, 1)} of your bet. Real money, but a small mistake.`
      : `${ACTION_LABEL[best]} beats ${ACTION_LABEL[second.a].toLowerCase()} by ${pct(gap, 1)} of your bet. Getting this one wrong is expensive.`
  );
  return paras;
}

/* ----------------------------------------------------------------------
   Visual components
---------------------------------------------------------------------- */
function PlayingCard({ card, hidden, index = 0 }) {
  const red = card && (card.s === "♥" || card.s === "♦");
  return (
    <div className={`bjcard ${hidden ? "back" : ""} ${red ? "red" : ""}`} style={{ animationDelay: `${index * 60}ms` }} aria-label={hidden ? "face-down card" : `${card.r} of ${card.s}`}>
      {!hidden && (
        <>
          <div className="corner tl">
            <span>{card.r}</span>
            <span>{card.s}</span>
          </div>
          <div className="pip">{card.s}</div>
          <div className="corner br">
            <span>{card.r}</span>
            <span>{card.s}</span>
          </div>
        </>
      )}
    </div>
  );
}

function EVBar({ acts, best, chosen }) {
  const min = Math.min(-0.6, ...acts.map((a) => a.ev));
  const max = Math.max(0.6, ...acts.map((a) => a.ev));
  const scale = (v) => ((v - min) / (max - min)) * 100;
  const zero = scale(0);
  return (
    <div className="evbars">
      {acts.map((a) => {
        const s = scale(a.ev);
        const left = Math.min(s, zero);
        const width = Math.abs(s - zero);
        return (
          <div className={`evrow ${a.a === best ? "best" : ""} ${a.a === chosen ? "chosen" : ""}`} key={a.a}>
            <div className="evlabel">{ACTION_LABEL[a.a]}</div>
            <div className="evtrack">
              <div className="evzero" style={{ left: `${zero}%` }} />
              <div className={`evfill ${a.ev >= 0 ? "pos" : "neg"}`} style={{ left: `${left}%`, width: `${width}%` }} />
            </div>
            <div className="evnum">{signedPct(a.ev)}</div>
          </div>
        );
      })}
    </div>
  );
}

/* ----------------------------------------------------------------------
   Main component
---------------------------------------------------------------------- */
export default function BlackjackTrainer() {
  const shoeRef = useRef(buildShoe());
  const [cardsLeft, setCardsLeft] = useState(shoeRef.current.length);
  const [runningCount, setRunningCount] = useState(0);
  const [needsShuffle, setNeedsShuffle] = useState(false);

  const [bankroll, setBankroll] = useState(START_BANKROLL);
  const [bet, setBet] = useState(25);
  const [phase, setPhase] = useState("bet"); // bet | insurance | play | dealer | settled
  const [dealer, setDealer] = useState({ cards: [], hole: true });
  const [hands, setHands] = useState([]);
  const [active, setActive] = useState(0);
  const [insurance, setInsurance] = useState(0);
  const [results, setResults] = useState([]);
  const [feedback, setFeedback] = useState(null);
  const [tab, setTab] = useState("coach");
  const [coachMode, setCoachMode] = useState("live"); // live | after | off
  const [rules, setRules] = useState({ h17: false, das: true, surrender: true, bj65: false });
  const [showCount, setShowCount] = useState(false);
  const [stats, setStats] = useState({ hands: 0, wins: 0, losses: 0, pushes: 0, blackjacks: 0, decisions: 0, correct: 0, evLost: 0, net: 0 });

  /* ---------- shoe helpers ---------- */
  const draw = useCallback((countIt = true) => {
    if (shoeRef.current.length === 0) shoeRef.current = buildShoe();
    const c = shoeRef.current.pop();
    setCardsLeft(shoeRef.current.length);
    if (countIt) setRunningCount((rc) => rc + hiLo(c.r));
    return c;
  }, []);

  const decksLeft = Math.max(cardsLeft / 52, 0.5);
  const trueCount = runningCount / decksLeft;

  /* ---------- derived ---------- */
  const upRank = dealer.cards[0]?.r;
  const hand = hands[active];
  const canAct = phase === "play" && hand && !hand.done;
  const opts = useMemo(() => {
    if (!hand) return {};
    const two = hand.cards.length === 2;
    const afford = (x) => bankroll >= x;
    return {
      canDouble: two && !hand.splitAces && afford(hand.bet) && (!hand.fromSplit || rules.das),
      canSplit: two && !hand.splitAces && hands.length < 4 && tenRank(hand.cards[0].r) === tenRank(hand.cards[1].r) && afford(hand.bet) && !(hand.fromSplit && hand.cards[0].r === "A"),
      canSurrender: two && !hand.fromSplit && rules.surrender,
    };
  }, [hand, hands.length, bankroll, rules]);

  const analysis = useMemo(() => {
    if (!hand || !upRank || phase !== "play") return null;
    return analyze(hand.cards, upRank, rules, opts);
  }, [hand, upRank, rules, opts, phase]);

  /* ---------- game flow ---------- */
  function startHand() {
    if (bet > bankroll || bet <= 0) return;
    if (needsShuffle) {
      shoeRef.current = buildShoe();
      setCardsLeft(shoeRef.current.length);
      setRunningCount(0);
      setNeedsShuffle(false);
    }
    setFeedback(null);
    setResults([]);
    setInsurance(0);
    setBankroll((b) => b - bet);
    const p1 = draw();
    const d1 = draw();
    const p2 = draw();
    const hole = draw(false);
    const h = [{ cards: [p1, p2], bet, done: false, doubled: false, surrendered: false, fromSplit: false, splitAces: false }];
    setHands(h);
    setActive(0);
    setDealer({ cards: [d1, hole], hole: true });
    if (d1.r === "A" && bankroll - bet >= bet / 2) {
      setPhase("insurance");
    } else {
      afterPeek(h, [d1, hole], 0);
    }
  }

  function afterPeek(h, dcards, ins) {
    const dealerBJ = isBlackjack(dcards);
    const playerBJ = isBlackjack(h[0].cards);
    if (dealerBJ) {
      setRunningCount((rc) => rc + hiLo(dcards[1].r));
      setDealer({ cards: dcards, hole: false });
      settle(h, dcards, ins, true);
      return;
    }
    if (playerBJ) {
      setRunningCount((rc) => rc + hiLo(dcards[1].r));
      setDealer({ cards: dcards, hole: false });
      settle(h, dcards, ins, false);
      return;
    }
    setPhase("play");
  }

  function decideInsurance(take) {
    const ins = take ? bet / 2 : 0;
    if (take) setBankroll((b) => b - ins);
    setInsurance(ins);
    const p = tenDensity();
    const ev = 2 * p - (1 - p);
    const right = ev > 0 ? take : !take;
    setFeedback({
      ok: right,
      title: right ? "Good insurance decision" : "Insurance mistake",
      body: `Unseen cards are ${pct(p)} tens, so insurance pays ${signedPct(ev)} of the insurance bet. ${ev > 0 ? "With this many tens left, insurance is a positive bet." : "It only becomes profitable when more than a third of unseen cards are tens (true count around +3)."}`,
    });
    afterPeek(hands, dealer.cards, ins);
  }

  function tenDensity() {
    // unseen cards: the shoe plus the dealer's hole card
    const unseen = [...shoeRef.current, dealer.cards[1]].filter(Boolean);
    const tens = unseen.filter((c) => cardValue(c.r) === 10).length;
    return tens / unseen.length;
  }

  function recordDecision(action) {
    if (!analysis) return;
    const best = analysis.best;
    const chosen = analysis.acts.find((x) => x.a === action);
    const gap = best.ev - (chosen ? chosen.ev : best.ev);
    const ok = action === best.a || gap < 0.0005;
    setStats((s) => ({ ...s, decisions: s.decisions + 1, correct: s.correct + (ok ? 1 : 0), evLost: s.evLost + gap * hand.bet }));
    setFeedback({
      ok,
      title: ok ? `${ACTION_LABEL[action]} on ${describeHand(analysis, hand.cards)} vs ${tenRank(upRank)}: correct` : `You chose ${ACTION_LABEL[action].toLowerCase()}; the better play was ${ACTION_LABEL[best.a].toLowerCase()}`,
      body: ok ? `Expected value ${signedPct(best.ev)} of your bet.` : `That choice gives up ${pct(gap)} of your bet in expectation (about ${money(gap * hand.bet)} on this hand). ${explain(analysis, hand.cards, upRank).slice(1, 2).join(" ")}`,
      acts: analysis.acts,
      best: best.a,
      chosen: action,
    });
  }

  function advance(nextHands) {
    let i = active;
    // find next unfinished hand
    let n = nextHands.findIndex((h, idx) => idx > i && !h.done);
    if (n === -1) n = nextHands.findIndex((h) => !h.done);
    if (n === -1) {
      setHands(nextHands);
      setPhase("dealer");
      return;
    }
    // deal second card to a freshly split hand if it has only one card
    const h = nextHands[n];
    if (h.cards.length === 1) {
      const c = draw();
      h.cards = [...h.cards, c];
      if (h.splitAces) h.done = true;
      else if (handValue(h.cards).total === 21) h.done = true;
    }
    setHands([...nextHands]);
    setActive(n);
    if (h.done) advance([...nextHands]);
  }

  function doHit() {
    if (!canAct) return;
    recordDecision("hit");
    const next = hands.map((h) => ({ ...h, cards: [...h.cards] }));
    const h = next[active];
    h.cards.push(draw());
    const v = handValue(h.cards).total;
    if (v >= 21) h.done = true;
    if (v > 21) h.busted = true;
    if (h.done) advance(next);
    else setHands(next);
  }
  function doStand() {
    if (!canAct) return;
    recordDecision("stand");
    const next = hands.map((h) => ({ ...h, cards: [...h.cards] }));
    next[active].done = true;
    advance(next);
  }
  function doDouble() {
    if (!canAct || !opts.canDouble) return;
    recordDecision("double");
    setBankroll((b) => b - hand.bet);
    const next = hands.map((h) => ({ ...h, cards: [...h.cards] }));
    const h = next[active];
    h.bet *= 2;
    h.doubled = true;
    h.cards.push(draw());
    h.done = true;
    if (handValue(h.cards).total > 21) h.busted = true;
    advance(next);
  }
  function doSplit() {
    if (!canAct || !opts.canSplit) return;
    recordDecision("split");
    setBankroll((b) => b - hand.bet);
    const next = hands.map((h) => ({ ...h, cards: [...h.cards] }));
    const h = next[active];
    const aces = h.cards[0].r === "A";
    const h2 = { cards: [h.cards[1]], bet: h.bet, done: false, doubled: false, surrendered: false, fromSplit: true, splitAces: aces };
    h.cards = [h.cards[0]];
    h.fromSplit = true;
    h.splitAces = aces;
    next.splice(active + 1, 0, h2);
    const c = draw();
    h.cards.push(c);
    if (aces || handValue(h.cards).total === 21) h.done = true;
    setHands(next);
    if (h.done) advance(next);
  }
  function doSurrender() {
    if (!canAct || !opts.canSurrender) return;
    recordDecision("surrender");
    const next = hands.map((h) => ({ ...h, cards: [...h.cards] }));
    next[active].surrendered = true;
    next[active].done = true;
    advance(next);
  }

  // dealer turn, stepped with a short delay so the draw is readable
  useEffect(() => {
    if (phase !== "dealer") return;
    const t = setTimeout(() => {
      if (dealer.hole) {
        setRunningCount((rc) => rc + hiLo(dealer.cards[1].r));
        setDealer((d) => ({ ...d, hole: false }));
        return;
      }
      const live = hands.some((h) => !h.busted && !h.surrendered);
      const { total, soft } = handValue(dealer.cards);
      const mustHit = live && (total < 17 || (total === 17 && soft && rules.h17));
      if (mustHit) {
        const c = draw();
        setDealer((d) => ({ ...d, cards: [...d.cards, c] }));
      } else {
        settle(hands, dealer.cards, insurance, false);
      }
    }, 550);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, dealer]);

  function settle(h, dcards, ins, dealerBJ) {
    const dv = handValue(dcards).total;
    const dBust = dv > 21;
    let payout = 0;
    let net = 0;
    const res = [];
    let wins = 0, losses = 0, pushes = 0, bjs = 0;
    for (const hd of h) {
      const pv = handValue(hd.cards).total;
      const pBJ = isBlackjack(hd.cards) && !hd.fromSplit;
      let r;
      if (hd.surrendered) {
        payout += hd.bet / 2;
        r = { text: "Surrendered", delta: -hd.bet / 2, kind: "loss" };
        losses++;
      } else if (dealerBJ) {
        if (pBJ) {
          payout += hd.bet;
          r = { text: "Both blackjack, push", delta: 0, kind: "push" };
          pushes++;
        } else {
          r = { text: "Dealer blackjack", delta: -hd.bet, kind: "loss" };
          losses++;
        }
      } else if (pBJ) {
        const mult = rules.bj65 ? 1.2 : 1.5;
        payout += hd.bet * (1 + mult);
        r = { text: `Blackjack, pays ${rules.bj65 ? "6:5" : "3:2"}`, delta: hd.bet * mult, kind: "win" };
        wins++;
        bjs++;
      } else if (pv > 21) {
        r = { text: `Bust with ${pv}`, delta: -hd.bet, kind: "loss" };
        losses++;
      } else if (dBust) {
        payout += hd.bet * 2;
        r = { text: `Dealer busts, ${pv} wins`, delta: hd.bet, kind: "win" };
        wins++;
      } else if (pv > dv) {
        payout += hd.bet * 2;
        r = { text: `${pv} beats ${dv}`, delta: hd.bet, kind: "win" };
        wins++;
      } else if (pv === dv) {
        payout += hd.bet;
        r = { text: `Push at ${pv}`, delta: 0, kind: "push" };
        pushes++;
      } else {
        r = { text: `${pv} loses to ${dv}`, delta: -hd.bet, kind: "loss" };
        losses++;
      }
      net += r.delta;
      res.push(r);
    }
    if (ins > 0) {
      if (dealerBJ) {
        payout += ins * 3;
        net += ins * 2;
        res.push({ text: "Insurance pays 2:1", delta: ins * 2, kind: "win" });
      } else {
        net -= ins;
        res.push({ text: "Insurance lost", delta: -ins, kind: "loss" });
      }
    }
    setBankroll((b) => b + payout);
    setResults(res);
    setStats((s) => ({ ...s, hands: s.hands + h.length, wins: s.wins + wins, losses: s.losses + losses, pushes: s.pushes + pushes, blackjacks: s.blackjacks + bjs, net: s.net + net }));
    setHands(h.map((x) => ({ ...x, done: true })));
    setPhase("settled");
    if (shoeRef.current.length < NUM_DECKS * 52 * (1 - PENETRATION)) setNeedsShuffle(true);
  }

  function nextHand() {
    setPhase("bet");
    setHands([]);
    setDealer({ cards: [], hole: true });
    setResults([]);
    setFeedback(null);
  }
  function resetBankroll() {
    setBankroll(START_BANKROLL);
    setStats({ hands: 0, wins: 0, losses: 0, pushes: 0, blackjacks: 0, decisions: 0, correct: 0, evLost: 0, net: 0 });
    nextHand();
  }

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (e) => {
      if (e.target && ["INPUT", "TEXTAREA"].includes(e.target.tagName)) return;
      const k = e.key.toLowerCase();
      if (phase === "play") {
        if (k === "h") doHit();
        else if (k === "s") doStand();
        else if (k === "d") doDouble();
        else if (k === "p") doSplit();
        else if (k === "r") doSurrender();
      } else if (phase === "bet" && (k === " " || k === "enter")) {
        e.preventDefault();
        startHand();
      } else if (phase === "settled" && (k === " " || k === "enter")) {
        e.preventDefault();
        nextHand();
      } else if (phase === "insurance") {
        if (k === "y") decideInsurance(true);
        if (k === "n") decideInsurance(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  /* ---------- strategy chart (derived from the same engine) ---------- */
  const chart = useMemo(() => {
    const ups = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"];
    const code = (eng, t, s, canSur) => {
      const st = eng.stand(t), hi = eng.hit(t, s), db = eng.dbl(t, s);
      let best = "H", bestEv = hi;
      if (st > bestEv) { best = "S"; bestEv = st; }
      if (db > bestEv) { best = st > hi ? "Ds" : "D"; bestEv = db; }
      if (canSur && -0.5 > bestEv) best = "R";
      return best;
    };
    const hard = [];
    for (let t = 20; t >= 5; t--) hard.push({ label: String(t), cells: ups.map((u) => code(getEngine(u, rules), t, false, rules.surrender)) });
    const soft = [];
    for (let t = 20; t >= 13; t--) soft.push({ label: `A,${t - 11}`, cells: ups.map((u) => code(getEngine(u, rules), t, true, false)) });
    const pairs = [];
    for (const r of ["A", "10", "9", "8", "7", "6", "5", "4", "3", "2"]) {
      const [t, s] = r === "A" ? [12, true] : [cardValue(r) * 2, false];
      pairs.push({
        label: `${r},${r}`,
        cells: ups.map((u) => {
          const eng = getEngine(u, rules);
          const base = code(eng, t, s, rules.surrender);
          const baseEv = Math.max(eng.stand(t), eng.hit(t, s), eng.dbl(t, s), rules.surrender ? -0.5 : -1);
          return eng.split(r) > baseEv ? "P" : base;
        }),
      });
    }
    return { ups, hard, soft, pairs };
  }, [rules]);

  const houseEdge = useMemo(() => {
    let e = 0.34; // 6 decks, S17, DAS, no surrender, 3:2, roughly
    if (rules.h17) e += 0.22;
    if (!rules.das) e += 0.14;
    if (rules.surrender) e -= 0.07;
    if (rules.bj65) e += 1.39;
    return e;
  }, [rules]);

  const dealerTable = useMemo(() => ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"].map((u) => ({ u, d: dealerDist(u, rules.h17) })), [rules.h17]);

  /* ---------- rendering ---------- */
  const dv = dealer.cards.length ? handValue(dealer.hole ? [dealer.cards[0]] : dealer.cards) : null;
  const showLiveCoach = coachMode === "live" && phase === "play" && analysis;
  const canBet = (x) => bet + x <= bankroll;

  return (
    <div className="bj">
      <style>{CSS}</style>

      <header className="masthead">
        <div>
          <h1>Blackjack, with the odds showing</h1>
          <p className="sub">Six decks, dealer {rules.h17 ? "hits" : "stands on"} soft 17, blackjack pays {rules.bj65 ? "6:5" : "3:2"}. Keys: H hit, S stand, D double, P split, R surrender, Enter deal.</p>
        </div>
        <div className="bank">
          <div className="bankval">{money(bankroll)}</div>
          <div className="banklabel">bankroll, session {stats.net >= 0 ? "+" : "-"}{money(stats.net)}</div>
        </div>
      </header>

      <div className="layout">
        {/* ------------------------- TABLE ------------------------- */}
        <section className="table" aria-label="Blackjack table">
          <div className="felt">
            <div className="seat dealerseat">
              <div className="seatname">
                Dealer {dv && <span className="total">{dealer.hole ? `shows ${dv.total}` : `${dv.soft && dv.total !== 21 ? "soft " : ""}${dv.total}${dv.total > 21 ? ", bust" : ""}`}</span>}
              </div>
              <div className="cards">
                {dealer.cards.map((c, i) => (
                  <PlayingCard key={c.id} card={c} hidden={i === 1 && dealer.hole} index={i} />
                ))}
                {dealer.cards.length === 0 && <div className="slot" />}
              </div>
            </div>

            <div className="shoeinfo">
              <span>{cardsLeft} cards left in the shoe{needsShuffle ? ", reshuffling next hand" : ""}</span>
              {showCount && (
                <span className="countchip">
                  running {runningCount >= 0 ? "+" : ""}{runningCount}, true {trueCount >= 0 ? "+" : ""}{trueCount.toFixed(1)}
                </span>
              )}
            </div>

            <div className="seat playerseat">
              <div className="hands">
                {hands.length === 0 && (
                  <div className="empty">
                    <div className="slot" />
                    <div className="slot" />
                  </div>
                )}
                {hands.map((h, i) => {
                  const v = handValue(h.cards);
                  const r = results[i];
                  return (
                    <div key={i} className={`hand ${i === active && phase === "play" ? "activehand" : ""} ${r ? r.kind : ""}`}>
                      <div className="cards">
                        {h.cards.map((c, j) => (
                          <PlayingCard key={c.id} card={c} index={j} />
                        ))}
                      </div>
                      <div className="handmeta">
                        <span className="total">
                          {h.surrendered ? "surrendered" : `${v.soft && v.total !== 21 ? "soft " : ""}${v.total}${v.total > 21 ? ", bust" : isBlackjack(h.cards) && !h.fromSplit ? ", blackjack" : ""}`}
                        </span>
                        <span className="betlabel">{money(h.bet)}{h.doubled ? " doubled" : ""}</span>
                        {r && <span className={`result ${r.kind}`}>{r.text}{r.delta !== 0 ? ` (${r.delta > 0 ? "+" : "-"}${money(r.delta)})` : ""}</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
              {results.filter((r) => r.text.startsWith("Insurance")).map((r, i) => (
                <div key={i} className={`result ${r.kind} insres`}>{r.text} ({r.delta > 0 ? "+" : "-"}{money(r.delta)})</div>
              ))}
            </div>
          </div>

          {/* controls */}
          <div className="controls">
            {phase === "bet" && (
              <div className="betrow">
                <div className="chips">
                  {CHIPS.map((c) => (
                    <button key={c} className={`chip c${c}`} onClick={() => canBet(c) && setBet(bet + c)} disabled={!canBet(c)} aria-label={`add ${c} to bet`}>
                      {c}
                    </button>
                  ))}
                  <button className="ghost" onClick={() => setBet(0)}>Clear</button>
                </div>
                <div className="betstate">Bet {money(bet)}</div>
                <button className="primary" onClick={startHand} disabled={bet <= 0 || bet > bankroll}>Deal</button>
                {bankroll < 5 && <button className="ghost" onClick={resetBankroll}>Reset bankroll</button>}
              </div>
            )}
            {phase === "insurance" && (
              <div className="betrow">
                <div className="prompt">Dealer shows an ace. Insurance costs {money(bet / 2)} and pays 2:1 if the dealer has blackjack.</div>
                <button className="primary" onClick={() => decideInsurance(true)}>Take insurance (Y)</button>
                <button className="ghost" onClick={() => decideInsurance(false)}>No insurance (N)</button>
              </div>
            )}
            {phase === "play" && (
              <div className="actions">
                <button onClick={doHit} className={showLiveCoach && analysis.best.a === "hit" ? "rec" : ""}>Hit</button>
                <button onClick={doStand} className={showLiveCoach && analysis.best.a === "stand" ? "rec" : ""}>Stand</button>
                <button onClick={doDouble} disabled={!opts.canDouble} className={showLiveCoach && analysis.best.a === "double" ? "rec" : ""}>Double</button>
                <button onClick={doSplit} disabled={!opts.canSplit} className={showLiveCoach && analysis.best.a === "split" ? "rec" : ""}>Split</button>
                <button onClick={doSurrender} disabled={!opts.canSurrender} className={showLiveCoach && analysis.best.a === "surrender" ? "rec" : ""}>Surrender</button>
              </div>
            )}
            {phase === "dealer" && <div className="prompt">Dealer is playing…</div>}
            {phase === "settled" && (
              <div className="betrow">
                <button className="primary" onClick={nextHand}>Next hand (Enter)</button>
                <span className="prompt">Same bet of {money(Math.min(bet, bankroll))} stays unless you change it.</span>
              </div>
            )}
          </div>
        </section>

        {/* ------------------------- LEDGER ------------------------- */}
        <aside className="ledger">
          <nav className="tabs" aria-label="panels">
            {[["coach", "Coach"], ["odds", "Odds & payoffs"], ["chart", "Strategy chart"], ["count", "Counting"], ["stats", "Stats & rules"]].map(([id, label]) => (
              <button key={id} className={tab === id ? "on" : ""} onClick={() => setTab(id)}>{label}</button>
            ))}
          </nav>

          {tab === "coach" && (
            <div className="panel">
              <div className="modeswitch">
                <span>Coach speaks</span>
                {[["live", "before I act"], ["after", "after I act"], ["off", "never"]].map(([id, l]) => (
                  <button key={id} className={coachMode === id ? "on" : ""} onClick={() => setCoachMode(id)}>{l}</button>
                ))}
              </div>

              {phase === "bet" && !feedback && (
                <div className="prose">
                  <h2>How to use this table</h2>
                  <p>Place a bet and deal. While you play, this panel shows every action available to you with its expected value: the average amount you win or lose per unit bet if you made that choice a million times from this exact spot. Positive is good, negative is bad, and the gap between the best and second best option tells you how much a mistake costs.</p>
                  <p>Numbers come from a live calculation of the dealer's finishing totals given the upcard, not a memorised table. Switch the rules in the Stats & rules tab and the advice updates.</p>
                  <p>The house edge under the current rules is about {houseEdge.toFixed(2)}% of each bet with perfect play. Most players give the casino 1.5% to 2% because of avoidable mistakes. Your job at this table is to close that gap.</p>
                </div>
              )}

              {showLiveCoach && (
                <div className="prose">
                  <h2>
                    {describeHand(analysis, hand.cards)} against {tenRank(upRank) === "A" ? "an ace" : "a " + tenRank(upRank)}
                    {hands.length > 1 ? ` (hand ${active + 1} of ${hands.length})` : ""}
                  </h2>
                  <div className="verdict">{ACTION_LABEL[analysis.best.a]}</div>
                  <EVBar acts={analysis.acts} best={analysis.best.a} />
                  {explain(analysis, hand.cards, upRank).map((p, i) => (
                    <p key={i}>{p}</p>
                  ))}
                  <DealerOutcomes dd={analysis.dd} up={tenRank(upRank)} />
                </div>
              )}

              {coachMode === "after" && phase === "play" && analysis && (
                <div className="prose">
                  <h2>{describeHand(analysis, hand.cards)} against {tenRank(upRank) === "A" ? "an ace" : "a " + tenRank(upRank)}</h2>
                  <p>Make your move. The verdict appears after you act.</p>
                </div>
              )}

              {feedback && coachMode !== "off" && (
                <div className={`feedback ${feedback.ok ? "good" : "bad"}`}>
                  <h3>{feedback.title}</h3>
                  <p>{feedback.body}</p>
                  {feedback.acts && !showLiveCoach && <EVBar acts={feedback.acts} best={feedback.best} chosen={feedback.chosen} />}
                </div>
              )}

              {phase === "settled" && (
                <div className="prose">
                  <h2>Hand settled</h2>
                  <p>
                    {results.every((r) => r.kind === "win") ? "Every wager won." : results.every((r) => r.kind === "loss") ? "A losing hand. Remember that the right play still loses a lot: the goal is to lose the least on bad hands and win the most on good ones." : "Mixed result."}
                    {" "}Over {stats.hands} hands you have played {stats.decisions} decisions at {stats.decisions ? pct(stats.correct / stats.decisions, 0) : "0%"} accuracy, and mistakes have cost you roughly {money(stats.evLost)} in expectation.
                  </p>
                </div>
              )}
            </div>
          )}

          {tab === "odds" && (
            <div className="panel prose">
              <h2>Payoffs</h2>
              <table className="plain">
                <tbody>
                  <tr><td>Win</td><td>1 to 1</td><td>bet $25, receive $25</td></tr>
                  <tr><td>Blackjack (ace + ten on your first two cards)</td><td>{rules.bj65 ? "6 to 5" : "3 to 2"}</td><td>bet $25, receive {rules.bj65 ? "$30" : "$37.50"}</td></tr>
                  <tr><td>Push (tie)</td><td>bet returned</td><td></td></tr>
                  <tr><td>Double</td><td>1 to 1 on twice the bet</td><td>one card only</td></tr>
                  <tr><td>Split</td><td>each hand paid on its own</td><td>up to four hands</td></tr>
                  <tr><td>Surrender</td><td>lose half</td><td>first two cards only</td></tr>
                  <tr><td>Insurance</td><td>2 to 1 on a side bet of half your stake</td><td>offered when dealer shows an ace</td></tr>
                </tbody>
              </table>

              <h2>Where the house edge comes from</h2>
              <p>Both you and the dealer bust about 28% of the time when you play the same way. The house edge exists because you act first: if you bust, you lose even when the dealer busts afterward. Everything that favours you is a counterweight to that: your blackjack pays 3:2 while the dealer's pays even, you may double, split and surrender, and you can stand on a stiff hand while the dealer must draw.</p>
              <p>Under the current rules the edge with perfect basic strategy is roughly {houseEdge.toFixed(2)}%. That means at $25 a hand and 80 hands an hour you expect to lose about {money(25 * 80 * houseEdge / 100)} per hour. Change the rules in the Stats & rules tab and watch this move: the 6:5 blackjack payout alone adds about 1.4%, which is why a 6:5 table is the single worst thing to sit down at.</p>

              <h2>How often the dealer busts, by upcard</h2>
              <p>This is the heart of basic strategy. Weak upcards (4, 5, 6) mean the dealer starts from a stiff total and breaks often, so you stand on your own stiffs and double aggressively. Strong upcards (9, 10, ace) mean the dealer usually makes a hand, so you must keep drawing until you have one too.</p>
              <table className="plain nums">
                <thead><tr><th>Upcard</th><th>Bust</th><th>17</th><th>18</th><th>19</th><th>20</th><th>21</th></tr></thead>
                <tbody>
                  {dealerTable.map(({ u, d }) => (
                    <tr key={u}><td>{u}</td><td className={d[5] > 0.4 ? "hi" : ""}>{pct(d[5], 0)}</td>{d.slice(0, 5).map((x, i) => <td key={i}>{pct(x, 0)}</td>)}</tr>
                  ))}
                </tbody>
              </table>
              <p className="note">Ace and ten columns are conditioned on the dealer not holding blackjack, since the hand ends before you act when they do.</p>

              <h2>The handful of rules worth memorising</h2>
              <p>Always split aces and eights. Never split fives or tens. Double 11 against everything but an ace (and against an ace too if the dealer hits soft 17). Stand on hard 12 to 16 against a 2 to 6, hit them against a 7 or higher, except hit 12 against a 2 or 3. Hit soft 17 and below; double soft hands against 4, 5, 6. Surrender hard 16 against 9, 10, ace and hard 15 against a 10. Never take insurance unless you are counting.</p>
              <p>Blackjack is a game where the perfect decision is knowable in every spot. Unlike poker there is no opponent to read: the dealer follows a fixed rule, so each decision is just a probability calculation, and the chart tab shows the full result of that calculation.</p>
            </div>
          )}

          {tab === "chart" && (
            <div className="panel prose">
              <h2>Basic strategy for the current rules</h2>
              <p>Rows are your hand, columns are the dealer's upcard. Computed from the same engine that advises you at the table. H hit, S stand, D double (hit if not allowed), Ds double (stand if not allowed), P split, R surrender.</p>
              <ChartGrid title="Hard totals" rows={chart.hard} ups={chart.ups} />
              <ChartGrid title="Soft totals" rows={chart.soft} ups={chart.ups} />
              <ChartGrid title="Pairs" rows={chart.pairs} ups={chart.ups} />
              <p className="note">Uses the infinite-deck model (each card drawn independently). For a six-deck shoe it agrees with published charts except at a couple of razor-thin cells such as 16 vs 10 with three or more cards, where the exact composition matters slightly.</p>
            </div>
          )}

          {tab === "count" && (
            <div className="panel prose">
              <h2>Card counting (Hi-Lo)</h2>
              <div className="modeswitch">
                <span>Show the count on the felt</span>
                <button className={showCount ? "on" : ""} onClick={() => setShowCount(true)}>yes</button>
                <button className={!showCount ? "on" : ""} onClick={() => setShowCount(false)}>no</button>
              </div>
              <div className="countbig">
                <div><span>{runningCount >= 0 ? "+" : ""}{runningCount}</span>running count</div>
                <div><span>{trueCount >= 0 ? "+" : ""}{trueCount.toFixed(1)}</span>true count</div>
                <div><span>{decksLeft.toFixed(1)}</span>decks left</div>
              </div>
              <p>Basic strategy assumes every card is equally likely. In a real shoe the cards already dealt change what is left. A shoe rich in tens and aces favours you: blackjacks (paid 3:2) become more likely, doubles land big cards, and the dealer busts more often when forced to draw. A shoe rich in small cards favours the dealer.</p>
              <p>Hi-Lo tracks this with one number. Each 2, 3, 4, 5, 6 seen adds one; each 10, J, Q, K, A subtracts one; 7, 8, 9 do nothing. That is the running count. Divide by the number of decks left to get the true count, which is what actually predicts your edge, because a +6 count means much more with one deck left than with five.</p>
              <p>Each point of true count is worth roughly half a percent to you. The house starts about {houseEdge.toFixed(2)}% ahead, so around true count +1 the game is close to even and at +2 or better you have the advantage. Counters bet the minimum until the count turns and then scale up: a common ramp is one unit at +1 or below, then two, four, six, eight units at +2, +3, +4, +5. Insurance becomes a positive bet at true count +3 or higher, because more than a third of unseen cards are tens.</p>
              <p>The catch is variance and penetration. Even with a 1% edge, swings of hundreds of units are normal, and the dealer here reshuffles after {Math.round(PENETRATION * 100)}% of the shoe, which limits how deep the count can get. Casinos also watch for bet spreads. Practice here by keeping the count yourself with the display off, then checking.</p>
            </div>
          )}

          {tab === "stats" && (
            <div className="panel prose">
              <h2>Session</h2>
              <div className="statgrid">
                <div><span>{stats.hands}</span>hands</div>
                <div><span>{stats.hands ? pct(stats.wins / stats.hands, 0) : "0%"}</span>won</div>
                <div><span>{stats.hands ? pct(stats.pushes / stats.hands, 0) : "0%"}</span>pushed</div>
                <div><span>{stats.blackjacks}</span>blackjacks</div>
                <div><span>{stats.decisions ? pct(stats.correct / stats.decisions, 0) : "0%"}</span>decisions correct</div>
                <div><span>{money(stats.evLost)}</span>cost of mistakes</div>
              </div>
              <p className="note">Expect to win about 42% of hands, push 9%, lose 49%. Winning fewer hands than you lose while still nearly breaking even is normal: blackjacks, doubles and splits pay more than plain losses cost. Over a session of {stats.hands || "a few hundred"} hands, luck dominates; over ten thousand hands the edge shows. The "cost of mistakes" line is the part that is under your control.</p>

              <h2>Rules</h2>
              <div className="rulelist">
                <label><input type="checkbox" checked={rules.h17} onChange={(e) => setRules({ ...rules, h17: e.target.checked })} /> Dealer hits soft 17 <em>(+0.22% to the house)</em></label>
                <label><input type="checkbox" checked={rules.bj65} onChange={(e) => setRules({ ...rules, bj65: e.target.checked })} /> Blackjack pays 6:5 <em>(+1.39%)</em></label>
                <label><input type="checkbox" checked={rules.das} onChange={(e) => setRules({ ...rules, das: e.target.checked })} /> Double after split allowed <em>(-0.14%)</em></label>
                <label><input type="checkbox" checked={rules.surrender} onChange={(e) => setRules({ ...rules, surrender: e.target.checked })} /> Late surrender allowed <em>(-0.07%)</em></label>
              </div>
              <p>Fixed: six decks, reshuffle at {Math.round(PENETRATION * 100)}% penetration, dealer peeks for blackjack, split up to four hands, split aces receive one card each and cannot be resplit, insurance offered on a dealer ace. Estimated house edge with perfect play: {houseEdge.toFixed(2)}%.</p>
              <button className="ghost" onClick={resetBankroll}>Reset bankroll and stats</button>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

function DealerOutcomes({ dd, up }) {
  return (
    <div className="dealerout">
      <div className="doutlabel">Dealer finishing on, from a {up}</div>
      <div className="doutrow">
        {["17", "18", "19", "20", "21", "bust"].map((l, i) => (
          <div key={l} className={l === "bust" ? "bust" : ""}>
            <b>{pct(dd[i], 0)}</b>
            <span>{l}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ChartGrid({ title, rows, ups }) {
  return (
    <div className="chartblock">
      <h3>{title}</h3>
      <table className="chart">
        <thead>
          <tr><th></th>{ups.map((u) => <th key={u}>{u}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label}>
              <th>{r.label}</th>
              {r.cells.map((c, i) => <td key={i} className={`c-${c}`}>{c}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ----------------------------------------------------------------------
   Styles
---------------------------------------------------------------------- */
const CSS = `
.bj{--felt:#1F4D3E;--felt2:#163A2F;--rim:#0F2A21;--cream:#F6EFE0;--paper:#EFE5CF;--paper2:#E3D6B8;--ink:#1F1D18;--ink2:#5A5347;--brass:#C9A227;--brass2:#8F7119;--red:#B23A32;--green:#2F7D4F;--back:#6E2B2B;
  font-family:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;color:var(--ink);background:var(--rim);min-height:100vh;padding:20px;box-sizing:border-box;font-variant-numeric:tabular-nums;}
.bj *{box-sizing:border-box}
.bj button{font-family:inherit;cursor:pointer}
.bj button:focus-visible{outline:3px solid var(--brass);outline-offset:2px}
.bj button:disabled{opacity:.4;cursor:not-allowed}
.masthead{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;color:var(--cream);margin-bottom:16px;flex-wrap:wrap}
.masthead h1{font-size:30px;font-weight:600;margin:0;letter-spacing:-.01em;line-height:1.1}
.masthead .sub{margin:6px 0 0;color:#B9CDBF;font-size:14px;max-width:60ch}
.bank{text-align:right}
.bankval{font-size:34px;font-weight:600;color:var(--brass);line-height:1}
.banklabel{font-size:13px;color:#B9CDBF;margin-top:4px}
.layout{display:grid;grid-template-columns:minmax(0,3fr) minmax(320px,2fr);gap:18px;align-items:start}
@media(max-width:860px){.layout{grid-template-columns:1fr}}

/* table */
.table{display:flex;flex-direction:column;gap:12px}
.felt{background:radial-gradient(ellipse at 50% 30%,#276150 0%,var(--felt) 55%,var(--felt2) 100%);border:10px solid #3A2A1F;border-radius:200px 200px 28px 28px;padding:26px 24px 20px;box-shadow:inset 0 0 40px rgba(0,0,0,.35),0 10px 30px rgba(0,0,0,.4);min-height:420px;display:flex;flex-direction:column;justify-content:space-between;gap:14px}
.seat{display:flex;flex-direction:column;align-items:center;gap:8px}
.seatname{color:#D7E6DB;font-size:14px}
.seatname .total{color:var(--cream);margin-left:8px;font-weight:600}
.cards{display:flex;min-height:96px}
.cards > *{margin-left:-14px}
.cards > *:first-child{margin-left:0}
.slot{width:66px;height:94px;border:1.5px dashed rgba(246,239,224,.35);border-radius:7px}
.empty{display:flex;gap:8px}
.empty .slot{margin-left:0}
.shoeinfo{text-align:center;color:#A9C2B2;font-size:12.5px;display:flex;justify-content:center;gap:16px;flex-wrap:wrap}
.countchip{color:var(--brass)}
.hands{display:flex;gap:22px;flex-wrap:wrap;justify-content:center}
.hand{display:flex;flex-direction:column;align-items:center;gap:6px;padding:8px 10px 6px;border-radius:12px;border:2px solid transparent;transition:border-color .2s}
.hand.activehand{border-color:var(--brass);background:rgba(201,162,39,.08)}
.handmeta{display:flex;flex-direction:column;align-items:center;font-size:13px;color:#D7E6DB;line-height:1.35}
.handmeta .total{color:var(--cream);font-weight:600;font-size:15px}
.betlabel{color:#A9C2B2}
.result{font-weight:600}
.result.win{color:#F2D77A}.result.loss{color:#F0A29A}.result.push{color:#D7E6DB}
.insres{text-align:center;margin-top:6px;font-size:13px}

.bjcard{width:66px;height:94px;background:var(--cream);border-radius:7px;position:relative;color:var(--ink);box-shadow:0 2px 6px rgba(0,0,0,.35),0 0 0 1px rgba(0,0,0,.15);animation:deal .32s ease-out both;flex:none}
.bjcard.red{color:var(--red)}
.bjcard .corner{position:absolute;display:flex;flex-direction:column;align-items:center;line-height:1;font-size:15px;font-weight:600}
.bjcard .corner span+span{font-size:12px;margin-top:1px}
.bjcard .tl{top:6px;left:6px}
.bjcard .br{bottom:6px;right:6px;transform:rotate(180deg)}
.bjcard .pip{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:34px}
.bjcard.back{background:var(--back);background-image:repeating-linear-gradient(45deg,rgba(255,255,255,.12) 0 4px,transparent 4px 10px),repeating-linear-gradient(-45deg,rgba(255,255,255,.12) 0 4px,transparent 4px 10px);box-shadow:inset 0 0 0 4px var(--cream),0 2px 6px rgba(0,0,0,.35)}
@keyframes deal{from{opacity:0;transform:translateY(-18px) rotate(-3deg)}to{opacity:1;transform:none}}
@media(prefers-reduced-motion:reduce){.bjcard{animation:none}}

.controls{background:#1B1712;border-radius:12px;padding:12px 14px;min-height:64px;display:flex;align-items:center}
.betrow{display:flex;align-items:center;gap:14px;flex-wrap:wrap;width:100%}
.chips{display:flex;gap:8px;align-items:center}
.chip{width:46px;height:46px;border-radius:50%;border:4px dashed rgba(255,255,255,.7);color:#fff;font-weight:700;font-size:14px;box-shadow:0 2px 0 rgba(0,0,0,.5);background:#555}
.chip.c5{background:#B23A32}.chip.c25{background:#2F7D4F}.chip.c100{background:#2C3E7A}.chip.c500{background:#5B2E7E}
.chip:not(:disabled):hover{transform:translateY(-2px)}
.betstate{color:var(--cream);font-size:18px;font-weight:600}
.prompt{color:#D7CFBE;font-size:14px}
.bj .primary{background:var(--brass);color:#1B1712;border:none;border-radius:8px;padding:10px 20px;font-size:16px;font-weight:600}
.bj .ghost{background:transparent;color:#D7CFBE;border:1px solid #4A4237;border-radius:8px;padding:8px 14px;font-size:14px}
.actions{display:flex;gap:8px;flex-wrap:wrap;width:100%}
.actions button{flex:1;min-width:90px;background:#2A2419;color:var(--cream);border:1px solid #4A4237;border-radius:8px;padding:12px 10px;font-size:16px;transition:background .15s}
.actions button:not(:disabled):hover{background:#3A3223}
.actions button.rec{border-color:var(--brass);box-shadow:inset 0 0 0 1px var(--brass);color:var(--brass)}

/* ledger */
.ledger{background:var(--paper);border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,.35);overflow:hidden;background-image:linear-gradient(rgba(0,0,0,.03) 1px,transparent 1px);background-size:100% 28px}
.tabs{display:flex;border-bottom:1px solid var(--paper2);background:var(--paper2);flex-wrap:wrap}
.tabs button{flex:1;background:transparent;border:none;padding:11px 8px;font-size:13.5px;color:var(--ink2);border-bottom:2px solid transparent;white-space:nowrap}
.tabs button.on{color:var(--ink);background:var(--paper);border-bottom-color:var(--brass);font-weight:600}
.panel{padding:18px 20px 22px}
.prose h2{font-size:19px;font-weight:600;margin:14px 0 8px;line-height:1.25}
.prose h2:first-child{margin-top:0}
.prose h3{font-size:15px;font-weight:600;margin:14px 0 6px}
.prose p{font-size:14.5px;line-height:1.55;margin:0 0 10px;max-width:62ch}
.prose .note{font-size:13px;color:var(--ink2)}
.verdict{font-size:40px;font-weight:600;color:var(--brass2);line-height:1;margin:2px 0 10px}
.modeswitch{display:flex;gap:6px;align-items:center;font-size:13px;color:var(--ink2);margin-bottom:14px;flex-wrap:wrap}
.modeswitch button{background:transparent;border:1px solid var(--paper2);border-radius:14px;padding:3px 10px;font-size:13px;color:var(--ink2)}
.modeswitch button.on{background:var(--ink);color:var(--paper);border-color:var(--ink)}

.evbars{margin:6px 0 14px;display:flex;flex-direction:column;gap:5px}
.evrow{display:grid;grid-template-columns:78px 1fr 62px;align-items:center;gap:8px;font-size:13.5px;padding:3px 6px;border-radius:6px}
.evrow.best{background:rgba(201,162,39,.16);font-weight:600}
.evrow.chosen:not(.best){background:rgba(178,58,50,.12)}
.evtrack{position:relative;height:12px;background:rgba(0,0,0,.07);border-radius:3px}
.evzero{position:absolute;top:-2px;bottom:-2px;width:1px;background:var(--ink2)}
.evfill{position:absolute;top:0;bottom:0;border-radius:2px}
.evfill.pos{background:var(--green)}.evfill.neg{background:var(--red)}
.evnum{text-align:right}

.feedback{border-left:4px solid;padding:10px 12px;margin-top:8px;border-radius:0 8px 8px 0;background:rgba(0,0,0,.035)}
.feedback.good{border-color:var(--green)}.feedback.bad{border-color:var(--red)}
.feedback h3{margin:0 0 6px;font-size:15px}
.feedback p{margin:0;font-size:14px;line-height:1.5}

.dealerout{margin-top:10px}
.doutlabel{font-size:12.5px;color:var(--ink2);margin-bottom:6px}
.doutrow{display:grid;grid-template-columns:repeat(6,1fr);gap:4px}
.doutrow div{background:rgba(0,0,0,.05);border-radius:6px;padding:6px 4px;text-align:center;font-size:12px;color:var(--ink2);display:flex;flex-direction:column}
.doutrow div b{font-size:14px;color:var(--ink)}
.doutrow div.bust b{color:var(--red)}

table.plain{width:100%;border-collapse:collapse;font-size:13.5px;margin:6px 0 12px}
table.plain td,table.plain th{padding:5px 6px;border-bottom:1px solid var(--paper2);text-align:left;vertical-align:top}
table.plain th{font-weight:600;color:var(--ink2)}
table.plain.nums td:not(:first-child),table.plain.nums th:not(:first-child){text-align:right}
table.plain td.hi{color:var(--red);font-weight:600}

.chartblock{margin:10px 0 6px}
table.chart{border-collapse:separate;border-spacing:2px;font-size:12px;width:100%}
table.chart th{font-weight:600;color:var(--ink2);padding:2px}
table.chart td{text-align:center;padding:4px 0;border-radius:3px;color:#fff;font-weight:600}
td.c-H{background:#B23A32}td.c-S{background:#2F7D4F}td.c-D,td.c-Ds{background:#2C5AA0}td.c-P{background:#7A3E9C}td.c-R{background:#6B6155}

.countbig,.statgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0 14px}
.countbig div,.statgrid div{background:rgba(0,0,0,.05);border-radius:8px;padding:10px;text-align:center;font-size:12.5px;color:var(--ink2);display:flex;flex-direction:column;gap:2px}
.countbig div span,.statgrid div span{font-size:24px;font-weight:600;color:var(--ink)}
.rulelist{display:flex;flex-direction:column;gap:8px;margin:6px 0 12px;font-size:14px}
.rulelist em{color:var(--ink2);font-style:normal;font-size:13px}
.rulelist input{margin-right:6px}
`;

import { legalActions, BB, fmt } from "./engine.js";
import { handKey, handPct } from "./preflop.js";
import { seatPos, openPct, bbDefendPct, heroInPosition } from "./positions.js";
import { readOf, readLabel } from "./reads.js";
import { simulate } from "./equity.js";
import {
  outsAnalysis,
  handClass,
  boardTexture,
  cbetPlan,
  combosNow,
  rangeLists,
  gridCounts,
  composition,
  nextCardMap,
  evSamples,
  evOfRaise,
} from "./analysis.js";

// The coach. analyze() looks at the hero's spot the way a good player would: what ranges the action
// implies, the hero's equity against them, the price being offered, what the board does for each side,
// and what each option is worth. recommend() turns that into one action, acceptable alternatives and the
// reasoning in words. grade() marks the hero's choice against it.

// The table as the coach sees it: engine state joined with what has been observed about each player.
function view(state, reads) {
  return {
    n: state.config.n,
    street: state.street,
    raises: state.raises,
    board: state.board,
    currentBet: state.currentBet,
    players: state.players.map((p) => ({
      ...p,
      ...reads.players[p.id],
      hero: p.id === 0,
      pos: seatPos(state, p.id),
      read: readOf(reads, p.id),
      label: readLabel(reads, p.id),
    })),
  };
}

function evContext(g, o) {
  const hero = g.players[0];
  return {
    P: o.pot,
    toCall: o.toCall,
    heroBet: hero.bet,
    currentBet: g.currentBet,
    opps: g.players
      .filter((p) => !p.folded && !p.hero)
      .map((p) => ({ bet: p.bet, stack: p.stack, allIn: p.allIn, station: p.read.label === "Station" })),
  };
}

export function decisionCategory(g, o) {
  if (g.street === 0) {
    if (g.raises === 0) return o.canCheck ? "Big blind, unopened" : "Open or fold";
    if (g.raises === 1) return "Facing a raise";
    return "Facing a 3-bet";
  }
  if (o.canCheck) return "Bet or check";
  return "Facing a bet";
}

// `quality` scales the inner sample counts (range composition, next-card map, EV samples): 1 on the main thread,
// more in the worker, where there is time to spare and steadier numbers are worth having.
export function analyze(state, reads, rng, { iters = 3000, light = false, quality = 1 } = {}) {
  const o = legalActions(state);
  if (!o || o.seat !== 0) throw new Error("analyze: the hero is not to act");
  const g = view(state, reads);
  const hero = g.players[0],
    posn = hero.pos;
  const opps = g.players.filter((p) => !p.folded && !p.hero);
  const ranges = opps.map((p) => ({ pct: p.pct, filters: p.filters }));
  const eq = simulate(hero.cards, g.board, ranges, iters, rng);
  const eqRandom = simulate(
    hero.cards,
    g.board,
    opps.map(() => ({ pct: 100, filters: [] })),
    Math.round(iters / 2),
    rng
  );
  const potNow = o.pot,
    toCall = o.toCall;
  const potOdds = toCall > 0 ? toCall / (potNow + toCall) : 0;
  const evCallVal = toCall > 0 ? eq * potNow - (1 - eq) * toCall : 0;
  const pct = handPct(hero.cards[0], hero.cards[1]);
  const key = handKey(hero.cards[0], hero.cards[1]);
  const oa = g.street > 0 ? outsAnalysis(hero.cards, g.board) : null;
  const hc = g.street > 0 ? handClass(hero.cards, g.board) : null;
  const tex = g.street > 0 ? boardTexture(g.board) : null;
  const combos = g.street > 0 ? combosNow(hero.cards, g.board, ranges) : null;
  const effStack = Math.min(hero.stack + hero.bet, Math.max(...opps.map((p) => p.stack + p.bet)));
  const spr = g.street > 0 ? effStack / Math.max(1, potNow) : null;
  const ip = heroInPosition(
    state,
    opps.map((p) => p.id)
  );
  const rec = recommend(g, hero, o, eq, pct, oa, opps, posn, hc, combos, ip);
  if (!o.canRaise && rec.action === "raise") {
    rec.action = "call";
    rec.size = 0;
    rec.alts = rec.alts.filter((x) => x !== "call");
  } // calling is already all-in
  const category = decisionCategory(g, o);
  // Decision-time panels: live ranges, composition, next cards, EV by action
  const lists = rangeLists(hero.cards, g.board, ranges);
  const post = g.street > 0 && !light;
  const comps = post ? lists.map((l) => composition(l, hero.cards, g.board, rng, Math.round(260 * quality))) : null;
  const nextCards = post ? nextCardMap(hero.cards, g.board, lists, Math.round((g.street === 1 ? 160 : 220) * quality), rng) : null;
  let ev = null,
    ctx = null,
    samples = null;
  if (post) {
    ctx = evContext(g, o);
    samples = o.canRaise ? evSamples(hero.cards, g.board, lists, Math.round(1400 * quality), { pct: hero.pct, filters: hero.filters }, rng) : null;
    const passive = toCall > 0 ? evCallVal : eq * potNow;
    const potTotal = potNow + toCall;
    const sizeTo = (f) => Math.max(o.minTo, Math.min(o.maxTo, hero.bet + toCall + Math.round(f * potTotal)));
    const rows = [];
    if (samples) {
      const seen = {};
      for (const [label, to] of [
        ["\u00bd pot", sizeTo(0.5)],
        ["\u2154 pot", sizeTo(0.66)],
        ["pot", sizeTo(1)],
        ["all-in", o.maxTo],
      ]) {
        if (seen[to]) continue;
        seen[to] = 1;
        rows.push(Object.assign({ label, to }, evOfRaise(ctx, samples, to)));
      }
    }
    const bestRaise = rows.length ? rows.reduce((x, y) => (y.ev > x.ev ? y : x)) : null;
    let best = toCall > 0 ? (passive > 0 ? "call" : "fold") : "check";
    let bestEV = Math.max(0, passive);
    if (bestRaise && bestRaise.ev > bestEV + BB * 0.5) {
      best = "raise";
      bestEV = bestRaise.ev;
    }
    ev = { passive, rows, best, bestEV, bestRaiseTo: bestRaise ? bestRaise.to : 0, disagree: false };
    if (best !== rec.action && !rec.alts.includes(best)) {
      ev.disagree = true;
      // Only let the EV model widen the grading for value raises and real semi-bluffs, never for pure bluffs.
      const soundRaise = best !== "raise" || (bestRaise.eqCalled !== null && bestRaise.eqCalled >= 0.45) || (oa && oa.bigOuts >= 8 && g.street < 3);
      if (soundRaise) rec.alts.push(best);
      else ev.pureBluff = true;
    }
  }
  return {
    eq,
    eqRandom,
    potNow,
    toCall,
    potOdds,
    evCall: evCallVal,
    pct,
    key,
    oa,
    hc,
    tex,
    combos,
    spr,
    ip,
    posn,
    rec,
    category,
    ev,
    ctx,
    samples,
    nextCards,
    opps: opps.map((p, k) => ({
      id: p.id,
      name: p.name,
      tag: p.tag,
      pct: p.pct,
      style: p.label,
      read: p.read,
      pos: p.pos,
      story: p.story.slice(),
      filters: p.filters.slice(),
      now: combos ? combos.perOpp[k] : null,
      nCombos: lists[k].length,
      grid: gridCounts(lists[k]),
      comp: comps ? comps[k] : null,
    })),
    canCheck: o.canCheck,
    canRaise: o.canRaise,
    minTo: o.minTo,
    maxTo: o.maxTo,
    street: g.street,
    heroBet: hero.bet,
    currentBet: g.currentBet,
  };
}

export function recommend(g, hero, o, eq, pct, oa, opps, posn, hc, combos, ip) {
  const R = { action: "", size: 0, reasons: [], alts: [] };
  const potNow = o.pot,
    toCall = o.toCall,
    street = g.street;
  const nOpp = opps.length;
  const p = (v) => Math.round(v) + "%";
  if (street === 0) {
    const openPctV = openPct(posn, g.n);
    const limpers = g.players.filter((q) => q.tag === "limp").length;
    if (g.raises === 0) {
      const isBB = posn === "BB" && o.canCheck;
      if (isBB) {
        if (pct <= 14) {
          R.action = "raise";
          R.size = BB * (3 + limpers);
          R.reasons.push(
            "Nobody raised and you close the action in the big blind. With a top " +
              p(pct) +
              " hand, raise to build a pot against limpers who have already shown weakness."
          );
          R.alts = ["check"];
        } else {
          R.action = "check";
          R.reasons.push(
            "You can see a flop for free. Never fold a free flop, and raising a weak hand here just builds a pot out of position (you act first every street after this)."
          );
          if (pct <= 30) R.alts = ["raise"];
        }
        return R;
      }
      if (pct <= openPctV) {
        R.action = "raise";
        R.size = BB * (2.5 + limpers);
        R.reasons.push(
          "Your hand (" +
            p(pct) +
            " by strength) is inside the standard open-raising range for " +
            posn +
            " (roughly the top " +
            openPctV +
            "% of hands). Raise rather than limp: raising wins the blinds outright, takes the initiative, and builds a pot when you are likely ahead."
        );
        if (limpers)
          R.reasons.push(
            "There " +
              (limpers === 1 ? "is 1 limper" : "are " + limpers + " limpers") +
              ", so size up by about 1 bb each. Limpers usually have weak, capped ranges and fold a lot or call with hands you dominate."
          );
      } else {
        R.action = "fold";
        R.reasons.push(
          "Your hand is outside a reasonable opening range from " +
            posn +
            " (top ~" +
            openPctV +
            "%). The earlier you act, the more players behind you can wake up with a strong hand, so early positions must be tighter."
        );
        if (pct <= openPctV + 10 && (posn === "BTN" || posn === "CO")) {
          R.alts = ["raise"];
          R.reasons.push(
            "It is borderline: in late position a raise is defensible because you will act last on every later street and can steal the blinds."
          );
        }
      }
      return R;
    }
    const raiser = g.players.find((q) => q.tag === (g.raises === 1 ? "raise" : g.raises === 2 ? "3bet" : "4bet"));
    const rRead = raiser ? raiser.read : null;
    const looseRaiser = !!(rRead && rRead.n >= 15 && rRead.pfr >= 33);
    const rn = raiser ? raiser.name + " (" + raiser.pos + (raiser.label ? ", reads as " + raiser.label : "") + ")" : "the raiser";
    const rp = raiser ? Math.round(raiser.pct) : 20;
    if (g.raises === 1) {
      const raiserPos = raiser ? raiser.pos : "";
      const callPct = posn === "BB" ? bbDefendPct(raiserPos, g.n) : ip ? 18 : 12;
      R.pre = { call: callPct, raise: looseRaiser ? 14 : 6, raiseWord: "3-bet", bb: posn === "BB", raiserPos };
      const vsManiac = looseRaiser;
      if (pct <= 6 || (vsManiac && pct <= 14)) {
        R.action = "raise";
        R.size = Math.min(o.maxTo, Math.round(g.currentBet * (ip ? 3 : 4)));
        R.reasons.push(
          (vsManiac && pct > 6
            ? "Against a player who has been raising about " +
              Math.round(rRead ? rRead.pfr : 45) +
              "% of hands, a hand this strong is a clear value 3-bet even though it would only be a call against a tight raiser."
            : "Premium hand (top " + p(pct) + ").") +
            " Re-raise (a 3-bet) for value: you want to grow the pot while you are far ahead of " +
            rn +
            "'s opening range (roughly the top " +
            rp +
            "%)."
        );
        R.reasons.push(
          ip
            ? "In position, 3x the raise is standard."
            : "Out of position, size up to about 4x so opponents pay more to use their positional advantage."
        );
        R.alts = ["call"];
      } else if (pct <= callPct) {
        R.action = "call";
        R.reasons.push(
          "Your hand plays well but is not strong enough to 3-bet for value against a top-" +
            rp +
            "% opening range. Calling keeps their weaker hands in and keeps the pot manageable."
        );
        R.reasons.push(
          ip
            ? "You are in position, which is worth a lot: you see their action first on every street and can control pot size."
            : posn === "BB"
              ? "In the big blind you already have 1 bb invested and close the action, so you continue much wider than any other seat: roughly the top " +
                callPct +
                "% against a raise from " +
                raiserPos +
                ". The later the raiser's seat, the wider their range, and the wider you defend."
              : "You are out of position, so you need a stronger hand to continue than you would in position."
        );
        if (pct <= 10) R.alts = ["raise"];
      } else {
        R.action = "fold";
        R.reasons.push(
          "Against a raise you should continue with only a fraction of the hands you would open yourself. Hands that are dominated (like weak aces or weak kings) lose big pots when they hit."
        );
        R.reasons.push("Rough guide: 3-bet the top ~6%, call with the next ~" + (callPct - 6) + "% (more in position), fold the rest.");
        if (pct <= callPct + 8) R.alts = [ip ? "call" : "fold"];
      }
      return R;
    }
    R.pre = { call: 7, raise: 2.5, raiseWord: "4-bet" };
    if (pct <= 2.5) {
      R.action = "raise";
      R.size = o.maxTo;
      R.reasons.push(
        "Facing a 3-bet with a premium (top " +
          p(pct) +
          ") hand: keep raising. With 100 bb stacks, a 4-bet often commits you, so moving all-in is fine."
      );
      R.alts = ["call"];
    } else if (pct <= 7) {
      R.action = "call";
      R.reasons.push(
        "Strong but not premium. Against a 3-betting range (top ~" +
          rp +
          "%) you are roughly even, and calling in a bloated pot with a hand that flops well is reasonable. 4-betting risks facing a shove where you are behind."
      );
      R.alts = ["fold"];
    } else {
      R.action = "fold";
      R.reasons.push(
        "A 3-bet represents a very tight range (top ~" +
          rp +
          "%). Most hands, even ones that look pretty, are badly dominated and should fold. Do not pay 10+ bb to see a flop as the underdog."
      );
    }
    return R;
  }
  // Postflop
  const strong = oa && oa.bigOuts >= 8 && street < 3;
  const nuts = combos && combos.isNuts;
  const wasRaiser = ["raise", "3bet", "4bet"].includes(hero.tag);
  const plan = o.canCheck && street === 1 && nOpp === 1 && wasRaiser ? cbetPlan(g.board) : null;
  if (plan) {
    R.plan = plan.kind;
    R.reasons.push(plan.why);
    const valueish = eq >= (plan.kind === "check" ? 0.65 : 0.6) || strong;
    if (plan.kind === "small") {
      R.action = "raise";
      R.size = Math.round(potNow * plan.frac);
      R.alts = ["check"];
      R.reasons.push(
        eq >= 0.6
          ? "Your own hand is strong here (about " +
              p(eq * 100) +
              " equity), which is a bonus. Keep the size small anyway: betting big only with your good hands tells opponents what you hold."
          : "Your own hand has about " +
              p(eq * 100) +
              " equity. That matters less than usual, because this bet is made with your whole range. Checking is also fine, especially out of position."
      );
      if (!ip)
        R.reasons.push(
          "Out of position you should check a bit more often than in position, because a raise or a call leaves you acting first on every later street."
        );
    } else if (valueish) {
      R.action = "raise";
      R.size = Math.round(potNow * plan.frac);
      R.alts = ["check"];
      R.reasons.push(
        strong && eq < 0.6
          ? "Your draw (" + oa.bigOuts + " outs) is one of the hands that wants to bet: it can win now or improve later."
          : "Your hand is in the strong part of your range (about " + p(eq * 100) + " equity), so it bets."
      );
    } else {
      R.action = "check";
      R.reasons.push(
        "Your hand (about " +
          p(eq * 100) +
          " equity) is in the middle or bottom of your range on this board, so it checks." +
          (eq >= 0.45 ? " It has enough showdown value that you do not need to turn it into a bluff." : "")
      );
    }
    return R;
  }
  if (o.canCheck) {
    if (nuts) {
      R.action = "raise";
      R.size = Math.round(potNow * 0.75);
      R.reasons.push(
        "You have the best possible hand right now (the nuts). Bet: nobody can be ahead, so every chip that goes in is profit, and checking only gives free cards to hands that could outdraw you. Size large; opponents with strong second-best hands will pay."
      );
      if (street < 3 && nOpp === 1) R.alts = ["check"];
    } else if (eq >= 0.6) {
      R.action = "raise";
      R.size = Math.round(potNow * 0.66);
      R.reasons.push(
        "Your equity is about " +
          p(eq * 100) +
          " against " +
          nOpp +
          " opponent" +
          (nOpp > 1 ? "s" : "") +
          ". That is a value-betting hand: bet so worse hands pay you, and so draws must pay a bad price to continue."
      );
      R.reasons.push(
        "Around two-thirds of the pot is a good default. It charges draws (a flush draw has ~35% equity with two cards to come, so it should not be offered better than 2 to 1) while still getting called by weaker made hands."
      );
      if (eq < 0.7) R.alts = ["check"];
    } else if (strong && nOpp <= 2) {
      R.action = "raise";
      R.size = Math.round(potNow * 0.5);
      R.reasons.push(
        "You have a strong draw (" +
          oa.bigOuts +
          " outs to a straight or flush). Betting is a semi-bluff: you win immediately when everyone folds, and when called you still have about " +
          p(eq * 100) +
          " equity to improve."
      );
      R.reasons.push("Semi-bluffs work best against one or two opponents; against many players someone usually has enough to call.");
      R.alts = ["check"];
    } else if (eq >= 0.45) {
      R.action = "check";
      R.reasons.push(
        "Medium strength (about " +
          p(eq * 100) +
          " equity). Betting mostly folds out worse hands and gets called by better ones, which is the worst of both. Check to control the pot size and see a cheap next card."
      );
      R.alts = ["raise"];
    } else {
      R.action = "check";
      R.reasons.push(
        "Weak hand (about " +
          p(eq * 100) +
          " equity) and no strong draw. Take the free card. Bluffing into " +
          nOpp +
          " opponent" +
          (nOpp > 1 ? "s" : "") +
          " with nothing works rarely; save bluffs for spots with fold equity and a backup draw."
      );
      if (nOpp === 1 && ip) {
        R.alts = ["raise"];
        R.reasons.push(
          "One exception: heads-up and in position, a small stab (a third of the pot) as a pure bluff can be fine after your opponent checks to you, because they have shown weakness."
        );
      }
    }
    return R;
  }
  const req = toCall / (potNow + toCall);
  const margin = eq - req;
  R.reasons.push(
    "Pot odds: you must call " +
      fmt(toCall) +
      " to win a pot of " +
      fmt(potNow) +
      ", so you need at least " +
      p(req * 100) +
      " equity to break even. Your equity is about " +
      p(eq * 100) +
      "."
  );
  if (nuts) {
    R.action = "raise";
    R.size = Math.min(o.maxTo, Math.round(g.currentBet * 3));
    R.reasons.push(
      "You hold the nuts. Raise: your opponent has bet into the best possible hand, so get as much money in as they will put in. Only slow down if a raise folds everything and a call keeps them bluffing on later streets."
    );
    R.alts = ["call"];
  } else if (eq >= 0.7) {
    R.action = "raise";
    R.size = Math.min(o.maxTo, Math.round(g.currentBet * 2.8));
    R.reasons.push(
      "You are far ahead. Raise for value: with " +
        p(eq * 100) +
        " equity every chip that goes in is profitable, and a raise now builds a bigger pot for the later streets."
    );
    R.alts = ["call"];
    if (street < 3)
      R.reasons.push(
        "Slow-playing (just calling) is occasionally right on very dry boards, but as a default, fast-play strong hands. Opponents call more than they should."
      );
  } else if (margin >= 0.03) {
    R.action = "call";
    R.reasons.push(
      "You have more equity than the price requires, so calling is +EV. You are not strong enough to raise for value, and raising as a bluff throws away a hand that already wins often enough."
    );
    if (strong) {
      R.alts = ["raise"];
      R.reasons.push("With a strong draw, a semi-bluff raise is also reasonable: it adds fold equity to your existing pot equity.");
    }
  } else if (margin >= -0.06 && strong && street < 3 && o.maxTo > toCall * 4) {
    R.action = "call";
    R.reasons.push(
      "Slightly short on direct pot odds, but you have " +
        oa.bigOuts +
        " clean outs and deep stacks behind. Implied odds (the extra chips you expect to win on later streets when you hit) make up the shortfall. This only works if your opponent is likely to pay you off when you improve."
    );
    R.alts = ["fold"];
  } else if (margin >= -0.04) {
    R.action = "fold";
    R.reasons.push(
      "Close, but your equity is slightly below the price. When it is this close, fold unless you have a specific reason to continue (implied odds, position, a read)."
    );
    R.alts = ["call"];
  } else {
    R.action = "fold";
    R.reasons.push(
      "Your equity (" +
        p(eq * 100) +
        ") is below the required " +
        p(req * 100) +
        ". Calling loses money on average: EV of calling is about " +
        fmt(eq * potNow - (1 - eq) * toCall) +
        ". Folding has EV zero, which beats a negative number."
    );
    if (street === 3)
      R.reasons.push(
        'On the river there are no more cards to come, so implied odds no longer exist. It is purely "is my hand good often enough for this price".'
      );
  }
  return R;
}

export function grade(rec, action) {
  if (action === rec.action) return "correct";
  if (rec.alts.includes(action)) return "acceptable";
  return "mistake";
}

// Lesson picker: which concept does this decision teach?
function conceptFor(a) {
  const c = a.category;
  if (c === "Open or fold" || c === "Big blind, unopened")
    return {
      concept: "Opening ranges by position",
      text: "Preflop discipline is the cheapest skill to fix: tight early, wide late, raise rather than limp.",
    };
  if (c === "Facing a raise" || c === "Facing a 3-bet")
    return {
      concept: "Ranges, not hands",
      text: "Continuing against a raise is about how your hand fares against the raiser's whole range, not about how pretty it looks.",
    };
  if (c === "Facing a bet") {
    if (a.oa && a.oa.hasRealDraw)
      return {
        concept: "Outs and the rule of 4 and 2",
        text: "With a draw: count outs, convert to a percentage, compare with the price, then think about implied odds.",
      };
    return {
      concept: "Pot odds and required equity",
      text: "Every call is a price. Compute the required equity first, then ask whether your hand clears it against their range.",
    };
  }
  if (c === "Bet or check") {
    if (a.rec && a.rec.plan)
      return { concept: "Range advantage", text: "As the preflop raiser on the flop, the board decides your plan before your own two cards do." };
    if (a.eq >= 0.6)
      return { concept: "Bet sizing", text: "With a strong hand, betting is about extracting value and charging draws. Size by the job." };
    if (a.oa && a.oa.bigOuts >= 8)
      return { concept: "Semi-bluffs and fold equity", text: "A draw plus a bet has two ways to win. That is why semi-bluffs are the best bluffs." };
    return { concept: "Board texture", text: "Whether to bet a medium hand depends on the board and on whose range it favours." };
  }
  return { concept: "Equity", text: "" };
}

// Which lesson does this decision teach? Used for the "study this" link after a mistake and for
// pointing the home screen at the learner's weakest area.
const LESSON_OF = {
  "Opening ranges by position": "start",
  "Ranges, not hands": "ranges",
  "Outs and the rule of 4 and 2": "outs",
  "Pot odds and required equity": "odds",
  "Bet sizing": "betting",
  "Semi-bluffs and fold equity": "betting",
  "Board texture": "texture",
  Equity: "equity",
  "Range advantage": "cbet",
};
export const CATEGORY_LESSON = {
  "Open or fold": "start",
  "Big blind, unopened": "start",
  "Facing a raise": "ranges",
  "Facing a 3-bet": "ranges",
  "Bet or check": "betting",
  "Facing a bet": "odds",
};
export function lessonFor(a) {
  const l = conceptFor(a);
  return { ...l, lessonId: LESSON_OF[l.concept] || CATEGORY_LESSON[a.category] || "equity" };
}

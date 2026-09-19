import { pot as potOf, fmt, seatPos, STREET_NAMES } from "./core/index.js";

// The one bridge between the poker engine and the stage. It speaks two languages:
//   scene  - a plain description of what is on the table. applyScene() makes the stage look like it, at once.
//   events - what the engine says just happened. play() acts them out, one after another, and then snaps
//            to the scene the engine's state describes, so the picture can never drift from the truth.

// What the table looks like for a given engine state. Opponents' cards stay face down (and are not even
// named in the scene) unless their seat is listed in `reveal`.
export function sceneFromState(state, { reveal = [], highlight = null, acting = true } = {}) {
  const live = !state.handOver || state.result?.showdown;
  return {
    n: state.config.n,
    dealer: state.dealer,
    handNo: state.handNo,
    seats: state.players.map((p) => {
      const shown = p.id === 0 || reveal.includes(p.id);
      return {
        seat: p.id,
        name: p.name,
        pos: state.dealer >= 0 ? seatPos(state, p.id) : "",
        stack: p.stack,
        bet: p.bet,
        hasCards: live && p.cards.length === 2 && !p.folded,
        cards: shown && p.cards.length === 2 ? p.cards : null,
        folded: p.folded,
        allIn: p.allIn,
        acting: acting && state.toAct === p.id,
      };
    }),
    board: state.board.slice(),
    pot: potOf(state) - state.players.reduce((s, p) => s + p.bet, 0),
    highlight,
  };
}

const cardKey = (c) => c.r * 4 + c.s;
const holeId = (seat, i) => "h" + seat + "ab"[i];

export function createDirector({ stage, table, cards, chips, labels, fx, rig }) {
  let seats = [];
  let speed = 1;
  const L = () => table.layout; // board, pot and deck positions for the table's current orientation
  const hooks = { sound: () => {} };

  function seatLabel(s, bubble) {
    const lay = seats[s.seat];
    const cls = "seat" + (s.folded ? " folded" : "") + (s.acting ? " acting" : "") + (s.seat === 0 ? " hero" : "");
    const html =
      `<span class="name">${s.name}</span>` +
      `<span class="meta"><b>${fmt(s.stack)}</b>${s.pos ? `<i>${s.pos}</i>` : ""}${s.allIn ? `<em>all-in</em>` : ""}</span>` +
      (bubble ? `<span class="bubble">${bubble}</span>` : "");
    labels.set("seat" + s.seat, { x: lay.label.x, y: 0.9, z: lay.label.z }, html, cls);
  }
  function betLabel(seat, amount) {
    const lay = seats[seat];
    if (amount > 0) labels.set("bet" + seat, { x: lay.bet.x, y: 0.2, z: lay.bet.z + 1.05 }, fmt(amount), "chips");
    else labels.remove("bet" + seat);
  }
  function potLabel(amount) {
    if (amount > 0) labels.set("pot", { x: L().pot.x, y: 0.2, z: L().pot.z + 1.7 }, `<small>pot</small> ${fmt(amount)}`, "chips pot");
    else labels.remove("pot");
  }

  // Make the stage match a scene, with no animation.
  function applyScene(scene) {
    const turned = table.setOrientation(stage.portrait);
    if (turned || seats.length !== scene.n) {
      seats = table.setSeats(scene.n);
      cards.setLayout(L());
      labels.clear();
      rig.refit();
    }
    current = structuredClone(scene);
    if (scene.dealer >= 0) table.moveButton(scene.dealer, false);
    cards.clear();
    chips.clear();
    const lit = new Set(scene.highlight?.cards?.map(cardKey) || []);
    const litIds = [],
      allIds = [];
    for (const s of scene.seats) {
      const lay = seats[s.seat];
      chips.setStack("stack" + s.seat, s.stack, lay.stack);
      chips.setStack("bet" + s.seat, s.bet, lay.bet);
      betLabel(s.seat, s.bet);
      seatLabel(s);
      if (!s.hasCards) continue;
      for (let i = 0; i < 2; i++) {
        const id = holeId(s.seat, i);
        cards.place(id, s.cards?.[i], s.cards && s.seat !== 0 ? lay.shown[i] : lay.cards[i], { faceUp: !!s.cards });
        allIds.push(id);
        if (s.cards && lit.has(cardKey(s.cards[i]))) litIds.push(id);
      }
    }
    scene.board.forEach((c, i) => {
      cards.place("b" + i, c, L().board[i], { faceUp: true });
      allIds.push("b" + i);
      if (lit.has(cardKey(c))) litIds.push("b" + i);
    });
    chips.setStack("pot", scene.pot, L().pot);
    potLabel(scene.pot);
    if (lit.size) {
      cards.lift(litIds, true);
      cards.dim(
        allIds.filter((id) => !litIds.includes(id)),
        true
      );
    }
    const actor = scene.seats.find((s) => s.acting);
    fx.spotlight(
      actor
        ? seats[actor.seat].cards[0] && {
            x: (seats[actor.seat].cards[0].x + seats[actor.seat].cards[1].x) / 2,
            z: (seats[actor.seat].cards[0].z + seats[actor.seat].cards[1].z) / 2,
          }
        : null
    );
  }

  let current = null; // the director's running picture of the table while it acts events out
  const seatOf = (i) => current.seats[i];
  const spotFor = (seat) => ({ x: (seats[seat].cards[0].x + seats[seat].cards[1].x) / 2, z: (seats[seat].cards[0].z + seats[seat].cards[1].z) / 2 });

  async function sweepBets() {
    const keys = current.seats.filter((s) => s.bet > 0).map((s) => "bet" + s.seat);
    if (!keys.length) return;
    for (const s of current.seats) {
      current.pot += s.bet;
      s.bet = 0;
      betLabel(s.seat, 0);
    }
    hooks.sound("chips");
    await chips.merge(keys, "pot", L().pot, 0.5);
    potLabel(current.pot);
  }

  const acts = {
    async handStart(e, state) {
      cards.clear();
      chips.clear();
      labels.remove("pot");
      current = sceneFromState(state, { acting: false });
      current.board = [];
      current.pot = 0;
      for (const s of current.seats) {
        const p = state.players[s.seat];
        Object.assign(s, { stack: p.startStack, bet: 0, folded: false, allIn: false, hasCards: false, acting: false });
        chips.setStack("stack" + s.seat, s.stack, seats[s.seat].stack);
        betLabel(s.seat, 0);
        seatLabel(s);
      }
      fx.spotlight(null);
      rig.to("deal", 0.7);
      await table.moveButton(e.dealer, true);
    },
    async post(e) {
      const s = seatOf(e.seat);
      s.stack -= e.amount;
      s.bet += e.amount;
      seatLabel(s, e.kind === "sb" ? "small blind" : "big blind");
      hooks.sound("chip");
      await chips.move("stack" + e.seat, "bet" + e.seat, e.amount, seats[e.seat].bet, 0.3);
      betLabel(e.seat, s.bet);
    },
    async deal(e) {
      const n = current.n;
      const order = Array.from({ length: n }, (_, k) => (current.dealer + 1 + k) % n);
      const flights = [];
      let k = 0;
      for (let round = 0; round < 2; round++)
        for (const seat of order) {
          const card = seat === 0 ? e.hands[0][round] : null;
          flights.push(cards.deal(holeId(seat, round), card, seats[seat].cards[round], { delay: k++ * 0.07 }));
          current.seats[seat].hasCards = true;
        }
      hooks.sound("deal");
      await Promise.all(flights);
      await Promise.all([cards.flip("h0a", true), cards.flip("h0b", true)]);
      rig.to("table", 0.8);
    },
    async action(e) {
      const s = seatOf(e.seat);
      fx.spotlight(spotFor(e.seat));
      if (e.kind === "fold") {
        s.folded = true;
        s.hasCards = false;
        seatLabel(s, "fold");
        hooks.sound("fold");
        await Promise.all([cards.muck(holeId(e.seat, 0)), cards.muck(holeId(e.seat, 1))]);
        return;
      }
      if (e.kind === "check") {
        seatLabel(s, "check");
        hooks.sound("check");
        await stage.anim.wait(0.28);
        return;
      }
      s.stack -= e.added;
      s.bet = e.to;
      s.allIn = e.allIn;
      const words = e.allIn
        ? "all-in " + fmt(e.to)
        : e.kind === "call"
          ? "call " + fmt(e.added)
          : e.kind === "bet"
            ? "bet " + fmt(e.to)
            : "raise to " + fmt(e.to);
      seatLabel(s, words);
      hooks.sound(e.allIn ? "allin" : "chip");
      await chips.move("stack" + e.seat, "bet" + e.seat, e.added, seats[e.seat].bet, 0.36);
      betLabel(e.seat, s.bet);
    },
    async refund(e) {
      const s = seatOf(e.seat);
      s.bet -= e.amount;
      s.stack += e.amount;
      await chips.move("bet" + e.seat, "stack" + e.seat, e.amount, seats[e.seat].stack, 0.36);
      betLabel(e.seat, s.bet);
      seatLabel(s);
    },
    async street(e) {
      await sweepBets();
      for (const s of current.seats) if (!s.folded) seatLabel(s);
      fx.spotlight(null);
      if (e.street === 1) rig.to("board", 0.9);
      const first = current.board.length;
      const slots = L().board;
      hooks.sound("deal");
      await Promise.all(e.cards.map((c, i) => cards.deal("b" + (first + i), c, slots[first + i], { delay: i * 0.09 })));
      await Promise.all(e.cards.map((c, i) => cards.flip("b" + (first + i), true)));
      current.board.push(...e.cards);
      labels.set("street", { x: 0, y: 0.2, z: L().board[0].z + 1.75 }, STREET_NAMES[e.street], "street");
      await stage.anim.wait(0.25);
      if (e.street === 1) rig.to("table", 0.9);
    },
    async showdown(e, state, rest) {
      await sweepBets();
      fx.spotlight(null);
      rig.to("showdown", 0.9);
      hooks.sound("flip");
      // Opponents turn their cards over and square them up to face the hero, so they can be read.
      const others = e.hands.filter((h) => h.seat !== 0);
      await Promise.all(others.flatMap((h) => [0, 1].map((i) => cards.flip(holeId(h.seat, i), true, h.cards[i]))));
      await Promise.all(others.flatMap((h) => [0, 1].map((i) => cards.moveTo(holeId(h.seat, i), seats[h.seat].shown[i], 0.3))));
      for (const h of e.hands) seatLabel(seatOf(h.seat), h.desc);
      await stage.anim.wait(0.45);
      // Light the five cards that win the main pot; everything else steps back.
      const main = rest.find((x) => x.type === "award");
      const winners = e.hands.filter((h) => main?.seats.includes(h.seat));
      const lit = new Set(winners.flatMap((h) => h.best5.map(cardKey)));
      const idOf = new Map();
      for (const h of e.hands) h.cards.forEach((c, i) => idOf.set(holeId(h.seat, i), cardKey(c)));
      current.board.forEach((c, i) => idOf.set("b" + i, cardKey(c)));
      const winnerSeats = new Set(winners.map((h) => h.seat));
      const litIds = [...idOf].filter(([id, key]) => lit.has(key) && (id[0] === "b" || winnerSeats.has(+id.slice(1, -1)))).map(([id]) => id);
      cards.dim(
        cards.ids().filter((id) => !litIds.includes(id)),
        true
      );
      await cards.lift(litIds, true);
      await stage.anim.wait(0.7);
    },
    async award(e) {
      await sweepBets();
      const shares = Object.entries(e.shares);
      hooks.sound("win");
      await Promise.all(
        shares.map(([seat, amount]) => {
          const s = seatOf(+seat);
          s.stack += amount;
          current.pot -= amount;
          if (e.contested || shares.length > 1 || amount > 3) fx.burst(seats[+seat].stack);
          seatLabel(s, "+" + fmt(amount));
          return chips.move("pot", "stack" + seat, amount, seats[+seat].stack, 0.7);
        })
      );
      potLabel(current.pot);
    },
    async handEnd() {
      labels.remove("street");
      await stage.anim.wait(0.5);
    },
  };

  // When the screen turns, the table turns with it: rebuild from the director's running picture.
  stage.onResized(() => {
    if (current && table.layout.portrait !== stage.portrait) applyScene(current);
    else rig.refit();
  });

  return {
    applyScene,
    on: (name, fn) => (hooks[name] = fn),
    setSpeed(x) {
      speed = x;
      stage.anim.speed = x;
    },
    get speed() {
      return speed;
    },
    skip: () => stage.anim.skip(),
    // Act out a batch of engine events. `state` is the engine state after them.
    async play(events, state, { reveal = [] } = {}) {
      if (seats.length !== state.config.n) applyScene(sceneFromState(state, { acting: false }));
      for (let i = 0; i < events.length; i++) await acts[events[i].type]?.(events[i], state, events.slice(i + 1));
      if (!state.handOver) applyScene(sceneFromState(state, { reveal }));
    },
  };
}

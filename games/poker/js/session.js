import * as core from "./core/index.js";

// The hand loop. It owns the engine state, the reads and the bots, asks the coach about the hero's spots, has
// the director act everything out, and publishes one plain `view` through the store for the panels to draw.
// Panels send intents back: act(), resume(), next(). Nothing in here touches the DOM or three.js.
const STOP = Symbol("stop");
const BOT_THINK = 0.55; // seconds a bot appears to think (scaled by the table speed, skippable)

export function createSession({ store, director, coach, rng }) {
  let state = null,
    reads = null,
    styles = null,
    level = "guided",
    running = false,
    looping = null,
    waiter = null, // { resolve, reject } of whatever the loop is waiting on the player for
    hand = null, // { snap, decisions: [], events: [] }
    raiseTouched = false;
  const sessionNet = { guided: 0, silent: 0 };

  const settings = () => store.settings;
  const fromPlayer = () =>
    new Promise((resolve, reject) => {
      waiter = { resolve, reject };
    });
  const answer = (value) => {
    const w = waiter;
    waiter = null;
    w?.resolve(value);
  };

  function newGame() {
    const n = settings().players;
    state = core.createGame({ n });
    reads = core.createReads(n);
    styles = core.assignStyles(n, rng);
    hand = null;
  }

  // Everything the panels need to know about the table right now (never anything the hero should not see).
  function publish(patch = {}) {
    const hero = state.players[0];
    store.set({
      level,
      handNo: state.handNo,
      net: sessionNet[level],
      n: state.config.n,
      street: state.street,
      board: state.board.slice(),
      pot: core.pot(state),
      hero: { cards: hero.cards.slice(), pos: state.dealer >= 0 ? core.seatPos(state, 0) : "", stack: hero.stack, folded: hero.folded },
      opponents: state.players
        .slice(1)
        .map((p) => ({
          seat: p.id,
          name: p.name,
          pos: core.seatPos(state, p.id),
          folded: p.folded,
          read: core.readOf(reads, p.id),
          label: core.readLabel(reads, p.id),
        })),
      decisions: level === "silent" && !state.handOver ? [] : hand ? hand.decisions : [],
      ...patch,
    });
  }

  const sizeTo = (legal, frac) => core.clampRaise(legal, state.players[0].bet + legal.toCall + frac * (legal.pot + legal.toCall));

  async function heroTurn() {
    const legal = core.legalActions(state);
    const guided = level === "guided";
    raiseTouched = false;
    publish({ phase: "hero", legal, analysis: null, analysing: guided, raiseTo: legal.canRaise ? sizeTo(legal, 0.66) : 0, lastGrade: null });
    const seed = Math.floor(rng() * 2 ** 31);
    const analysed = coach.analyze(state, reads, { seed, iters: 6000 }).then((a) => {
      if (guided && store.view.phase === "hero") {
        const pick = settings().showPick && a.rec.action === "raise" && a.rec.size && legal.canRaise && !raiseTouched;
        publish({ analysis: a, analysing: false, ...(pick ? { raiseTo: core.clampRaise(legal, a.rec.size) } : {}) });
      }
      return a;
    });
    analysed.catch(() => {}); // a cancelled request is handled where it is awaited, below

    const choice = await fromPlayer(); // { type, to }
    publish({ phase: "bots", legal: null });
    const a = await analysed;

    // Grade it. Checking and calling are the same button when there is nothing to call.
    let grade = core.grade(a.rec, choice.type);
    if (grade === "mistake" && ["check", "call"].includes(a.rec.action) && ["check", "call"].includes(choice.type)) grade = "correct";
    let evChosen = null;
    if (a.ev)
      evChosen =
        choice.type === "fold" ? 0 : choice.type === "raise" ? (a.samples ? core.evOfRaise(a.ctx, a.samples, choice.to).ev : null) : a.ev.passive;
    const lesson = core.lessonFor(a);
    const decision = {
      grade,
      action: choice.type,
      raiseTo: choice.type === "raise" ? choice.to : 0,
      street: state.street,
      handNo: state.handNo,
      category: a.category,
      evChosen,
      lesson,
      index: state.actions.length,
      analysis: { ...a, samples: null },
    };
    hand.decisions.push(decision);
    store.recordDecision(level, {
      grade,
      action: decision.action,
      raiseTo: decision.raiseTo,
      street: decision.street,
      handNo: decision.handNo,
      category: a.category,
      recommended: a.rec.action,
      lessonId: lesson.lessonId,
      key: a.key,
      pos: a.posn,
    });

    const events = core.observeAll(reads, core.apply(state, { seat: 0, type: choice.type, ...(choice.type === "raise" ? { to: choice.to } : {}) }));
    hand.events.push(...events);
    // The hero's own action is acted out at once; whatever follows it (the next street, a showdown) waits for any pause.
    await director.play(events.slice(0, 1), state, { partial: events.length > 1 });
    const pauses = guided && !state.handOver && (settings().pause === "always" || (settings().pause === "mistake" && grade !== "correct"));
    if (pauses) {
      publish({ phase: "feedback", lastGrade: decision, analysis: a });
      await fromPlayer();
      publish({ phase: "bots", analysis: null });
    } else if (guided) publish({ lastGrade: decision, analysis: null });
    if (events.length > 1) await director.play(events.slice(1), state);
  }

  async function botTurn() {
    publish({ phase: "bots" });
    await director.wait(BOT_THINK);
    if (!running) throw STOP;
    const seat = state.toAct;
    const events = core.observeAll(reads, core.apply(state, core.botDecide(state, reads, seat, core.STYLES[styles[seat]], rng)));
    hand.events.push(...events);
    await director.play(events, state);
  }

  async function loop() {
    try {
      for (;;) {
        if (!state || (state.handOver && state.config.n !== settings().players)) newGame();
        if (state.handOver) {
          const events = core.observeAll(reads, core.startHand(state, rng));
          hand = { snap: core.snapshot(state), decisions: [], events: events.slice() };
          publish({ phase: "dealing", result: null, analysis: null, lastGrade: null, legal: null });
          await director.play(events, state);
        } else director.show(state); // coming back to a hand that was left half-played
        while (!state.handOver) {
          if (!running) throw STOP;
          if (state.toAct === 0) await heroTurn();
          else await botTurn();
        }
        const net = state.result.net[0];
        sessionNet[level] += net;
        store.recordHand(level, net);
        const end = hand.events.find((e) => e.type === "showdown");
        publish({
          phase: "handOver",
          analysis: null,
          legal: null,
          result: {
            net,
            showdown: state.result.showdown,
            awards: state.result.awards,
            shown: end ? end.hands : [],
            heroFolded: state.players[0].folded,
          },
        });
        await fromPlayer();
      }
    } catch (e) {
      if (e !== STOP && !e?.cancelled) throw e;
    } finally {
      looping = null;
    }
  }

  return {
    get state() {
      return state;
    },
    get hand() {
      return hand;
    },
    // Sit down (or come back). Starting while already running only changes the coach level.
    start(toLevel = level) {
      level = toLevel;
      if (looping) return looping;
      running = true;
      director.setSpeed(settings().animations ? settings().speed : 50);
      return (looping = loop());
    },
    // Leave the table. The hand stays exactly where it is and carries on when start() is called again.
    stop() {
      running = false;
      coach.cancelAll();
      const w = waiter;
      waiter = null;
      w?.reject(STOP);
      director.skip();
      return looping || Promise.resolve();
    },
    setLevel(toLevel) {
      level = toLevel;
      if (state) publish();
    },
    setRaise(to) {
      if (store.view.phase !== "hero" || !store.view.legal?.canRaise) return;
      raiseTouched = true;
      store.set({ raiseTo: core.clampRaise(store.view.legal, to) });
    },
    sizeFor: (frac) => (store.view.legal ? sizeTo(store.view.legal, frac) : 0),
    act(type, to) {
      const legal = store.view.legal;
      if (store.view.phase !== "hero" || !legal || !waiter) return false;
      if (type === "call" && legal.canCheck) type = "check";
      if (type === "check" && !legal.canCheck) type = "call";
      if (type === "raise" && !legal.canRaise) return false;
      answer({ type, to: type === "raise" ? core.clampRaise(legal, to ?? store.view.raiseTo) : 0 });
      return true;
    },
    resume() {
      if (store.view.phase === "feedback") answer();
    },
    next() {
      if (store.view.phase === "handOver") answer();
    },
  };
}

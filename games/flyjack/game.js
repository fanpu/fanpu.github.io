/* FlyJack blackjack engine, ported from server/app.py (class Game, card, hand_value) so the page can run
   without a backend. Blackjack-v1 rules: infinite deck, natural=False, dealer stands on soft 17, one unit
   per hand. Seeded so ?seed= reproduces a session. */
(function () {
  'use strict';
  const RANKS = { 1: 'A', 11: 'J', 12: 'Q', 13: 'K' };
  const SUITS = ['♠', '♥', '♦', '♣'];
  const START_BANKROLL = 100;

  // mulberry32: the server used np.random.default_rng; any decent PRNG is equivalent for a demo.
  function rng32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function handValue(cards) {
    let s = 0, ace = false;
    for (const c of cards) { s += c.value; if (c.value === 1) ace = true; }
    const soft = ace && s + 10 <= 21;
    return { total: soft ? s + 10 : s, soft };
  }

  class Game {
    /** @param obsIndex Map from "v,d,u" to the observation index, built from cfg.policy.observations. */
    constructor(seed, obsIndex) {
      this.seed = seed >>> 0;
      this.rand = rng32(this.seed);
      this.obsIndex = obsIndex;
      this.bankroll = START_BANKROLL;
      this.record = { win: 0, draw: 0, loss: 0 };
      this.hands = 0;
      this.player = [];
      this.dealer = [];
      this.phase = 'idle';
      this.outcome = null;
      this.decisionId = null;
    }

    int(lo, hi) { return lo + Math.floor(this.rand() * (hi - lo)); }

    card() {
      const rank = this.int(1, 14);
      return { rank: RANKS[rank] || String(rank), suit: SUITS[this.int(0, 4)], value: Math.min(rank, 10) };
    }

    deal() {
      this.player = [this.card(), this.card()];
      this.dealer = [this.card(), this.card()];
      this.phase = 'decide';
      this.outcome = null;
      this.decisionId = null;
      this.hands++;
    }

    /** Index into the 280 reachable observations, matching flyjack.bj.OBS_INDEX. */
    observation() {
      const { total, soft } = handValue(this.player);
      return this.obsIndex.get(`${total},${this.dealer[0].value},${soft ? 1 : 0}`);
    }

    hit() {
      this.player.push(this.card());
      if (handValue(this.player).total > 21) this.finish(-1);
    }

    stick() {
      while (handValue(this.dealer).total < 17) this.dealer.push(this.card());
      const pv = handValue(this.player).total, dv = handValue(this.dealer).total;
      this.finish(dv > 21 || pv > dv ? 1 : pv === dv ? 0 : -1);
    }

    finish(reward) {
      this.phase = 'done';
      this.outcome = reward;
      this.bankroll += reward;
      this.record[{ 1: 'win', 0: 'draw', '-1': 'loss' }[reward]]++;
    }

    /** Same shape as server/app.py Game.state(): the dealer's hole card stays hidden until the hand is done. */
    state() {
      const done = this.phase === 'done';
      const p = this.player.length ? handValue(this.player) : null;
      return {
        player_cards: this.player, player_value: p ? p.total : null, usable_ace: p ? p.soft : null,
        dealer_cards: done ? this.dealer : this.dealer.slice(0, 1),
        dealer_value: done ? handValue(this.dealer).total : null,
        phase: this.phase, outcome: this.outcome, bankroll: this.bankroll,
        record: this.record, hands: this.hands, decision_id: this.decisionId,
      };
    }
  }

  window.FlyGame = { Game, handValue, rng32, START_BANKROLL };
})();

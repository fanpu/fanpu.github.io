/* FlyJack stage app: runs each round (deal -> think -> act -> ... -> dealer -> result) through FlyApi, drives the
   3D stage, the HUD and the Explore drawer. The page only animates cards the backend actually dealt.
   FlyApi is either the FastAPI server (web/api.js server mode) or the in-browser static bundle. */
(function () {
  'use strict';
  const $ = (s) => document.querySelector(s);
  const params = new URLSearchParams(location.search);
  const SLOWMO = 20;
  const dbg = window.__flyjack = { phases: [], hands: [], errors: [], ready: false };

  let cfg = null, stage = null, gameId = null, busy = false, playback = null, handNo = 0, lastState = null;

  const api = window.FlyApi;
  const getJSON = api.getJSON, buffer = api.buffer;
  const status = (t) => { $('#status').textContent = t; };

  function phase(name) {
    dbg.phases.push({ hand: handNo, phase: name, t: performance.now() });
    $('#phaseChip').textContent = { deal: 'Dealing', think: 'The fly is thinking', act: 'The fly acts', dealer: 'Dealer’s turn', result: 'Result', idle: 'Ready' }[name] || name;
    document.body.dataset.phase = name;
  }

  const handValue = window.FlyGame.handValue;

  function notice(text, withNewGame = true) {
    $('#noticeText').textContent = text;
    $('#newGameBtn').hidden = !withNewGame;
    $('#notice').hidden = false;
  }

  // ------------------------------------------------------------------ HUD
  function setBar(el, v) {
    const w = Math.min(1, Math.abs(v)) * 50;
    el.style.width = `${w}%`;
    el.style.left = v >= 0 ? '50%' : `${50 - w}%`;
    el.className = `qbar ${v >= 0 ? 'pos' : 'neg'}`;
  }
  function clearDecision() {
    ['#qStick', '#qHit'].forEach((s) => { $(s).style.width = '0'; $(s).style.left = '50%'; });
    ['#qOptStick', '#qOptHit'].forEach((s) => { $(s).style.display = 'none'; });
    $('#qStickVal').textContent = '—';
    $('#qHitVal').textContent = '—';
    $('#verdict').innerHTML = '<span class="wait">Deciding…</span>';
  }
  function showDecision(d) {
    if (d.q) {
      setBar($('#qStick'), d.q.stick);
      setBar($('#qHit'), d.q.hit);
      $('#qStickVal').textContent = d.q.stick.toFixed(3);
      $('#qHitVal').textContent = d.q.hit.toFixed(3);
    }
    [['#qOptStick', d.q_optimal.stick], ['#qOptHit', d.q_optimal.hit]].forEach(([s, v]) => {
      $(s).style.display = 'block';
      $(s).style.left = `calc(${50 + 50 * Math.max(-1, Math.min(1, v))}% - 1px)`;
    });
    const match = d.action === d.optimal;
    $('#verdict').innerHTML = `Fly: <b>${d.action.toUpperCase()}</b> · optimal: <b>${d.optimal.toUpperCase()}</b> <span class="badge ${match ? 'good' : 'bad'}">${match ? '✓ matches' : '✗ differs'}</span>`;
  }
  function updateBank(s) {
    $('#bankroll').textContent = s.bankroll;
    $('#wins').textContent = s.record.win;
    $('#draws').textContent = s.record.draw;
    $('#losses').textContent = s.record.loss;
  }
  function setTotals(playerCards, dealerCards, dealerDone) {
    const p = handValue(playerCards);
    $('#lblPlayer').textContent = playerCards.length ? `Fly ${p.total}${p.soft ? ' soft' : ''}${p.total > 21 ? ' · bust' : ''}` : '';
    const d = handValue(dealerCards);
    $('#lblDealer').textContent = dealerCards.length ? `Dealer ${dealerDone ? d.total : 'shows ' + (dealerCards[0].value === 1 ? 'A' : dealerCards[0].value)}${dealerDone && d.total > 21 ? ' · bust' : ''}` : '';
  }

  function positionLabels() {
    const a = stage.anchors();
    const place = (el, p, show = true) => {
      el.style.transform = `translate(${p.x}px, ${p.y}px) translate(-50%, -100%)`;
      el.style.opacity = show && p.visible && el.textContent ? 1 : 0;
    };
    place($('#lblPlayer'), a.player);
    place($('#lblDealer'), a.dealer);
    place($('#lblBrain'), a.brain, stage.brain.visible && stage.brainMat.uniforms.uOpacity.value > 0.2);
    $('#fps').textContent = `${stage.fps} fps${stage.quality ? ` · quality ${['high', 'medium', 'low'][stage.quality]}` : ''}`;
    dbg.fps = stage.fps;
  }

  // ------------------------------------------------------------------ round phases
  async function think(decisionId) {
    phase('think');
    const [d, buf] = await Promise.all([api.decision(decisionId), api.frames(decisionId)]);
    stage.loadFrames(buf);
    FlyLearn.beginDecision(d);
    $('#trialChip').textContent = `cached simulation · trial ${d.trial}`;
    clearDecision();
    status(`Replaying trial ${d.trial}: ${d.n_spikes.toLocaleString('en-US')} spikes in 300 ms of simulated brain time`);
    $('#counters').hidden = false;
    $('#caption').hidden = false;
    await Promise.all([stage.shot('think', 1.1), stage.showBrain(true, 0.9)]);
    await new Promise((resolve) => { playback = { t: 0, T: cfg.T_ms, resolve }; });
    FlyLearn.endDecision(d);
    showDecision(d);
    await stage.anim.wait(1.4);
    $('#counters').hidden = true;
    await Promise.all([stage.showBrain(false, 0.7), stage.shot('table', 1.0)]);
    return d;
  }

  async function round() {
    if (busy) return;
    busy = true;
    $('#dealBtn').disabled = true;
    handNo++;
    try {
      phase('deal');
      $('#caption').hidden = true;
      await stage.clearTable();
      setTotals([], [], false);
      let s = await api.deal();
      gameId = s.game_id;
      lastState = s;
      await stage.placeBet(1);
      await stage.dealCard('player', s.player_cards[0]);
      setTotals(s.player_cards.slice(0, 1), [], false);
      await stage.dealCard('dealer', s.dealer_cards[0]);
      setTotals(s.player_cards.slice(0, 1), s.dealer_cards, false);
      await stage.dealCard('player', s.player_cards[1]);
      setTotals(s.player_cards, s.dealer_cards, false);
      await stage.dealCard('dealer', null, false);
      while (s.phase === 'decide') {
        const d = await think(s.decision_id);
        phase('act');
        status(d.action === 'hit' ? 'The fly taps the table: hit' : 'The fly sweeps its leg: stand');
        const clip = stage.playClip(d.action === 'hit' ? 'hit' : 'stand');
        const next = await api.act();
        await clip;
        if (d.action === 'hit') {
          await stage.dealCard('player', next.player_cards[next.player_cards.length - 1]);
          setTotals(next.player_cards, next.phase === 'done' ? next.dealer_cards.slice(0, 1) : next.dealer_cards, false);
        }
        s = next;
        lastState = s;
      }
      phase('dealer');
      const bust = handValue(s.player_cards).total > 21;
      status(bust ? 'The fly busts. The dealer reveals the hole card.' : 'Dealer’s turn: reveal, then draw to 17');
      await stage.shot('dealer', 1.0);
      await stage.revealHole(s.dealer_cards[1]);
      setTotals(s.player_cards, s.dealer_cards.slice(0, 2), true);
      for (let i = 2; i < s.dealer_cards.length; i++) {
        await stage.anim.wait(0.35);
        await stage.dealCard('dealer', s.dealer_cards[i]);
        setTotals(s.player_cards, s.dealer_cards.slice(0, i + 1), true);
      }
      await stage.anim.wait(0.8);                     // let the dealer's final total register before paying out
      phase('result');
      const outcome = s.outcome;
      status(outcome > 0 ? 'The fly wins!' : outcome < 0 ? (bust ? 'Bust: the dealer takes the bet' : 'The dealer wins') : 'Push: the bet is returned');
      $('#resultChip').textContent = outcome > 0 ? 'Fly wins +1' : outcome < 0 ? 'Dealer wins −1' : 'Push';
      $('#resultChip').className = `chip-result ${outcome > 0 ? 'good' : outcome < 0 ? 'bad' : ''}`;
      $('#resultChip').hidden = false;
      if (outcome > 0) FlyAudio.win(); else if (outcome < 0) FlyAudio.lose(); else FlyAudio.push();
      await stage.shot('result', 0.9);
      await Promise.all([stage.settle(outcome, s.bankroll),
                         stage.playClip(outcome > 0 ? 'win' : outcome < 0 ? 'lose' : 'idle')]);
      updateBank(s);
      dbg.hands.push({ hand: handNo, player_cards: s.player_cards, dealer_cards: s.dealer_cards, outcome, bankroll: s.bankroll,
                       record: s.record, table: stage.debugCards() });
      await stage.anim.wait(1.0);
      $('#resultChip').hidden = true;
      phase('idle');
      status('Deal the next hand, or turn on auto-play.');
    } catch (e) {
      dbg.errors.push(String(e));
      notice(api.mode === 'server' && (e.status === 404 || e.status === 409)
        ? 'The server lost this game (it may have restarted). Start a new game.'
        : `Something went wrong: ${e.message}`);
    } finally {
      busy = false;
      $('#dealBtn').disabled = false;
      scheduleAuto();
    }
  }

  function scheduleAuto() {
    if (!$('#autoplay').checked || busy || !$('#notice').hidden) return;
    setTimeout(() => {
      if ($('#autoplay').checked && !busy && !document.hidden) round();
    }, 1500 / stage.anim.speed);
  }

  async function newGame() {
    $('#notice').hidden = true;
    gameId = null;
    api.newGame();
    await stage.clearTable();
    stage.rebuildBank(100);
    updateBank({ bankroll: 100, record: { win: 0, draw: 0, loss: 0 } });
    setTotals([], [], false);
    phase('idle');
    status('New game. Deal a hand.');
  }

  // ------------------------------------------------------------------ init
  function webglAvailable() {
    try { const c = document.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')); } catch (e) { return false; }
  }

  async function init() {
    if (!webglAvailable() || !window.THREE) {
      notice('This demo needs WebGL, which this browser does not provide.', false);
      return;
    }
    if (api.mode === 'static' && typeof DecompressionStream === 'undefined') {
      // The spike frames ship gzipped; without DecompressionStream there is no way to read them.
      notice('This demo needs DecompressionStream (Chrome 80+, Firefox 113+, Safari 16.4+).', false);
      return;
    }
    try {
      cfg = await api.config();
      const [neuronsBuf, meshJson, meshBuf, clipsJson, clipsBuf] = await Promise.all([
        buffer(api.asset('neurons.bin')), getJSON(api.stageAsset('fly_mesh.json')), buffer(api.stageAsset('fly_mesh.bin')),
        getJSON(api.stageAsset('fly_clips.json')), buffer(api.stageAsset('fly_clips.bin'))]);
      stage = new FlyStage.Stage($('#stage'));
      await stage.init({ cfg, neuronsBuf, meshJson, meshBuf, clipsJson, clipsBuf });
    } catch (e) {
      dbg.errors.push(String(e));
      notice(`The 3D stage could not load: ${e.message}`, false);
      return;
    }
    stage.onSound = (s) => { if (s === 'slide') FlyAudio.cardSlide(); else if (s === 'flip') FlyAudio.cardFlip(); else if (s === 'chip') FlyAudio.chip(1); };
    stage.onFrames = (idx, k) => FlyLearn.onFrames(idx, k);
    stage.onTick = (dt) => {
      if (playback) {
        playback.t = Math.min(playback.T, playback.t + dt * stage.anim.speed / SLOWMO);
        stage.applyBrainTo(playback.t);
        FlyLearn.update(playback.t);
        $('#clock').textContent = `${playback.t.toFixed(0)} / ${playback.T} ms brain time · ${SLOWMO / stage.anim.speed}× slow motion`;
        if (playback.t >= playback.T) { const p = playback; playback = null; p.resolve(); }
      }
      positionLabels();
    };
    FlyLearn.init(cfg, { stage });
    dbg.learn = FlyLearn;
    dbg.cards = () => stage.debugCards();
    dbg.stage = stage;
    $('#visualNote').textContent = cfg.visual_note || '';
    $('#modelChip').innerHTML = cfg.trained
      ? `Readout: ${cfg.agent.readout} (${cfg.agent.n_features.toLocaleString('en-US')} neurons), exact return ${cfg.agent.exact_return.toFixed(3)} per hand`
      : 'Untrained readout';
    const anim = Number(params.get('anim')) || 1;
    stage.anim.speed = anim;
    $('#animSpeed').value = String(anim);
    $('#animSpeed').addEventListener('change', () => { stage.anim.speed = Number($('#animSpeed').value); });
    $('#skipBtn').addEventListener('click', () => { if (playback) playback.t = playback.T; stage.anim.skip(); });
    const soundLabel = () => { $('#soundBtn').textContent = FlyAudio.enabled ? 'Sound on' : 'Sound off'; $('#soundBtn').setAttribute('aria-pressed', FlyAudio.enabled); };
    if (params.get('sound') === '0') FlyAudio.setEnabled(false);
    soundLabel();
    $('#soundBtn').addEventListener('click', () => { FlyAudio.setEnabled(!FlyAudio.enabled); soundLabel(); });
    $('#dealBtn').addEventListener('click', round);
    $('#autoplay').addEventListener('change', scheduleAuto);
    $('#newGameBtn').addEventListener('click', newGame);
    $('#drawerBtn').addEventListener('click', () => {
      const open = $('#drawer').classList.toggle('open');
      $('#drawerBtn').setAttribute('aria-expanded', open);
    });
    $('#drawerClose').addEventListener('click', () => { $('#drawer').classList.remove('open'); $('#drawerBtn').setAttribute('aria-expanded', false); });
    document.querySelectorAll('#tabs button').forEach((b) => b.addEventListener('click', () => {
      document.querySelectorAll('#tabs button').forEach((x) => x.setAttribute('aria-selected', x === b));
      document.querySelectorAll('.tabpanel').forEach((p) => { p.hidden = p.id !== `tab-${b.dataset.tab}`; });
      if (b.dataset.tab === 'circuit' && FlyLearn.decision) FlyLearn.renderCircuit(FlyLearn.decision.circuit);
    }));
    document.querySelectorAll('#legendGroups input').forEach((cb) => cb.addEventListener('change', () => stage.setGroupVisible(+cb.dataset.group, cb.checked)));
    document.addEventListener('visibilitychange', () => { if (!document.hidden) scheduleAuto(); });
    await stage.shot('table', 0);
    await new Promise((resolve) => {                // ready only once the render loop has drawn a few frames
      const check = () => ((stage.ticks || 0) > 2 ? resolve() : requestAnimationFrame(check));
      check();
    });
    phase('idle');
    status('Deal a hand to start.');
    $('#dealBtn').disabled = false;
    dbg.ready = true;
    if (params.get('tour') !== '0') FlyLearn.startTour(false);
    if (params.get('autoplay') === '1') { $('#autoplay').checked = true; round(); }
  }

  init();
})();

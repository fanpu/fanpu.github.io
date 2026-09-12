/* FlyJack learning layers: guided tour, live narration of the decision trial, brain counters, strategy chart, science
   panel, glossary and fact cards. Every number shown comes from the server config or the decision being replayed. */
(function () {
  'use strict';
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const fmt = (n) => Number(n).toLocaleString('en-US');
  const DARK = { surface: '#1a1a19', ink: '#ffffff', ink2: '#c3c2b7', muted: '#898781', grid: '#2c2c2a', axis: '#383835' };
  const SERIES = {   // dark-mode categorical slots 1-4 (validated as adjacent line series)
    real: ['Connectome, real', '#3987e5'], shuffled: ['Connectome, shuffled', '#d95926'],
    rf_noisy: ['Random features + noise', '#199e70'], rf_clean: ['Random features, noise-free', '#c98500'],
  };

  const GLOSSARY = {
    'ORN': 'Olfactory receptor neuron: a smell sensor on the antenna. Each ORN type responds to particular odours and sends spikes to the antennal lobe.',
    'projection neuron': 'Carries smell information from the antennal lobe to higher centres such as the mushroom body and the lateral horn.',
    'Kenyon cell': 'The main neurons of the mushroom body. Their sparse activity patterns are thought to support odour learning.',
    'mushroom body': 'A paired brain structure that is the fly’s best-studied centre for learning and memory.',
    'MBON': 'Mushroom body output neuron: reads out Kenyon-cell activity. Dopamine-driven changes at Kenyon-cell-to-MBON synapses are a classic model of fly learning.',
    'dopamine neuron': 'Dopaminergic neurons (for example the PAM cluster) signal reward or punishment to the mushroom body in real flies.',
    'descending neuron': 'Carries commands from the brain down to the ventral nerve cord, which drives the legs and wings. Here their spike counts feed the decision readout.',
    'spike': 'A brief electrical pulse. This model uses leaky integrate-and-fire neurons: each adds up its inputs and fires when it crosses a threshold.',
    'connectome': 'A complete wiring diagram of a nervous system: every neuron and every synapse, here reconstructed from electron microscopy by FlyWire.',
    'Q-value': 'The readout’s estimate of the average winnings from an action (hit or stick) in this situation. The fly picks the larger one.',
    'basic strategy': 'The best hit or stick choice for every player total and dealer card, computed exactly by dynamic programming.',
    'random features': 'A control: replace the brain with the same number of fixed random nonlinear functions of the input. If the real connectome cannot beat that, its specific wiring is not buying anything.',
  };

  const L = {
    cfg: null, deps: null, milestones: [], shown: 0, counters: null, strategyMode: 'optimal', currentObs: null,
    circuit: null, decision: null, dnGroup: -1,

    init(cfg, deps) {
      this.cfg = cfg;
      this.deps = deps;
      this.dnGroup = cfg.neurons.groups.indexOf('DN');
      this.obsIndex = new Map(cfg.policy.observations.map((o, i) => [o.join(','), i]));
      this.initGlossary();
      this.renderHowItWorks();
      this.renderFacts();
      this.renderScience();
      this.renderStrategy();
      $('#tourBtn').addEventListener('click', () => this.startTour(true));
      document.querySelectorAll('#strategyModes button').forEach((b) => b.addEventListener('click', () => {
        this.strategyMode = b.dataset.mode;
        document.querySelectorAll('#strategyModes button').forEach((x) => x.setAttribute('aria-pressed', x === b));
        this.renderStrategy();
      }));
    },

    // ------------------------------------------------------------------ glossary
    term(word, label) { return `<span class="term" tabindex="0" data-term="${esc(word)}">${esc(label || word)}</span>`; },
    initGlossary() {
      const tip = $('#tooltip');
      const show = (el) => {
        const text = GLOSSARY[el.dataset.term];
        if (!text) return;
        tip.innerHTML = `<b>${esc(el.dataset.term)}</b><br>${esc(text)}`;
        tip.hidden = false;
        const r = el.getBoundingClientRect();
        const x = Math.min(window.innerWidth - tip.offsetWidth - 12, Math.max(12, r.left));
        const y = r.bottom + 8 + tip.offsetHeight > window.innerHeight ? r.top - tip.offsetHeight - 8 : r.bottom + 8;
        tip.style.transform = `translate(${x}px, ${y}px)`;
      };
      const hide = () => { tip.hidden = true; };
      document.addEventListener('mouseover', (e) => { const t = e.target.closest('.term'); if (t) show(t); });
      document.addEventListener('mouseout', (e) => { if (e.target.closest('.term')) hide(); });
      document.addEventListener('focusin', (e) => { const t = e.target.closest('.term'); if (t) show(t); });
      document.addEventListener('focusout', (e) => { if (e.target.closest('.term')) hide(); });
    },

    // ------------------------------------------------------------------ tour
    tourSteps() {
      const f = this.cfg.facts || {};
      return [
        { target: '#stageWrap', title: 'A fly at the blackjack table',
          text: `This is NeuroMechFly, a 3D model of a fruit fly, standing at a fly-sized table (cards 1.25 mm wide). Cards come from the shoe under Blackjack-v1 rules: no doubling, no splitting, and the dealer stands on 17.` },
        { target: '#inputChip', title: 'How the fly “sees” the cards',
          text: `The fly cannot read. Its hand is encoded as smell: three types of ${this.term('ORN', 'olfactory receptor neurons')} fire faster for a higher player total, a higher dealer card, and a usable ace. Visual input was tried first and did not reach the central brain.` },
        { target: '#stageWrap', title: 'Watching the brain decide',
          text: `While the fly decides, its brain appears above it, magnified about 10×. Each dot is one of ${fmt(f.neurons || 138639)} neurons from the FlyWire ${this.term('connectome')}. Dots flash when a neuron ${this.term('spike', 'spikes')} in a simulation of that exact wiring, replayed 20× slower than real time.` },
        { target: '#hud', title: 'Turning spikes into a choice',
          text: `Only a small readout learns. It weighs spike counts of ${this.term('descending neuron', 'descending neurons')} into a ${this.term('Q-value')} for hit and for stick, and the fly takes the larger. The leg tap or sweep afterwards is a scripted animation: this brain model has no nerve cord.` },
        { target: '#drawerBtn', title: 'Explore the experiment',
          text: 'Open Explore. “How it works” walks through the whole setup from cards to decision; the other tabs follow the signal through cell types, compare the fly’s strategy with perfect play, and show what the experiment found. Spoiler: the real connectome did not beat random features.' },
      ];
    },
    startTour(force = false) {
      let seen = false;
      try { seen = localStorage.getItem('flyjack.tour') === 'done'; } catch (e) { seen = false; }
      if (seen && !force) return;
      const steps = this.tourSteps();
      let i = 0;
      const box = $('#tour');
      const render = () => {
        const s = steps[i];
        $('#tourTitle').textContent = s.title;
        $('#tourText').innerHTML = s.text;
        $('#tourStep').textContent = `${i + 1} / ${steps.length}`;
        $('#tourBack').disabled = i === 0;
        $('#tourNext').textContent = i === steps.length - 1 ? 'Done' : 'Next';
        const el = $(s.target);
        const spot = $('#tourSpot');
        if (el) {
          const r = el.getBoundingClientRect();
          Object.assign(spot.style, { left: `${r.left - 6}px`, top: `${r.top - 6}px`, width: `${r.width + 12}px`, height: `${r.height + 12}px` });
          spot.hidden = false;
        } else spot.hidden = true;
      };
      const close = () => {
        box.hidden = true;
        try { localStorage.setItem('flyjack.tour', 'done'); } catch (e) { /* storage unavailable */ }
      };
      $('#tourNext').onclick = () => { if (i < steps.length - 1) { i++; render(); } else close(); };
      $('#tourBack').onclick = () => { if (i > 0) { i--; render(); } };
      $('#tourClose').onclick = close;
      box.hidden = false;
      render();
    },

    // ------------------------------------------------------------------ decision: narration and counters
    beginDecision(d) {
      this.decision = d;
      this.shown = 0;
      this.firstDN = null;
      const N = this.cfg.neurons.N;
      this.counters = { spikes: 0, active: new Uint8Array(N), nActive: 0, byGroup: new Array(this.cfg.neurons.groups.length).fill(0) };
      $('#captionList').innerHTML = '';
      this.milestones = this.buildMilestones(d);
      this.currentObs = d.obs;
      this.renderStrategy();
      this.renderCircuit(d.circuit);
      this.renderInput(d);
      this.updateCounters();
    },

    buildMilestones(d) {
      const fm = d.frame_ms;
      const groups = this.cfg.neurons.groups;
      const first = (n) => { const k = n.activity.findIndex((a) => a > 0); return k < 0 ? null : k * fm; };
      const nodes = d.circuit.nodes.map((n) => ({ ...n, t: first(n) })).filter((n) => n.t !== null);
      const out = [];
      const topTypes = (list, k = 2) => list.sort((a, b) => b.spikes - a.spikes).slice(0, k).map((n) => n.type).join(', ');
      const inputs = nodes.filter((n) => n.hop === 0);
      if (inputs.length) {
        const t = Math.min(...inputs.map((n) => n.t));
        const ch = this.cfg.channels || [];
        const rates = d.input_rates_hz.map((hz, i) => `${ch[i] ? ch[i].type.replace('ORN_', '') : 'ch' + i} ${Math.round(hz)} Hz`).join(', ');
        out.push({ t, text: `Smell receptor neurons (${this.term('ORN', 'ORNs')}) fire: ${esc(rates)}.` });
      }
      const hop1 = nodes.filter((n) => n.hop === 1);
      if (hop1.length) {
        out.push({ t: Math.min(...hop1.map((n) => n.t)),
                   text: `One synapse later, antennal-lobe neurons respond (${esc(topTypes(hop1))}), including ${this.term('projection neuron', 'projection neurons')}.` });
      }
      const kc = nodes.filter((n) => /^KC/.test(n.type));
      if (kc.length) {
        out.push({ t: Math.min(...kc.map((n) => n.t)),
                   text: `${this.term('Kenyon cell', 'Kenyon cells')} light up (${esc(topTypes(kc))}): the ${this.term('mushroom body')}, the fly’s centre for learning and memory.` });
      }
      const dan = nodes.filter((n) => /^(PAM|PPL)/.test(n.type));
      if (dan.length) {
        out.push({ t: Math.min(...dan.map((n) => n.t)),
                   text: `${this.term('dopamine neuron', 'Dopamine neurons')} fire (${esc(topTypes(dan))}). No learning happens here: the wiring is frozen.` });
      }
      return out.sort((a, b) => a.t - b.t).map((m) => ({ ...m, text: `<span class="t">${m.t.toFixed(0)} ms</span> ${m.text}` }));
    },

    onFrames(idx, k) {
      const c = this.counters;
      if (!c) return;
      c.spikes += idx.length;
      const groups = this.deps.stage.groups;
      for (let j = 0; j < idx.length; j++) {
        const n = idx[j];
        if (!c.active[n]) { c.active[n] = 1; c.nActive++; c.byGroup[groups[n]]++; }
        if (this.firstDN === null && groups[n] === this.dnGroup) {
          this.firstDN = k * this.decision.frame_ms;
          this.milestones.push({ t: this.firstDN, text: `<span class="t">${this.firstDN.toFixed(0)} ms</span> First ${this.term('descending neuron')} spikes: a command channel from the brain to the body.` });
          this.milestones.sort((a, b) => a.t - b.t);
        }
      }
    },

    update(simT) {
      while (this.shown < this.milestones.length && this.milestones[this.shown].t <= simT) {
        const li = document.createElement('li');
        li.innerHTML = this.milestones[this.shown].text;
        $('#captionList').appendChild(li);
        while ($('#captionList').children.length > 3) $('#captionList').firstElementChild.remove();
        this.shown++;
      }
      this.updateCounters();
      this.updateCircuitGlow(simT);
    },

    updateCounters() {
      const c = this.counters;
      if (!c) return;
      const g = this.cfg.neurons.groups;
      const by = (name) => fmt(c.byGroup[g.indexOf(name)] || 0);
      $('#counters').innerHTML = `<div><b>${fmt(c.spikes)}</b> spikes</div><div><b>${fmt(c.nActive)}</b> neurons active</div>
        <div class="row"><span class="dot" style="background:#3987e5"></span>central ${by('central')}</div>
        <div class="row"><span class="dot" style="background:#d95926"></span>mushroom body ${by('mushroom body')}</div>
        <div class="row"><span class="dot" style="background:#199e70"></span>descending ${by('DN')}</div>
        <div class="row"><span class="dot" style="background:#fff"></span>stimulated inputs</div>`;
    },

    endDecision(d) {
      const top = (d.top_neurons || []).slice(0, 3).map((t) => t.type);
      const text = d.q
        ? `Readout: ${this.term('descending neuron')} spike counts (50–300 ms) weigh in at ${this.term('Q-value', 'Q')}(hit) ${d.q.hit.toFixed(3)} vs Q(stick) ${d.q.stick.toFixed(3)}, so the fly will <b>${d.action.toUpperCase()}</b>. Biggest contributors: ${esc(top.join(', '))}.`
        : `Untrained readout: left vs right descending-neuron activity picks <b>${d.action.toUpperCase()}</b>.`;
      const li = document.createElement('li');
      li.innerHTML = `<span class="t">300 ms</span> ${text}`;
      $('#captionList').appendChild(li);
      while ($('#captionList').children.length > 3) $('#captionList').firstElementChild.remove();
    },

    // ------------------------------------------------------------------ circuit and input tabs
    renderCircuit(c) {
      const svg = $('#circuitSvg');
      const W = Math.max(320, svg.clientWidth || 380);
      const hops = [...new Set(c.nodes.map((n) => n.hop))].sort((a, b) => (a < 0) - (b < 0) || a - b);
      const labelW = 92, pad = 14, rowH = 21, top = 24;
      const colX = (h) => pad + 8 + hops.indexOf(h) * ((W - 2 * pad - labelW - 8) / Math.max(1, hops.length - 1));
      const byCol = {};
      c.nodes.forEach((n, i) => { (byCol[n.hop] = byCol[n.hop] || []).push(i); });
      const xy = [];
      let maxRows = 1;
      for (const h of hops) {
        const col = byCol[h].sort((a, b) => c.nodes[b].spikes - c.nodes[a].spikes);
        maxRows = Math.max(maxRows, col.length);
        col.forEach((i, j) => { xy[i] = [colX(h), top + 10 + j * rowH]; });
      }
      const H = top + 20 + maxRows * rowH;
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      svg.setAttribute('height', H);
      const maxSpikes = Math.max(1, ...c.nodes.map((n) => n.spikes));
      const maxSyn = Math.max(2, ...c.edges.map((e) => e.synapses));
      const colors = window.FlyStage.GROUP_COLORS;
      let html = hops.map((h) => `<text class="col-head" x="${colX(h)}" y="12" text-anchor="middle">${h < 0 ? 'unreached' : h === 0 ? 'input' : 'hop ' + h}</text>`).join('');
      html += c.edges.map((e) => {
        const [x1, y1] = xy[e.source], [x2, y2] = xy[e.target];
        const mx = (x1 + x2) / 2 + (x1 === x2 ? 30 : 0);
        return `<path d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}" fill="none" stroke="#898781" stroke-opacity="0.32" stroke-width="${(0.4 + 2.2 * Math.log(e.synapses) / Math.log(maxSyn)).toFixed(2)}"><title>${esc(c.nodes[e.source].type)} → ${esc(c.nodes[e.target].type)}: ${e.synapses} synapses</title></path>`;
      }).join('');
      html += c.nodes.map((n, i) => {
        const [x, y] = xy[i];
        const r = 3 + 7 * Math.sqrt(n.spikes / maxSpikes);
        const color = colors[this.cfg.neurons.groups[n.group]];
        const label = n.type.length > 14 ? n.type.slice(0, 13) + '…' : n.type;
        return `<g class="node" data-i="${i}" tabindex="0" transform="translate(${x},${y})"><circle r="${r.toFixed(1)}" fill="${color}" fill-opacity="0.15" stroke="${color}"></circle><text x="${(r + 4).toFixed(1)}" y="3">${esc(label)}</text><title>${esc(n.type)} (${this.cfg.neurons.groups[n.group]}): ${n.spikes} spikes, ${n.hop < 0 ? 'not reached' : n.hop + ' synaptic hops'} from the stimulated neurons. Click to highlight in the brain.</title></g>`;
      }).join('');
      svg.innerHTML = html;
      const decay = Math.exp(-this.cfg.frame_ms / window.FlyStage.TAU_MS);
      this.circuit = { circles: [...svg.querySelectorAll('.node circle')], traces: c.nodes.map((n) => {
        const tr = new Float32Array(n.activity.length);
        let v = 0;
        n.activity.forEach((a, k) => { v = v * decay + a; tr[k] = v; });
        const m = Math.max(1e-9, ...tr);
        return tr.map((x) => x / m);
      }) };
      svg.querySelectorAll('.node').forEach((g) => {
        const act = async () => {
          const n = c.nodes[+g.dataset.i];
          const res = await window.FlyApi.typeNeurons(n.type_id);
          svg.querySelectorAll('.node.selected').forEach((x) => x.classList.remove('selected'));
          g.classList.add('selected');
          this.deps.stage.highlight(res.idx);
          $('#circuitNote').textContent = `Highlighting ${res.type}: ${res.idx.length} neurons (visible while the brain is shown).`;
        };
        g.addEventListener('click', act);
        g.addEventListener('keydown', (e) => { if (e.key === 'Enter') act(); });
      });
    },

    updateCircuitGlow(simT) {
      if (!this.circuit || !this.decision) return;
      const f = Math.min(Math.floor(simT / this.decision.frame_ms), this.decision.n_frames - 1);
      this.circuit.circles.forEach((el, i) => el.setAttribute('fill-opacity', (0.12 + 0.88 * (simT > 0 ? this.circuit.traces[i][f] : 0)).toFixed(2)));
    },

    renderInput(d) {
      const o = d.observation;
      const ch = this.cfg.channels || [];
      $('#obsText').innerHTML = `Player <b>${o.player_value}</b>${o.usable_ace ? ' (usable ace)' : ''} vs dealer <b>${o.dealer_card === 1 ? 'A' : o.dealer_card}</b> · cached trial ${d.trial}`;
      $('#inputRows').innerHTML = d.input_rates_hz.map((hz, i) => `
        <div class="irow"><span class="name">${ch[i] ? esc(ch[i].type.replace('ORN_', '')) : 'ch ' + i}<small>${ch[i] ? esc(ch[i].meaning) : ''} · ${ch[i] ? ch[i].n_neurons : ''} neurons</small></span>
          <div class="track"><div class="fill" style="width:${(100 * hz / 150).toFixed(1)}%"></div></div><span class="val">${hz.toFixed(0)} Hz</span></div>`).join('');
      $('#inputChip').innerHTML = d.input_rates_hz.map((hz, i) => `<span>${ch[i] ? esc(ch[i].type.replace('ORN_', '')) : 'ch' + i} <b>${Math.round(hz)}</b> Hz</span>`).join('');
    },

    // ------------------------------------------------------------------ strategy chart
    renderStrategy() {
      const p = this.cfg.policy;
      if (!p || !p.optimal_hit) return;
      const fly = p.fly_p_hit;
      const dealers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 1];
      const mode = this.strategyMode;
      const cell = (v, d, u) => {
        const i = this.obsIndex.get(`${v},${d},${u}`);
        const opt = p.optimal_hit[i] === 1;
        const ph = fly ? fly[i] : null;
        const flyHit = ph !== null && ph > 0.5;
        const differs = ph !== null && flyHit !== opt;
        let cls = 'cell', label = '', bg = '';
        if (mode === 'optimal') { cls += opt ? ' hit' : ' stick'; label = opt ? 'H' : 'S'; }
        else if (mode === 'fly' && ph !== null) {
          bg = `background: rgba(57, 135, 229, ${(0.08 + 0.85 * ph).toFixed(2)})`;
          label = flyHit ? 'H' : 'S';
        } else if (mode === 'diff') { cls += differs ? ' differs' : ' agrees'; label = differs ? '✗' : ''; }
        if (i === this.currentObs) cls += ' current';
        const title = `Player ${v}${u ? ' soft' : ''} vs dealer ${d === 1 ? 'A' : d}: optimal ${opt ? 'HIT' : 'STICK'}` +
                      (ph !== null ? `; fly hits in ${Math.round(ph * 100)}% of cached trials` : '');
        return `<td class="${cls}" style="${bg}" title="${title}" data-obs="${i}">${label}</td>`;
      };
      const table = (u, totals, caption) => `<table class="strat"><caption>${caption}</caption>
        <tr><th></th>${dealers.map((d) => `<th>${d === 1 ? 'A' : d}</th>`).join('')}</tr>
        ${totals.map((v) => `<tr><th>${v}</th>${dealers.map((d) => cell(v, d, u)).join('')}</tr>`).join('')}</table>`;
      const hard = Array.from({ length: 18 }, (_, k) => 4 + k);
      const soft = Array.from({ length: 10 }, (_, k) => 12 + k);
      const agree = fly ? p.optimal_hit.filter((o, i) => (fly[i] > 0.5) === (o === 1)).length : null;
      $('#strategyGrid').innerHTML = table(0, hard, 'No usable ace (rows: player total, columns: dealer card)') + table(1, soft, 'Usable ace');
      $('#strategyLegend').innerHTML = mode === 'optimal'
        ? `<span class="swatch hit"></span>H = hit <span class="swatch stick"></span>S = stick · ${this.term('basic strategy')} for these rules`
        : mode === 'fly'
          ? 'Shade = how often the fly hits across its 5 cached trials; letter = its majority choice'
          : `✗ = the fly’s majority choice differs from optimal${agree !== null ? ` · agrees on ${agree} of 280 situations` : ''}`;
    },

    // ------------------------------------------------------------------ science and about tabs
    renderScience() {
      const r = this.cfg.results;
      const el = $('#science');
      if (!r) { el.textContent = 'Results not exported.'; return; }
      const ref = r.references || {};
      const e1 = (r.e1 || []).map((row) => `<tr><td>${esc(row.label)}</td><td class="num">${row.mean.toFixed(4)}</td><td class="num">± ${(row.se || 0).toFixed(4)}</td></tr>`).join('');
      const e3 = r.e3 ? `<table class="data"><tr><th>held-out split</th>${r.e3.rows.map((row) => `<th class="num">${esc(row.label)}</th>`).join('')}</tr>
        ${['Leave one dealer card out', 'Leave one player total out', 'Random 20% held out'].map((fam, k) => `<tr><td>${fam}</td>${r.e3.rows.map((row) => `<td class="num">${row.agreement[k] == null ? '—' : row.agreement[k].toFixed(2)}</td>`).join('')}</tr>`).join('')}</table>` : '';
      el.innerHTML = `
        <p class="headline">${esc(r.headline)}</p>
        <h3>Readout width sweep</h3>
        <p class="note">Each point: a readout of F neurons (or F random features) trained to predict the best action, scored by exact expected winnings per hand (mean ± SE over 5 draws). Higher is better; optimal play is ${(ref.optimal || -0.0466).toFixed(3)}.</p>
        <div id="sweepChart" class="chart"></div>
        <button id="sweepTableBtn" class="small">Show numbers</button>
        <div id="sweepTable" hidden></div>
        <h3>Trained agents (1.5 million practice hands)</h3>
        <table class="data"><tr><th>features</th><th class="num">return / hand</th><th class="num">SE</th></tr>${e1}
          <tr class="ref"><td>Optimal play</td><td class="num">${(ref.optimal || 0).toFixed(4)}</td><td></td></tr>
          <tr class="ref"><td>Hit below 18</td><td class="num">${(ref.hit_below_18 || 0).toFixed(4)}</td><td></td></tr>
          <tr class="ref"><td>Always stick</td><td class="num">${(ref.always_stick || 0).toFixed(4)}</td><td></td></tr></table>
        <h3>Generalizing to unseen situations</h3>
        <p class="note">Agreement with the optimal action on situations the readout never trained on.</p>
        ${e3}
        <h3>Caveats</h3>
        <ul>${(r.caveats || []).map((c) => `<li>${esc(c)}</li>`).join('')}</ul>
        <p class="note">Full details: ${esc(r.report)}.</p>`;
      this.drawSweep(r.e2_sweep, ref.optimal);
      $('#sweepTableBtn').addEventListener('click', () => {
        const t = $('#sweepTable');
        t.hidden = !t.hidden;
        $('#sweepTableBtn').textContent = t.hidden ? 'Show numbers' : 'Hide numbers';
      });
    },

    drawSweep(sweep, optimal) {
      const W = 380, H = 230, m = { l: 48, r: 14, t: 12, b: 36 };
      const keys = Object.keys(SERIES).filter((k) => sweep[k]);
      const all = keys.flatMap((k) => sweep[k]);
      const Fs = [...new Set(all.map((p) => p.F))].sort((a, b) => a - b);
      const yMin = Math.min(...all.map((p) => p.mean - (p.se || 0))) - 0.006;
      const yMax = Math.max(optimal || -0.05, ...all.map((p) => p.mean + (p.se || 0))) + 0.006;
      const x = (F) => m.l + (Math.log(F) - Math.log(Fs[0])) / (Math.log(Fs[Fs.length - 1]) - Math.log(Fs[0])) * (W - m.l - m.r);
      const y = (v) => m.t + (yMax - v) / (yMax - yMin) * (H - m.t - m.b);
      const ticks = [];
      for (let v = Math.ceil(yMin * 50) / 50; v <= yMax; v += 0.02) ticks.push(v);
      let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Exact return per hand versus readout width">`;
      svg += ticks.map((v) => `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="${DARK.grid}"/><text x="${m.l - 6}" y="${y(v) + 3}" text-anchor="end" class="tick">${v.toFixed(2)}</text>`).join('');
      svg += Fs.map((F) => `<text x="${x(F)}" y="${H - m.b + 14}" text-anchor="middle" class="tick">${F}</text>`).join('');
      svg += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle" class="axis-label">readout width F (log scale)</text>`;
      if (optimal) svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(optimal)}" y2="${y(optimal)}" stroke="${DARK.axis}" stroke-width="1"/><text x="${W - m.r}" y="${y(optimal) - 4}" text-anchor="end" class="tick">optimal ${optimal.toFixed(3)}</text>`;
      for (const k of keys) {
        const [name, color] = SERIES[k];
        const pts = sweep[k];
        svg += `<polyline fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${pts.map((p) => `${x(p.F)},${y(p.mean)}`).join(' ')}"/>`;
        svg += pts.map((p) => `<circle class="pt" tabindex="0" cx="${x(p.F)}" cy="${y(p.mean)}" r="4.5" fill="${color}" stroke="${DARK.surface}" stroke-width="2"><title>${name}, F = ${p.F}: ${p.mean.toFixed(4)} ± ${(p.se || 0).toFixed(4)}</title></circle>`).join('');
      }
      svg += '</svg>';
      const legend = keys.map((k) => `<span class="lg"><span class="lgline" style="background:${SERIES[k][1]}"></span>${SERIES[k][0]}</span>`).join('');
      $('#sweepChart').innerHTML = svg + `<div class="legend-row">${legend}</div>`;
      $('#sweepTable').innerHTML = `<table class="data"><tr><th>F</th>${keys.map((k) => `<th class="num">${SERIES[k][0]}</th>`).join('')}</tr>
        ${Fs.map((F) => `<tr><td>${F}</td>${keys.map((k) => { const p = sweep[k].find((q) => q.F === F); return `<td class="num">${p ? p.mean.toFixed(4) : '—'}</td>`; }).join('')}</tr>`).join('')}</table>`;
    },

    // ------------------------------------------------------------------ how it works
    /* Vertical pipeline diagram. Each stage is {title, detail, kind}; kind picks the box style
       ('' frozen/plumbing, 'brain' the simulated connectome, 'trained' the learned readout). */
    howFlowSvg(stages) {
      const W = 412, BX = 14, BW = W - 2 * BX, BH = 48, GAP = 24;
      const H = stages.length * BH + (stages.length - 1) * GAP + 4;
      const parts = stages.map((s, i) => {
        const y = 2 + i * (BH + GAP);
        const arrow = i === stages.length - 1 ? ''
          : `<path class="arrow" d="M${W / 2} ${y + BH + 3} V${y + BH + GAP - 5}"/>`;
        return `<rect class="box ${s.kind || ''}" x="${BX}" y="${y}" width="${BW}" height="${BH}" rx="7"/>
          <text class="t" x="${BX + 12}" y="${y + 20}">${s.title}</text>
          <text class="d" x="${BX + 12}" y="${y + 36}">${s.detail}</text>${arrow}`;
      }).join('');
      return `<svg id="howFlow" viewBox="0 0 ${W} ${H}" height="${H}" role="img"
        aria-label="Pipeline: the hand becomes three numbers, then firing rates, then input spikes into the frozen connectome, then spike counts, then a trained linear readout, then an action.">
        <defs><marker id="howArrow" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0 0 L8 4 L0 8 z" fill="#383835"/></marker></defs>${parts}</svg>`;
    },
    renderHowItWorks() {
      const c = this.cfg, f = c.facts || {}, r = c.results || {};
      const ref = r.references || {};
      const e1 = (label) => ((r.e1 || []).find((x) => x.label === label) || {}).mean;
      const num = (v, d = 3) => (v == null ? '—' : v.toFixed(d));
      const nObs = (c.policy.observations || []).length;
      const nStim = (c.stimulated_idx || []).length;
      const nOut = (c.readout_idx || []).length || f.descending_neurons;
      const nTrials = (c.trials_with_frames || []).length;
      const [w0, w1] = c.window_ms || [50, 300];
      const ch = c.channels || [];
      const chList = ch.map((x) => `${esc(x.meaning)} (${esc(x.type)}, ${x.n_neurons} cells)`).join(', ');

      const flow = this.howFlowSvg([
        { title: 'The hand', detail: `player total, dealer card, usable ace — ${fmt(nObs)} possible situations` },
        { title: 'Rate code', detail: 'each of the three numbers becomes a firing rate, 20–150 Hz' },
        { title: `${fmt(nStim)} smell-receptor neurons`, detail: `${ch.length} types, driven with random (Poisson) spikes` },
        { title: `The fly’s brain, frozen`, detail: `${fmt(f.neurons || 0)} neurons · ${fmt(f.connections || 0)} connections · ${c.T_ms} ms` , kind: 'brain' },
        { title: `${fmt(nOut)} spike counts`, detail: `output neurons, spikes counted from ${w0} to ${w1} ms` },
        { title: 'Linear readout, trained', detail: 'Q(hit) and Q(stick) — the fly takes the larger', kind: 'trained' },
      ]);

      $('#how').innerHTML = `
        <p class="headline">FlyJack takes the complete wiring diagram of a real fruit-fly brain — every neuron and every
        connection, traced from electron-microscope images — and uses it <b>unchanged</b> as a fixed feature extractor.
        Nothing inside the brain is trained. The only thing that learns is a small linear readout on top, which turns the
        brain’s activity into a hit-or-stick decision.</p>
        <p class="note">If you know machine learning: this is ${this.term('random features', 'reservoir computing')}, where the
        reservoir is an actual animal’s circuit instead of a random matrix. The interesting question is whether a reservoir
        shaped by evolution beats a random one. Short answer, from the Science tab: it does not.</p>

        <h3>The pipeline, end to end</h3>
        ${flow}
        <div class="howkey"><span><i class="sw frozen"></i>frozen — never trained</span><span><i class="sw trained"></i>the only trained part</span></div>

        <ol class="howsteps">
          <li><b>The cards become three numbers.</b> A blackjack hand only needs three facts to play correctly: the
          player’s total (4–21), the dealer’s face-up card (A–10), and whether an ace can still count as 11 without
          busting. That is ${fmt(nObs)} distinct situations, and everything downstream sees only those three numbers —
          never the card images.</li>

          <li><b>The three numbers become a smell.</b> A brain simulation cannot be handed a number; it can only be fed
          through its own sense organs. The fly’s most usable input is smell: its antenna carries about 50 types of
          ${this.term('ORN', 'olfactory receptor neuron')}, and each type is a bundle of cells that all answer to the same
          odour and all wire into the same spot. One type therefore behaves like one input unit. Three types are used as
          three channels — ${chList} — and each is driven faster for a higher value, from 20 Hz up to 150 Hz. The spikes
          are random: at every 0.1 ms step each cell fires with probability rate × 0.1 ms, so the same hand never produces
          exactly the same input twice. Vision was the first attempt and it failed (see the Input tab).</li>

          <li><b>The brain runs for ${c.T_ms} ms.</b> Every one of the ${fmt(f.neurons || 0)} neurons is a
          ${this.term('spike', 'leaky integrate-and-fire')} unit: a single number (its voltage) that leaks back toward rest,
          adds up its weighted inputs, emits a 1 when it crosses a threshold, and then resets. The weight matrix <i>is</i>
          the ${this.term('connectome')}: ${fmt(f.connections || 0)} non-zero entries, each one the measured number of
          synapses between a pair of cells converted to millivolts, positive or negative according to the chemical the
          sending cell releases. Those weights never change — not during a hand, not during training. ${c.T_ms} ms of fly
          time at 0.1 ms per step is ${fmt(Math.round((c.T_ms || 300) / 0.1))} forward steps through a
          ${fmt(f.neurons || 0)}-unit recurrent network.</li>

          <li><b>The activity becomes a feature vector.</b> The readout is not allowed to peek anywhere it likes. It listens
          to the ${fmt(nOut)} ${this.term('descending neuron', 'descending neurons')} — the brain’s actual output layer, the
          cells that in a living fly carry commands down to the legs and wings. Counting each one’s spikes between ${w0} and
          ${w1} ms (the first ${w0} ms is skipped while the signal is still travelling inward) gives one vector of
          ${fmt(nOut)} non-negative integers per hand. That vector is the feature representation.</li>

          <li><b>A linear readout decides.</b> Two weight vectors, one per action, (plus a bias) were fitted on 1.5 million
          practice hands: play a batch, record what each decision actually paid out, refit both weight vectors by ridge
          regression, repeat, exploring less each round. They predict the average winnings of hitting and of sticking,
          which are the two
          ${this.term('Q-value', 'Q-values')} in the panel on the left, and the fly takes the larger one. That is
          ${fmt(nOut)} × 2 trained numbers sitting on top of a frozen ${fmt(f.neurons || 0)}-neuron network.</li>
        </ol>

        <h3>The biology words, in machine-learning terms</h3>
        <table class="data"><tr><th>word</th><th>what it is here</th></tr>
          <tr><td>connectome</td><td>the weight matrix — who connects to whom, measured, not invented</td></tr>
          <tr><td>neuron</td><td>one unit with one scalar state that leaks, sums inputs and fires</td></tr>
          <tr><td>synapse</td><td>one entry of the weight matrix; more synapses means a larger weight</td></tr>
          <tr><td>spike</td><td>a 1 instead of a real number: units emit discrete events, not activations</td></tr>
          <tr><td>firing rate (Hz)</td><td>spikes per second — how a number is represented on a wire</td></tr>
          <tr><td>olfactory receptor neuron</td><td>the input layer we are allowed to clamp</td></tr>
          <tr><td>descending neuron</td><td>the output layer the readout reads</td></tr>
          <tr><td>${this.term('mushroom body')}</td><td>a named region, the fly’s learning centre — here just a subset of units</td></tr>
        </table>

        <details><summary class="note">Show the neuron model</summary>
          <p class="note">Leaky integrate-and-fire with alpha-function synapses, as published with the connectome:
          resting and reset voltage −52 mV, threshold −45 mV, membrane time constant 20 ms, synaptic time constant 5 ms,
          refractory period 2.2 ms, axonal delay 1.8 ms, weight = synapse count × 0.275 mV signed by neurotransmitter,
          step 0.1 ms. Run on the GPU as a batched PyTorch loop; validated against the original Brian2 code
          (Pearson 0.999 on log firing rates).</p></details>

        <h3>Why it is set up this way</h3>
        <p>The frozen brain plus a trained linear head is what makes the claim testable. If the readout were allowed to be
        a deep network, any result would be about the readout. Keeping it linear means the score measures how much
        blackjack-relevant structure the brain’s own activity exposes. The comparisons that matter, in expected winnings
        per hand (higher is better, and every scheme loses — the house edge is real):</p>
        <table class="data"><tr><th>what the readout sees</th><th class="num">return / hand</th></tr>
          <tr><td>Perfect play (lookup table)</td><td class="num">${num(ref.optimal)}</td></tr>
          <tr><td>The three input rates, nothing else</td><td class="num">${num(e1('Encoder inputs only'))}</td></tr>
          <tr><td>Random features, width-matched, noise-matched</td><td class="num">${num(e1('Random features + Poisson noise (DN width)'))}</td></tr>
          <tr><td>The real connectome</td><td class="num">${num(e1('Connectome readout, real (DN)'))}</td></tr>
          <tr><td>Nothing — always stick</td><td class="num">${num(ref.always_stick)}</td></tr></table>
        <p>Read top to bottom: the real brain is beaten by random features of the same width, and beaten by feeding the
        same linear readout the three input numbers directly. So passing the hand through the connectome does not add
        usable structure — it loses some of what went in. A shuffled connectome scores worse still, but only because a
        shuffled brain is silent, which makes that comparison uninformative. This is a negative result, and it is the
        finding; the Science tab has the error bars and caveats.</p>

        <h3>What you are actually watching</h3>
        <ul>
          <li>Nothing is simulated live. Each situation has ${nTrials} pre-computed trials, and a decision replays one of
          them, slowed about 20×. Same wiring, same model, just cached — and these ${nTrials} are held-out trials, run
          with different random input spikes from the ones the readout was fitted on.</li>
          <li>The brain floating above the fly is drawn about 10× life size; the real thing is
          ${(f.brain_size_um || []).join(' × ')} µm.</li>
          <li>The leg tap and sweep are hand-animated. This connectome stops at the neck, so no simulated signal can
          reach a leg.</li>
          <li>The cards, the shoe and the payouts are a real blackjack implementation; the bankroll is the fly’s actual
          running score.</li>
        </ul>
        <p class="note">The About tab lists exactly what is real, cached, scripted and magnified.</p>`;
    },
    renderFacts() {
      const f = this.cfg.facts;
      if (!f) return;
      const card = (value, label, detail) => `<div class="fact"><div class="fact-value">${value}</div><div class="fact-label">${label}</div>${detail ? `<div class="note">${detail}</div>` : ''}</div>`;
      $('#facts').innerHTML = [
        card(fmt(f.neurons), 'neurons', `in the ${esc(f.connectome)} ${this.term('connectome')}`),
        card(fmt(f.synapses), 'synapses', `between ${fmt(f.connections)} connected neuron pairs`),
        card(fmt(f.cell_types), 'cell types', 'labels used in the circuit view'),
        card(fmt(f.kenyon_cells), this.term('Kenyon cell', 'Kenyon cells'), 'mushroom body neurons'),
        card(fmt(f.descending_neurons), this.term('descending neuron', 'descending neurons'), 'brain-to-body output; the readout listens to these'),
        card(`${f.brain_size_um[0]} × ${f.brain_size_um[1]} × ${f.brain_size_um[2]} µm`, 'brain size', 'shown about 10× magnified above the fly'),
      ].join('');
    },
  };

  window.FlyLearn = L;
})();

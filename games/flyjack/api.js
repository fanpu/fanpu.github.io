/* One interface over two backends. In server mode every call hits server/app.py; in static mode the game runs
   in the browser (web/game.js) and every decision is a precomputed file produced by scripts/07_export_static.py.
   Static mode is selected by window.FLYJACK_STATIC = {base, bundle}, which the exported index.html sets. */
(function () {
  'use strict';
  const S = window.FLYJACK_STATIC || null;

  async function getJSON(url, opts) {
    const r = await fetch(url, opts);
    if (!r.ok) { const e = new Error(`${url}: ${r.status}`); e.status = r.status; throw e; }
    return r.json();
  }
  async function buffer(url) {
    const r = await fetch(url);
    if (!r.ok) { const e = new Error(`${url}: ${r.status}`); e.status = r.status; throw e; }
    return r.arrayBuffer();
  }
  const post = (url, body) =>
    getJSON(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

  // ---------------------------------------------------------------- server mode
  const server = {
    mode: 'server',
    gameId: null,
    config() { return getJSON('/config'); },
    async deal() { const s = await post('/deal', { game_id: this.gameId }); this.gameId = s.game_id; return s; },
    act() { return post('/act', { game_id: this.gameId }); },
    decision(id) { return getJSON(`/decision/${id}`); },
    frames(id) { return buffer(`/decision/${id}/frames`); },
    typeNeurons(t) { return getJSON(`/type/${t}/neurons`); },
    asset(name) { return `/assets/${name}`; },
    stageAsset(name) { return `/assets/stage/${name}`; },
    newGame() { this.gameId = null; },
  };

  // ---------------------------------------------------------------- static mode
  const pad = (n) => String(n).padStart(3, '0');

  /** Inverse of the exporter's writer: "FJF1", uint32 n_frames, uint32 counts[n], then per-frame
      ascending neuron indices delta-encoded from -1 as LEB128. Returns the flat Int32Array layout
      [n_frames, offsets[n+1], idx[]] that FlyStage.loadFrames already expects. */
  function decodeFrames(bytes) {
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (bytes[0] !== 0x46 || bytes[1] !== 0x4A || bytes[2] !== 0x46 || bytes[3] !== 0x31) throw new Error('bad frames magic');  // "FJF1"
    const n = dv.getUint32(4, true);
    const counts = new Uint32Array(n);
    let total = 0;
    for (let i = 0; i < n; i++) { counts[i] = dv.getUint32(8 + 4 * i, true); total += counts[i]; }
    const out = new Int32Array(1 + (n + 1) + total);
    out[0] = n;
    let p = 8 + 4 * n, w = n + 2, acc = 0;  // idx starts after [n_frames] + offsets[n+1]
    out[1] = 0;
    for (let f = 0; f < n; f++) {
      let prev = -1;
      for (let j = 0; j < counts[f]; j++) {
        let shift = 0, d = 0, b;
        do { b = bytes[p++]; d |= (b & 0x7f) << shift; shift += 7; } while (b & 0x80);
        prev += d;
        out[w++] = prev;
      }
      acc += counts[f];
      out[2 + f] = acc;
    }
    return out.buffer;
  }

  async function gunzip(url) {
    const r = await fetch(url);
    if (!r.ok) { const e = new Error(`${url}: ${r.status}`); e.status = r.status; throw e; }
    const stream = r.body.pipeThrough(new DecompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  const staticApi = {
    mode: 'static',
    base: S ? S.base : '',
    game: null,
    obsIndex: null,
    _decisions: new Map(),
    _typeIdx: null,

    async config() {
      const manifest = await getJSON(this.base + 'manifest.json');
      if (S && S.bundle && manifest.bundle_version !== S.bundle) {
        throw new Error(`data bundle ${manifest.bundle_version} does not match page build ${S.bundle}`);
      }
      this.manifest = manifest;
      const cfg = await getJSON(this.base + 'config.json');
      this.obsIndex = new Map(cfg.policy.observations.map((o, i) => [`${o[0]},${o[1]},${o[2]}`, i]));
      return cfg;
    },

    _ensureGame() {
      if (!this.game) {
        const seed = Number(new URLSearchParams(location.search).get('seed'));
        this.game = new window.FlyGame.Game(seed || (Math.random() * 4294967296) >>> 0, this.obsIndex);
      }
      return this.game;
    },

    _payload(g) { return { game_id: 'local', seed: g.seed, ...g.state() }; },

    async deal() {
      const g = this._ensureGame();
      g.deal();
      g.decisionId = pad(g.observation());
      return this._payload(g);
    },

    async act() {
      const g = this.game;
      const d = await this.decision(g.decisionId);
      const acted = { decision_id: g.decisionId, action: d.action };
      if (d.action === 'hit') {
        g.hit();
        if (g.phase === 'decide') g.decisionId = pad(g.observation());
      } else {
        g.stick();
      }
      if (g.phase === 'done') g.decisionId = null;
      return { ...this._payload(g), acted };
    },

    async decision(id) {
      if (!this._decisions.has(id)) {
        this._decisions.set(id, await getJSON(`${this.base}decisions/${id}.json`));
      }
      return { decision_id: id, ...this._decisions.get(id) };
    },

    async frames(id) { return decodeFrames(await gunzip(`${this.base}frames/${id}.bin.gz`)); },

    async typeNeurons(t) {
      if (!this._typeIdx) this._typeIdx = new Uint16Array(await buffer(this.base + 'common/type_idx.bin'));
      const a = this._typeIdx, idx = [];
      for (let i = 0; i < a.length; i++) if (a[i] === t) idx.push(i);
      return { type: String(t), idx };
    },

    asset(name) { return this.base + 'common/' + name; },
    stageAsset(name) { return this.base + 'stage/' + name; },
    newGame() { this.game = null; },
  };

  window.FlyApi = S ? staticApi : server;
  window.FlyApi._decodeFrames = decodeFrames;   // exercised against the Python encoder in tests
  window.FlyApi.getJSON = getJSON;
  window.FlyApi.buffer = buffer;
})();

// Text that belongs to places on the table (names, stacks, bets, the pot) is DOM, not 3D: it stays crisp at
// any size and is readable by assistive tech. Each label tracks a world point and is re-projected every frame.
export function createLabels(stage, container) {
  const layer = document.createElement("div");
  layer.className = "labels";
  container.appendChild(layer);
  const labels = new Map();

  stage.onFrame(() => {
    for (const l of labels.values()) {
      const p = stage.project(l.at.x, l.at.y, l.at.z);
      const show = !!p && !l.hidden;
      l.el.style.visibility = show ? "visible" : "hidden";
      if (!show) continue;
      // Keep the whole label on screen: on a phone the side seats would otherwise hang off the edge.
      const half = l.width / 2 + 4,
        w = stage.container.clientWidth;
      const x = half * 2 < w ? Math.max(half, Math.min(w - half, p.x)) : p.x;
      l.el.style.transform = `translate(-50%, -50%) translate(${x.toFixed(1)}px, ${p.y.toFixed(1)}px)`;
    }
  });

  return {
    layer,
    set(key, at, html, className = "") {
      let l = labels.get(key);
      if (!l) {
        l = { el: document.createElement("div"), html: null, cls: null, width: 0 };
        layer.appendChild(l.el);
        labels.set(key, l);
      }
      l.at = at;
      l.hidden = false;
      if (l.cls !== className) l.el.className = "label " + (l.cls = className);
      if (l.html !== html) {
        l.el.innerHTML = l.html = html;
        l.width = l.el.offsetWidth;
      }
      return l.el;
    },
    hide(key) {
      const l = labels.get(key);
      if (l) l.hidden = true;
    },
    remove(key) {
      labels.get(key)?.el.remove();
      labels.delete(key);
    },
    clear() {
      for (const key of [...labels.keys()]) this.remove(key);
    },
  };
}

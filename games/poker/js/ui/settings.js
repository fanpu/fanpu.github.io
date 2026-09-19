// Settings, as a small popover under the gear.
const FIELDS = [
  {
    key: "players",
    label: "Players at the table",
    type: "select",
    options: [2, 3, 4, 5, 6, 7, 8, 9].map((n) => [n, n === 2 ? "2 (heads-up)" : String(n)]),
    note: "Applies from the next hand.",
  },
  {
    key: "speed",
    label: "Table speed",
    type: "select",
    options: [
      [0.75, "Relaxed"],
      [1, "Normal"],
      [1.6, "Brisk"],
      [2.5, "Fast"],
    ],
  },
  {
    key: "pause",
    label: "Pause to show the grade",
    type: "select",
    options: [
      ["mistake", "When I stray from the coach"],
      ["always", "After every decision"],
      ["never", "Never"],
    ],
    note: "Train only. Prove it never pauses.",
  },
  { key: "showPick", label: "Show the coach’s pick", type: "check", note: "Highlights the recommended action and adds the Why tab." },
  { key: "fourColour", label: "Four-colour deck", type: "check", note: "Blue diamonds and green clubs: flushes are easier to spot." },
  { key: "animations", label: "Animations", type: "check" },
  { key: "muted", label: "Mute sound", type: "check" },
];

export function createSettings(root, { store, onChange }) {
  let open = false;
  function render() {
    root.hidden = !open;
    if (!open) return;
    const s = store.settings;
    root.innerHTML =
      `<h2>Settings</h2>` +
      FIELDS.map((f) =>
        f.type === "check"
          ? `<label class="check"><input type="checkbox" data-key="${f.key}" ${s[f.key] ? "checked" : ""} /><span>${f.label}${
              f.note ? `<small>${f.note}</small>` : ""
            }</span></label>`
          : `<label class="field"><span>${f.label}${f.note ? `<small>${f.note}</small>` : ""}</span><select data-key="${f.key}">${f.options
              .map(([v, l]) => `<option value="${v}" ${String(s[f.key]) === String(v) ? "selected" : ""}>${l}</option>`)
              .join("")}</select></label>`
      ).join("") +
      `<button class="quiet" data-reset>Reset my scores</button>`;
  }
  root.addEventListener("change", (e) => {
    const key = e.target.dataset.key;
    if (!key) return;
    const f = FIELDS.find((x) => x.key === key);
    const value = f.type === "check" ? e.target.checked : typeof f.options[0][0] === "number" ? +e.target.value : e.target.value;
    store.setSetting(key, value);
    onChange?.(key, value);
  });
  root.addEventListener("click", (e) => {
    if (!e.target.closest("[data-reset]")) return;
    if (confirm("Clear your scorecards at both coach levels? Lesson progress is kept.")) store.reset();
  });
  addEventListener("pointerdown", (e) => open && !root.contains(e.target) && !e.target.closest("[data-settings]") && api.toggle(false));
  addEventListener("keydown", (e) => e.key === "Escape" && open && api.toggle(false));
  const api = {
    toggle(to = !open) {
      open = to;
      render();
    },
  };
  render();
  return api;
}

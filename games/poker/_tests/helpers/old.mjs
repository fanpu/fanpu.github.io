// Loads the DOM-free core of the old single-file trainer so tests can use it as a second oracle.
import { execFileSync } from "node:child_process";
import vm from "node:vm";

export function loadOld(seedRng) {
  const html = execFileSync("git", ["show", "5a64bf9c:games/poker/index.html"], { maxBuffer: 1 << 26 }).toString();
  const first = html.split("<script>")[1].split("</script>")[0];
  const module = { exports: {} };
  const ctx = vm.createContext({ module, Math: Object.assign(Object.create(Math), { random: seedRng }) });
  vm.runInContext(first, ctx);
  return module.exports;
}

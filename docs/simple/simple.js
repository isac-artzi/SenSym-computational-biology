/* ============================================================
   simple.js — helpers for the plain-language companion pages
   (docs/simple/). Depends on cd.js for Phi().

   SIMPLE.calc(holderId, fields, compute)
     fields:  [{ id, label, value, step }]
     compute: function(values) -> HTML string for the readout
   Re-computes every time any box changes.
   ============================================================ */
window.SIMPLE = (function () {
  "use strict";

  /* Inverse of Phi by bisection — slow but exact enough for a readout. */
  function PhiInv(p) {
    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;
    let lo = -10, hi = 10;
    for (let i = 0; i < 80; i++) {
      const mid = (lo + hi) / 2;
      if (CD.Phi(mid) < p) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  function fmt(x, d) {
    if (!isFinite(x)) return x > 0 ? "∞" : "−∞";
    return Number(x).toFixed(d === undefined ? 3 : d);
  }
  function pct(x, d) { return (100 * x).toFixed(d === undefined ? 1 : d) + "%"; }

  function calc(holderId, fields, compute) {
    const holder = document.getElementById(holderId);
    if (!holder) return;
    const row = document.createElement("div");
    row.className = "controls";
    const inputs = {};
    fields.forEach(f => {
      const lab = document.createElement("label");
      lab.innerHTML = f.label + " ";
      const inp = document.createElement("input");
      inp.type = "number";
      inp.value = f.value;
      inp.step = f.step || "any";
      inp.style.width = (f.width || 90) + "px";
      inp.style.fontFamily = "var(--mono)";
      inputs[f.id] = inp;
      lab.appendChild(inp);
      row.appendChild(lab);
    });
    const out = document.createElement("div");
    out.className = "readout";
    holder.appendChild(row);
    holder.appendChild(out);
    function run() {
      const v = {};
      for (const k in inputs) v[k] = parseFloat(inputs[k].value);
      try { out.innerHTML = compute(v); }
      catch (e) { out.innerHTML = "<span style='color:var(--red)'>" + e.message + "</span>"; }
    }
    for (const k in inputs) inputs[k].addEventListener("input", run);
    run();
  }

  return { calc, PhiInv, fmt, pct };
})();

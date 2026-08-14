/* ============================================================
   cd.js — shared helper library for the interactive textbook
   "How Much Data Does It Take to Tell Two Cell Types Apart?"

   Vanilla JS, no dependencies (KaTeX is loaded separately).

   What lives here:
     1. Seeded random numbers (so every lab is reproducible)
     2. Counting-statistics samplers: Poisson, binomial, normal
     3. Detection-theory helpers: d', Phi, error probability, ROC/AUC
     4. Tiny classifiers (naive Bayes, logistic regression, kNN)
     5. A small canvas plotting toolkit (axes, lines, histograms, ROC)
     6. UI helpers: sliders, buttons, readouts
     7. Page plumbing: KaTeX auto-render, objectives checklists

   Every chapter page uses this file. Read it — it is deliberately
   short and heavily commented, and the JS mirrors the NumPy code
   you will write in the `celldetect` Python package. Where a
   function has a Python twin, the comment names it.
   ============================================================ */

window.CD = (function () {
  "use strict";

  /* ============================================================
     1. SEEDED RANDOMNESS
     ============================================================ */

  /* mulberry32: a tiny, high-quality 32-bit PRNG. Given the same
     seed it always produces the same sequence — which is exactly
     what a scientist wants. Every lab on this site seeds explicitly
     so that "re-run" reproduces, and "resample" is a deliberate act. */
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* ============================================================
     2. COUNTING-STATISTICS SAMPLERS
        Python twins: celldetect.counting
     ============================================================ */

  /* Standard normal via Box–Muller. */
  function randn(r) {
    let u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  /* Poisson sampler.
     Small lambda: Knuth's product-of-uniforms method — multiply
     uniforms until the product drops below e^{-lambda}; the number of
     multiplications is Poisson(lambda). Exact, but O(lambda) time.
     Large lambda: a Poisson(lambda) is approximately N(lambda, lambda)
     (this IS the approximation the whole course rests on), so we use a
     rounded normal above lambda = 30 to keep the labs responsive.
     The threshold matters: at lambda = 30 the two agree to well inside
     the visual resolution of any histogram on this page. */
  function poisson(lambda, r) {
    if (lambda <= 0) return 0;
    if (lambda < 30) {
      const L = Math.exp(-lambda);
      let k = 0, p = 1;
      do { k++; p *= r(); } while (p > L);
      return k - 1;
    }
    return Math.max(0, Math.round(lambda + Math.sqrt(lambda) * randn(r)));
  }

  /* Binomial(n, p) by summing Bernoulli trials for small n, normal
     approximation for large n. Used by the bead-jar labs, where a
     "draw of n beads" is literally n Bernoulli trials. */
  function binomial(n, p, r) {
    if (n <= 0) return 0;
    if (n < 200) {
      let k = 0;
      for (let i = 0; i < n; i++) if (r() < p) k++;
      return k;
    }
    const mu = n * p, sd = Math.sqrt(n * p * (1 - p));
    return Math.min(n, Math.max(0, Math.round(mu + sd * randn(r))));
  }

  /* Hypergeometric-ish draw WITHOUT replacement is deliberately NOT
     provided: the bead jars are treated as large enough that drawing
     with replacement is the honest model, and the chapter says so. */

  /* Thin a count by keeping each molecule with probability `keep`.
     This is the mathematical content of BOTH "sequencing at lower
     depth" and the bead experiment's dropout rule: Poisson thinning.
     Python twin: celldetect.counting.downsample_counts */
  function thin(count, keep, r) { return binomial(count, keep, r); }

  /* ============================================================
     3. DETECTION THEORY
        Python twins: celldetect.detection
     ============================================================ */

  /* Abramowitz & Stegun 7.1.26 rational approximation to erf.
     Accurate to ~1.5e-7 — far below anything visible on a plot. */
  function erf(x) {
    const s = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
          a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    const t = 1 / (1 + p * x);
    const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return s * y;
  }

  /* Standard normal CDF. */
  function Phi(z) { return 0.5 * (1 + erf(z / Math.SQRT2)); }

  /* The separation index d' for two Poisson counts with means l1, l2.
     Derived in Week 3:  d' = |l1 - l2| / sqrt((l1 + l2)/2).
     The denominator is the standard deviation at the *pooled* mean —
     legitimate because Poisson variance equals its mean. */
  function dPrime(l1, l2) {
    const lbar = 0.5 * (l1 + l2);
    if (lbar <= 0) return 0;
    return Math.abs(l1 - l2) / Math.sqrt(lbar);
  }

  /* Combine independent genes: d' values add in quadrature, so k
     identical genes give d'_total = sqrt(k) * d'_single (Week 5). */
  function combineDPrime(list) {
    return Math.sqrt(list.reduce((s, d) => s + d * d, 0));
  }

  /* Optimal two-class accuracy under the Gaussian approximation with
     equal priors: acc = Phi(d'/2). Week 4's headline formula. */
  function accuracyFromDPrime(dp) { return Phi(dp / 2); }

  /* Empirical ROC from scores and binary labels (1 = positive).
     Returns {points: [[fpr, tpr], ...], auc}.

     DELIBERATE DIFFERENCE from the Python twin
     (celldetect.detection.roc_curve): this version walks the samples one at
     a time and does NOT collapse tied scores, so with integer counts it
     inflates the AUC — a classifier that gives every cell the same score
     comes out at 0 or 1 depending on the input ordering rather than at 1/2.
     That is a bug in any real analysis, and the Python version guards
     against it. It is kept here on purpose so that Week 4's lab can print
     the honest, tie-aware AUC next to this one and let you see the size of
     the error. Never use this function to report a number. */
  function roc(scores, labels) {
    const idx = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
    const P = labels.reduce((s, l) => s + (l === 1 ? 1 : 0), 0);
    const N = labels.length - P;
    let tp = 0, fp = 0, auc = 0, prevFpr = 0;
    const pts = [[0, 0]];
    idx.forEach(i => {
      if (labels[i] === 1) tp++; else fp++;
      const tpr = P ? tp / P : 0, fpr = N ? fp / N : 0;
      auc += (fpr - prevFpr) * tpr;          // trapezoid-free step integral
      prevFpr = fpr;
      pts.push([fpr, tpr]);
    });
    return { points: pts, auc: auc };
  }

  /* Wilson score interval for a binomial proportion — the honest error
     bar for "k successes out of n trials". Used everywhere accuracy is
     reported, because the normal interval lies badly near 0 and 1.
     Python twin: celldetect.stats.wilson_interval */
  function wilson(k, n, z) {
    z = z || 1.96;
    if (n === 0) return [0, 1];
    const p = k / n, z2 = z * z;
    const denom = 1 + z2 / n;
    const centre = p + z2 / (2 * n);
    const half = z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n));
    // Clip into [0, 1], matching celldetect.stats.wilson_interval. Without
    // the clip, floating-point dust prints intervals like [-0.000, 0.161].
    return [Math.max(0, (centre - half) / denom),
            Math.min(1, (centre + half) / denom)];
  }

  /* ============================================================
     4. TINY CLASSIFIERS
        Python twins: celldetect.classify
     ============================================================ */

  /* Poisson naive-Bayes log-likelihood ratio for one cell.
     For counts x and rate vectors l1, l2 (equal priors), the optimal
     decision is: call class 1 iff  sum_g [ x_g log(l1_g/l2_g) - (l1_g - l2_g) ] > 0.
     This is the ONLY classifier the theory in Weeks 3-5 describes;
     everything else (logistic, kNN) is measured against it. */
  function nbLogRatio(x, l1, l2) {
    let s = 0;
    for (let g = 0; g < x.length; g++) {
      const a = Math.max(l1[g], 1e-12), b = Math.max(l2[g], 1e-12);
      s += x[g] * Math.log(a / b) - (a - b);
    }
    return s;
  }

  /* Logistic regression by plain gradient descent on the mean
     log-loss. No regularization, no fuss: the point is that the
     student can read every line. Returns {w, b, predict, prob}. */
  function logisticFit(X, y, opts) {
    opts = opts || {};
    const lr = opts.lr || 0.1, iters = opts.iters || 400;
    const n = X.length, d = X[0].length;
    let w = new Array(d).fill(0), b = 0;
    for (let it = 0; it < iters; it++) {
      const gw = new Array(d).fill(0);
      let gb = 0;
      for (let i = 0; i < n; i++) {
        let z = b;
        for (let j = 0; j < d; j++) z += w[j] * X[i][j];
        const p = 1 / (1 + Math.exp(-z));
        const e = p - y[i];
        for (let j = 0; j < d; j++) gw[j] += e * X[i][j];
        gb += e;
      }
      for (let j = 0; j < d; j++) w[j] -= lr * gw[j] / n;
      b -= lr * gb / n;
    }
    const prob = x => {
      let z = b;
      for (let j = 0; j < d; j++) z += w[j] * x[j];
      return 1 / (1 + Math.exp(-z));
    };
    return { w: w, b: b, prob: prob, predict: x => (prob(x) > 0.5 ? 1 : 0) };
  }

  /* k-nearest neighbours with Euclidean distance. O(n) per query —
     fine for the few-hundred-cell datasets in these labs. */
  function knnPredict(Xtrain, ytrain, x, k) {
    const ds = Xtrain.map((xt, i) => {
      let s = 0;
      for (let j = 0; j < x.length; j++) { const dd = xt[j] - x[j]; s += dd * dd; }
      return [s, ytrain[i]];
    }).sort((a, b) => a[0] - b[0]);
    let votes = 0;
    for (let i = 0; i < k && i < ds.length; i++) votes += ds[i][1];
    return votes * 2 > Math.min(k, ds.length) ? 1 : 0;
  }

  /* ============================================================
     5. SYNTHETIC CELLS
        Python twin: celldetect.simulate.simulate_dataset
     ============================================================ */

  /* Sample one cell: for each of k genes, a Poisson count with mean
     depth * p_g, where p is the type's expression profile (a vector of
     per-molecule probabilities). Optionally thin by (1 - dropout). */
  function sampleCell(profile, depth, r, dropout) {
    const x = new Array(profile.length);
    for (let g = 0; g < profile.length; g++) {
      let c = poisson(depth * profile[g], r);
      if (dropout) c = thin(c, 1 - dropout, r);
      x[g] = c;
    }
    return x;
  }

  /* Two profiles differing by a fold change on every marker gene.
     Returns [p1, p2], each of length k, each summing to `mass`
     (the fraction of the transcriptome these markers account for). */
  function markerProfiles(k, fold, mass) {
    mass = mass === undefined ? 0.02 : mass;
    const base = mass / k;
    const p1 = new Array(k).fill(base * 2 / (1 + 1 / fold));
    const p2 = new Array(k).fill(base * 2 / (1 + fold));
    return [p1, p2];
  }

  /* ============================================================
     6. CANVAS PLOTTING TOOLKIT
        Usage:
          const P = CD.plot('myCanvas', {xmin:0, xmax:1, ymin:0, ymax:2,
                                         xlabel:'depth', ylabel:'accuracy'});
          P.clear(); P.axes(); P.line(points, {color: CD.C.blue});
        Handles high-DPI screens automatically.
     ============================================================ */

  const C = {           // shared, colorblind-friendly palette
    blue: "#2563eb", orange: "#ea580c", teal: "#0d9488",
    purple: "#7c3aed", gold: "#b45309", gray: "#94a3b8",
    navy: "#1f3a5f", red: "#b91c1c", green: "#15803d",
    magenta: "#c026d3",
  };

  function plot(canvasId, opts) {
    const cv = typeof canvasId === "string" ? document.getElementById(canvasId) : canvasId;
    const dpr = window.devicePixelRatio || 1;
    // Careful: assigning to cv.width / cv.height writes back into the element's
    // width and height attributes. So the authored size can only be read ONCE --
    // after the first call those attributes hold device-pixel values, and
    // re-reading them would multiply by the pixel ratio again on every redraw
    // (the canvas doubles in height each time a slider moves, on any display
    // with dpr > 1). Stash the authored values the first time and reuse them.
    if (cv.dataset.baseHeight === undefined) {
      cv.dataset.baseWidth = String(parseInt(cv.getAttribute("width")) || 640);
      cv.dataset.baseHeight = String(parseInt(cv.getAttribute("height")) || 340);
    }
    const cssW = cv.clientWidth || parseInt(cv.dataset.baseWidth);
    const cssH = parseInt(cv.dataset.baseHeight);
    cv.width = cssW * dpr; cv.height = cssH * dpr;
    cv.style.height = cssH + "px";
    const ctx = cv.getContext("2d");
    ctx.scale(dpr, dpr);

    const o = Object.assign({ xmin: 0, xmax: 1, ymin: 0, ymax: 1, xlabel: "", ylabel: "",
                              pad: { l: 58, r: 16, t: 12, b: 42 } }, opts);
    const W = cssW, H = cssH, P = o.pad;
    const xw = W - P.l - P.r, yh = H - P.t - P.b;

    const X = x => P.l + (x - o.xmin) / (o.xmax - o.xmin) * xw;
    const Y = y => H - P.b - (y - o.ymin) / (o.ymax - o.ymin) * yh;

    function clear() { ctx.clearRect(0, 0, W, H); }

    function axes(xticks, yticks) {
      ctx.strokeStyle = "#cbd5e1"; ctx.lineWidth = 1;
      ctx.font = "12px sans-serif"; ctx.fillStyle = "#5b6472";
      ctx.strokeRect(P.l, P.t, xw, yh);
      const xt = xticks || ticks(o.xmin, o.xmax, 6);
      const yt = yticks || ticks(o.ymin, o.ymax, 5);
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      xt.forEach(t => {
        ctx.strokeStyle = "#eef2f7";
        strokeLine(X(t), P.t, X(t), H - P.b);
        ctx.fillStyle = "#5b6472";
        ctx.fillText(o.xfmt ? o.xfmt(t) : fmt(t), X(t), H - P.b + 6);
      });
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      yt.forEach(t => {
        ctx.strokeStyle = "#eef2f7";
        strokeLine(P.l, Y(t), W - P.r, Y(t));
        ctx.fillStyle = "#5b6472";
        ctx.fillText(o.yfmt ? o.yfmt(t) : fmt(t), P.l - 6, Y(t));
      });
      ctx.fillStyle = "#1f3a5f"; ctx.font = "600 13px sans-serif";
      if (o.xlabel) { ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(o.xlabel, P.l + xw / 2, H - 4); }
      if (o.ylabel) {
        ctx.save(); ctx.translate(13, P.t + yh / 2); ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(o.ylabel, 0, 0); ctx.restore();
      }
    }

    function strokeLine(x1, y1, x2, y2) { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }

    function line(pts, st) {
      st = st || {};
      ctx.strokeStyle = st.color || C.blue; ctx.lineWidth = st.width || 2;
      if (st.dash) ctx.setLineDash(st.dash);
      ctx.beginPath();
      pts.forEach((p, i) => { const x = X(p[0]), y = Y(p[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke(); ctx.setLineDash([]);
    }

    function funcPlot(f, st) {
      const pts = [];
      for (let i = 0; i <= 300; i++) {
        const x = o.xmin + (o.xmax - o.xmin) * i / 300;
        const y = f(x);
        if (isFinite(y)) pts.push([x, y]);
      }
      line(pts, st);
    }

    function scatter(pts, st) {
      st = st || {};
      ctx.fillStyle = st.color || C.blue;
      const r = st.r || 3;
      pts.forEach(p => { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), r, 0, 2 * Math.PI); ctx.fill(); });
    }

    /* Error bars: points as [x, y, lo, hi]. The course reports every
       measured proportion with a Wilson interval — this draws them. */
    function errorbars(pts, st) {
      st = st || {};
      ctx.strokeStyle = st.color || C.blue; ctx.lineWidth = st.width || 1.5;
      const cap = st.cap || 4;
      pts.forEach(p => {
        const x = X(p[0]);
        strokeLine(x, Y(p[2]), x, Y(p[3]));
        strokeLine(x - cap, Y(p[2]), x + cap, Y(p[2]));
        strokeLine(x - cap, Y(p[3]), x + cap, Y(p[3]));
      });
      scatter(pts.map(p => [p[0], p[1]]), st);
    }

    /* Histogram of raw data; normalize:'density' makes total area 1.
       Integer mode (step 1 bins) is right for counts. */
    function histogram(data, nbins, st) {
      st = st || {};
      const bins = new Array(nbins).fill(0);
      const lo = o.xmin, hi = o.xmax, bw = (hi - lo) / nbins;
      data.forEach(v => {
        const k = Math.floor((v - lo) / bw);
        if (k >= 0 && k < nbins) bins[k]++;
      });
      let scaleY = 1;
      if (st.normalize === "density") scaleY = 1 / (data.length * bw);
      ctx.fillStyle = st.color || "rgba(37,99,235,0.5)";
      ctx.strokeStyle = st.edge || "rgba(37,99,235,0.9)";
      ctx.lineWidth = 1;
      bins.forEach((c, k) => {
        const x0 = X(lo + k * bw), x1 = X(lo + (k + 1) * bw), y = Y(c * scaleY);
        ctx.fillRect(x0, y, x1 - x0, Y(0) - y);
        ctx.strokeRect(x0, y, x1 - x0, Y(0) - y);
      });
      return bins;
    }

    function label(x, y, text, st) {
      st = st || {};
      ctx.fillStyle = st.color || C.navy; ctx.font = st.font || "600 13px sans-serif";
      ctx.textAlign = st.align || "left"; ctx.textBaseline = "alphabetic";
      ctx.fillText(text, X(x), Y(y));
    }

    /* A vertical rule with a caption — for marking a threshold. */
    function vline(x, text, st) {
      st = st || {};
      ctx.strokeStyle = st.color || C.gray; ctx.lineWidth = st.width || 1.5;
      ctx.setLineDash(st.dash || [5, 4]);
      strokeLine(X(x), P.t, X(x), H - P.b);
      ctx.setLineDash([]);
      if (text) {
        ctx.fillStyle = st.color || C.gray; ctx.font = "600 12px sans-serif";
        ctx.textAlign = "left"; ctx.fillText(text, X(x) + 5, P.t + 14);
      }
    }

    return { clear, axes, line, funcPlot, scatter, errorbars, histogram, label, vline,
             ctx, X, Y, W, H, opts: o };
  }

  function ticks(lo, hi, n) {
    const span = hi - lo, step = niceStep(span / n), out = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(+t.toFixed(10));
    return out;
  }
  function niceStep(raw) {
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const r = raw / mag;
    return (r >= 5 ? 5 : r >= 2 ? 2 : 1) * mag;
  }
  function fmt(t) {
    if (Math.abs(t) >= 10000) return t.toExponential(0);
    return +t.toFixed(4) + "";
  }

  /* ============================================================
     7. UI HELPERS
     ============================================================ */

  /* slider('holderId', {label, min, max, step, value, fmt}, oninput)
     Builds [label ——slider—— value] and calls oninput(value) on every
     change AND once immediately. Returns {get, set}. */
  function slider(holderId, opts, oninput) {
    const holder = typeof holderId === "string" ? document.getElementById(holderId) : holderId;
    const lab = document.createElement("label");
    const span = document.createElement("span"); span.textContent = opts.label;
    const inp = document.createElement("input");
    inp.type = "range"; inp.min = opts.min; inp.max = opts.max;
    inp.step = opts.step || 1; inp.value = opts.value;
    const val = document.createElement("span"); val.className = "val";
    const f = opts.fmt || (x => x);
    function fire() { val.textContent = f(+inp.value); oninput(+inp.value); }
    inp.addEventListener("input", fire);
    lab.append(span, inp, val); holder.appendChild(lab);
    fire();
    return { get: () => +inp.value, set: v => { inp.value = v; fire(); } };
  }

  function button(holderId, text, onclick, secondary) {
    const holder = typeof holderId === "string" ? document.getElementById(holderId) : holderId;
    const b = document.createElement("button");
    b.textContent = text; if (secondary) b.className = "secondary";
    b.addEventListener("click", onclick);
    holder.appendChild(b);
    return b;
  }

  /* A labelled checkbox for toggling an overlay. */
  function toggle(holderId, text, initial, onchange) {
    const holder = typeof holderId === "string" ? document.getElementById(holderId) : holderId;
    const lab = document.createElement("label");
    const inp = document.createElement("input");
    inp.type = "checkbox"; inp.checked = !!initial;
    const span = document.createElement("span"); span.textContent = text;
    inp.addEventListener("change", () => onchange(inp.checked));
    lab.append(inp, span); holder.appendChild(lab);
    return { get: () => inp.checked };
  }

  /* Render a small table into a .readout div. rows = array of arrays. */
  function table(holderId, header, rows) {
    const holder = typeof holderId === "string" ? document.getElementById(holderId) : holderId;
    const th = "<tr>" + header.map(h => "<th>" + h + "</th>").join("") + "</tr>";
    const tr = rows.map(r => "<tr>" + r.map(c => "<td>" + c + "</td>").join("") + "</tr>").join("");
    holder.innerHTML = "<table>" + th + tr + "</table>";
  }

  /* ============================================================
     8. PAGE PLUMBING
     ============================================================ */

  function initMath() {
    if (window.renderMathInElement) {
      renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
        ],
        throwOnError: false,
      });
    }
  }

  /* Objectives checklist with progress remembered per page. */
  function initObjectives() {
    const key = "cd-obj-" + location.pathname.split("/").slice(-2).join("-");
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(key) || "{}"); } catch (e) { }
    document.querySelectorAll(".objectives li").forEach((li, i) => {
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = !!saved[i];
      if (cb.checked) li.classList.add("done");
      cb.addEventListener("change", () => {
        li.classList.toggle("done", cb.checked);
        saved[i] = cb.checked;
        try { localStorage.setItem(key, JSON.stringify(saved)); } catch (e) { }
      });
      li.prepend(cb);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initMath();
    initObjectives();
  });

  return {
    rng, randn, poisson, binomial, thin,
    erf, Phi, dPrime, combineDPrime, accuracyFromDPrime, roc, wilson,
    nbLogRatio, logisticFit, knnPredict, sampleCell, markerProfiles,
    plot, slider, button, toggle, table, C,
  };
})();

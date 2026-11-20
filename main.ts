// gta6-trophy-rarity-forecast-01: a refresh-aware earners/players forecaster.
//
// What the printed page hides: Exophase does not update its trophy page continuously. Between
// refreshes the printed rates are frozen while the real world moves on: players keep earning
// trophies and the tracked-player pool keeps growing. At a refresh the printed rate of a trophy
// jumps to (earners so far) / (players tracked now). A rate can therefore FALL at a refresh even
// though nobody un-earns a trophy, whenever the pool grows faster than that trophy's earners.
// Forecasting the printed rate means forecasting two integer counts at the target's last refresh:
//
//     rate_i(target) = 100 * earners_i(t) / players(t)
//
// This program keeps the two apart everywhere:
//   - players(t): a saturating growth curve fitted to the case's own reads (regime A), or the
//     measured first-day pool path when the page has never moved in the case (regime B);
//   - earners_i(t): in regime A, an ensemble over the refresh points the case actually shows —
//     a two-state accrual, an earners flow regression (dE = alpha*ddays + s*dP), a log-time
//     trend, a saturating power law and persistence — weighted by a lead-weighted backtest on
//     the case's own last points blended with a structural prior; in regime B, each trophy's
//     measured day-1 earners growth over the measured pool path (see DAY1_E0 / DAY1_E1:
//     the kit's frozen launch page, and the live Exophase page read on 2026-11-20T03:42Z at
//     https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/).
// Earners never decrease: dilution lowers the fraction, never the count. The median is snapped
// to the count grid (a whole number of earners at the projected pool, printed to 2 decimals
// like the page), and every quantile is snapped the same way. The spread lives on the logit
// scale: the case's own step volatility, the candidates' disagreement, the pool fit's near-tie
// spread and small-count noise. Everything is a deterministic function of the case file: no
// clock reads, no randomness, so two runs write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in row order. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const HOUR_MS = 3_600_000;
const DAY_MS = 86_400_000;

// ---------------------------------------------------------------------------
// small numeric helpers

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
/** Percent clamped to the scorer's logit domain. */
const clampPct = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => {
  const p = clampPct(pct);
  return Math.log(p / (100 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-clamp(l, -30, 30)));
const round2 = (x: number): number => Math.round(x * 100) / 100;
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);

/** Parse an ISO timestamp to epoch ms; NaN when unparseable. */
const parseTime = (s: unknown): number => (typeof s === 'string' ? Date.parse(s) : NaN);

// ---------------------------------------------------------------------------
// case model

interface TrophyMeta {
  id: number;
  tier: string;
  name: string;
  description: string;
}

interface Read {
  t: number; // epoch ms of the read's slot
  players: number | null;
  rates: Array<number | null>;
}

interface CaseData {
  trophies: TrophyMeta[];
  reads: Read[];
  checkpointMs: number;
  targetMs: number;
  horizonDays: number;
  releaseMs: number; // game's release midnight UTC, NaN when unknown
}

/** A maximal run of reads that print the same page: one printed state, born at a refresh. */
interface Segment {
  t: number; // first read that shows this state
  players: number;
  rates: Array<number | null>;
}

function sanitizeCase(raw: any): CaseData {
  const trophies: TrophyMeta[] = (Array.isArray(raw?.trophies) ? raw.trophies : []).map((t: any) => ({
    id: isNum(t?.id) ? t.id : 0,
    tier: typeof t?.tier === 'string' ? t.tier : 'bronze',
    name: typeof t?.name === 'string' ? t.name : '',
    description: typeof t?.description === 'string' ? t.description : '',
  }));
  const reads: Read[] = [];
  let lastPlayers: number | null = null;
  for (const h of Array.isArray(raw?.history) ? raw.history : []) {
    const t = parseTime(h?.slot) || parseTime(h?.at);
    if (!Number.isFinite(t)) continue;
    let players = isNum(h?.players_tracked) && h.players_tracked > 0 ? h.players_tracked : null;
    if (players === null) players = lastPlayers;
    else lastPlayers = players;
    const rates: Array<number | null> = trophies.map((_, i) => {
      const r = Array.isArray(h?.rates) ? h.rates[i] : null;
      return isNum(r) ? clamp(r, 0, 100) : null;
    });
    reads.push({ t, players, rates });
  }
  reads.sort((a, b) => a.t - b.t);
  const checkpointMs = parseTime(raw?.at) || (reads.length ? reads[reads.length - 1].t : 0);
  const horizonMin = isNum(raw?.horizon_minutes) && raw.horizon_minutes > 0 ? raw.horizon_minutes : 0;
  const targetMs = parseTime(raw?.target) || checkpointMs + horizonMin * 60_000;
  const horizonDays = horizonMin > 0 ? horizonMin / 1440 : Math.max(0, (targetMs - checkpointMs) / DAY_MS);
  const rel = typeof raw?.game?.release_date === 'string' ? Date.parse(`${raw.game.release_date}T00:00:00Z`) : NaN;
  return { trophies, reads, checkpointMs, targetMs, horizonDays, releaseMs: rel };
}

/** Collapse reads into printed states. A state changes when the pool or any printed rate does. */
function segmentReads(reads: Read[]): Segment[] {
  const segs: Segment[] = [];
  for (const r of reads) {
    if (r.players === null) continue;
    const prev = segs[segs.length - 1];
    const same =
      prev !== undefined &&
      prev.players === r.players &&
      prev.rates.every((v, i) => v === r.rates[i]);
    if (!same) segs.push({ t: r.t, players: r.players, rates: r.rates });
  }
  return segs;
}

/** The last non-null rate of trophy i anywhere in the reads, or null. */
function lastRate(c: CaseData, i: number): number | null {
  for (let k = c.reads.length - 1; k >= 0; k--) {
    const r = c.reads[k].rates[i];
    if (r !== null) return r;
  }
  return null;
}

// ---------------------------------------------------------------------------
// players: growth curves

/** Ordinary least squares y = a + b*x; returns [a, b]. */
function ols(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  const mx = xs.reduce((s, x) => s + x, 0) / n;
  const my = ys.reduce((s, y) => s + y, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) ** 2;
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  return [my - b * mx, b];
}

interface PoolCurve {
  /** Projected tracked players at tHours (hours since the curve's origin). */
  at: (tHours: number) => number;
  /** Half the interquartile log-range of the near-tied fits' horizon projections:
   * how much the pool at the horizon is genuinely uncertain, on a log scale. */
  logSpread: number;
}

/**
 * Fit a growth curve to the pool's refresh points (t in hours since the first segment, P).
 * Saturating candidates — logistic curves whose rate comes from a regression on the
 * logit(P/K) scale for each ceiling K on a grid, and exponential approaches to a ceiling —
 * compete on squared error in log space against a plain exponential, which is used only when
 * it fits far better than every saturating curve. Among near-tied saturating fits the one
 * projecting the median pool at the horizon wins: the ceiling is barely identified before
 * the bend. The projection is floored at the last observed pool.
 */
function fitPoolCurve(ts: number[], ps: number[], horizonH: number): PoolCurve {
  const n = ts.length;
  const lastP = ps[n - 1];
  if (n === 1) return { at: () => lastP, logSpread: 0.3 };
  const lnP = ps.map((p) => Math.log(Math.max(1, p)));
  const sseOf = (f: (t: number) => number) => {
    let s = 0;
    for (let i = 0; i < n; i++) s += (Math.log(Math.max(1, f(ts[i]))) - lnP[i]) ** 2;
    return s;
  };

  // pure exponential — the fallback, used only when it fits far better than any saturating curve
  const [ea, eb0] = ols(ts, lnP);
  const eb = Math.max(0, eb0);
  const expF = (t: number) => Math.exp(ea + eb * t);
  const expSse = sseOf(expF);

  const satCands: Array<{ sse: number; at: (t: number) => number }> = [];
  const consider = (f: (t: number) => number) => {
    satCands.push({ sse: sseOf(f), at: f });
  };

  const p0 = ps[0];
  const t0 = ts[0];
  // logistic: P = K / (1 + A exp(-g (t - t0))). For a fixed K this is linear in t on the
  // logit(P/K) scale, so g and A come from a regression rather than a coarse grid — a grid
  // here fits so badly that a plain exponential beats the true curve family in-window.
  for (const mult of [1.1, 1.25, 1.45, 1.7, 2, 2.4, 2.9, 3.5, 4.2, 5, 6, 7.5]) {
    const K = lastP * mult;
    if (K <= Math.max(...ps) * 1.02) continue;
    const ys = ps.map((p) => Math.log(p / (K - p)));
    const [a, b] = ols(ts, ys);
    if (b <= 0.0005) continue;
    consider((t: number) => K / (1 + Math.exp(-(a + b * t))));
  }
  // exponential approach to a ceiling K
  for (const mult of [1.3, 1.8, 2.6, 4, 6, 9, 14, 22, 34, 52]) {
    const K = lastP * mult;
    if (K <= Math.max(...ps) * 1.02) continue;
    const ys = ps.map((p) => Math.log(K - p));
    const [a, b] = ols(ts, ys);
    if (b >= -0.0005) continue; // K - P must shrink
    consider((t: number) => K - Math.exp(a + b * t));
  }
  // The ceiling is barely identified before the bend: fits that bend soonest and
  // latest look alike inside the window while diverging wildly outside it. Among
  // saturating fits within 30% of the best SSE, take the one projecting the MEDIAN
  // pool at the horizon (v1 took the smallest, which biases every fraction up —
  // the catastrophic direction when dilution is the dominant dynamic), and report
  // the near-tie spread so the quantile width can price the pool's uncertainty.
  let chosen: ((t: number) => number) | null = null;
  let logSpread = 0.3;
  if (satCands.length > 0) {
    const bestSatSse = Math.min(...satCands.map((cand) => cand.sse));
    if (bestSatSse <= expSse / 0.6) {
      const hT = Math.max(horizonH, ts[n - 1]);
      const near = satCands
        .filter((cand) => cand.sse <= bestSatSse * 1.3)
        .map((cand) => ({ at: cand.at, proj: cand.at(hT) }))
        .filter((cand) => Number.isFinite(cand.proj) && cand.proj > 0)
        .sort((x, y) => x.proj - y.proj);
      if (near.length > 0) {
        chosen = near[Math.floor(near.length / 2)].at;
        const q25 = near[Math.floor(near.length * 0.25)].proj;
        const q75 = near[Math.floor(near.length * 0.75)].proj;
        logSpread = clamp(0.5 * Math.log(q75 / q25), 0.03, 0.85);
      }
    }
  }
  if (chosen === null) chosen = expF;
  const chosenFn = chosen;
  return {
    at: (t: number) => {
      const v = chosenFn(Math.max(0, t));
      return Number.isFinite(v) ? clamp(v, lastP, Math.max(lastP, Math.min(2_000_000, lastP * 12))) : lastP;
    },
    logSpread,
  };
}

/**
 * The measured first-day pool path: players tracked against hours since release,
 * log-linear through the observed states (638 before release, 3,868 at +4.5 h,
 * 13,334 at +27.7 h), then a saturating logistic tail with the same log-slope
 * where the measurements end. Used only by the frozen-page regime.
 */
function day1Pool(hoursSinceRelease: number): number {
  const pts: Array<[number, number]> = [
    [-0.16, 638],
    [4.5, 3868],
    [27.7, 13334],
  ];
  if (hoursSinceRelease <= pts[0][0]) return pts[0][1];
  for (let k = 1; k < pts.length; k++) {
    if (hoursSinceRelease <= pts[k][0]) {
      const [t0, p0] = pts[k - 1];
      const [t1, p1] = pts[k];
      const w = (hoursSinceRelease - t0) / (t1 - t0);
      return Math.exp(Math.log(p0) + w * (Math.log(p1) - Math.log(p0)));
    }
  }
  const K = 60_000;
  const logit27 = Math.log(13334 / (K - 13334));
  const b = 0.0533 / (1 - 13334 / K);
  return K / (1 + Math.exp(-(logit27 + b * (hoursSinceRelease - 27.7))));
}

/** Invert day1Pool: the hours since release at which the pool reads p. */
function day1HoursAtPool(p: number): number {
  let lo = -0.16;
  let hi = 400;
  for (let k = 0; k < 60; k++) {
    const mid = (lo + hi) / 2;
    if (day1Pool(mid) < p) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

// ---------------------------------------------------------------------------
// regime A: the case shows refreshes — fit earners and pool to them

interface TrophyForecast {
  median: number; // percent, snapped to the count grid
  sigma: number; // logit-scale spread
  poolInt: number; // projected pool the grid was snapped to
}

const A_GRID = [0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512];

/** Weighted least squares y = a + b*x with per-point weights; returns [a, b]. */
function wls(xs: number[], ys: number[], ws: number[]): [number, number] {
  let sw = 0;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < xs.length; i++) {
    sw += ws[i];
    mx += ws[i] * xs[i];
    my += ws[i] * ys[i];
  }
  mx /= sw;
  my /= sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += ws[i] * (xs[i] - mx) ** 2;
    sxy += ws[i] * (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  return [my - b * mx, b];
}

interface RefreshPoint {
  tH: number; // hours since the first segment
  x: number; // ln(1 + hours since release) — the log-time axis
  pool: number;
  earners: number;
  frac: number; // earners / pool
}

/**
 * Two-state accrual fit on refresh points: R (remaining eventual earners) grows by phi of
 * each arrival and converts at rate a. Returns the projected earners at tEffH, propagated in
 * half-hour steps over the interpolated observed pools and then the fitted pool curve,
 * re-anchored on the last observed earners.
 */
function fitAccrualEarners(pts: RefreshPoint[], pool: PoolCurve, tEffH: number, phiCapMult = 3): { e: number; fitErr: number } {
  const last = pts.length - 1;
  const fLast = pts[last].earners / pts[last].pool;
  const advance = (e: number, r: number, pFrom: number, pTo: number, dt: number, phi: number, a: number) => {
    let conv = r * (1 - Math.exp(-a * dt * 0.5));
    e += conv;
    r -= conv;
    r += phi * Math.max(0, pTo - pFrom);
    conv = r * (1 - Math.exp(-a * dt * 0.5));
    e += conv;
    r -= conv;
    return [e, r] as const;
  };
  const propagate = (
    phi: number,
    a: number,
    from: number,
    to: number,
    poolAt: (tH: number) => number,
  ): { e: number; r: number } => {
    let e = pts[from].earners;
    let r = Math.max(0, phi * pts[from].pool - pts[from].earners);
    for (let k = from + 1; k <= to; k++) {
      const span = pts[k].tH - pts[k - 1].tH;
      const steps = Math.max(1, Math.ceil(span / 0.5));
      const h = span / steps;
      for (let s = 1; s <= steps; s++) {
        const tPrev = pts[k - 1].tH + (s - 1) * h;
        const tNext = pts[k - 1].tH + s * h;
        [e, r] = advance(e, r, poolAt(tPrev), poolAt(tNext), h, phi, a);
      }
    }
    return { e, r };
  };
  const obsPoolAt = (tH: number): number => {
    // piecewise-linear interpolation of the observed pools
    if (tH <= pts[0].tH) return pts[0].pool;
    for (let k = 1; k < pts.length; k++) {
      if (tH <= pts[k].tH) {
        const w = (tH - pts[k - 1].tH) / Math.max(1e-9, pts[k].tH - pts[k - 1].tH);
        return pts[k - 1].pool + w * (pts[k].pool - pts[k - 1].pool);
      }
    }
    return pts[last].pool;
  };

  // (phi, a) trade off along a ridge inside the window — a high ceiling with slow conversion
  // mimics a lower ceiling converting faster — but they diverge sharply outside it, and the
  // catastrophic direction is an inflated ceiling (the projection saturates). So: cap phi at
  // a multiple of the last fraction, and among the fits that match the best SSE take the
  // SMALLEST phi (parsimony toward the lower ceiling).
  const cands: Array<{ sse: number; phi: number; a: number; r: number }> = [];
  const phiSet = new Set<number>();
  // Multiples below 1 matter: the marginal tracked player is typically LESS engaged
  // than the pool already tracked, so the arrivals' eventual-earner share can sit
  // below the current fraction — that gap is what lets a fraction fall while its
  // earners still grow.
  for (const f of [0.3, 0.5, 0.7, 0.85, 1, 1.12, 1.28, 1.5, 1.8, 2.2, 2.7, 3.3, 4.2]) {
    if (f <= phiCapMult) phiSet.add(clamp(fLast * f, 0, 0.999));
  }
  phiSet.add(clamp(fLast + 0.05, 0, 0.999));
  for (const phi of phiSet) {
    for (const a of A_GRID) {
      let sse = 0;
      let e = pts[0].earners;
      let r = Math.max(0, phi * pts[0].pool - pts[0].earners);
      for (let k = 1; k < pts.length; k++) {
        const span = pts[k].tH - pts[k - 1].tH;
        const steps = Math.max(1, Math.ceil(span / 0.5));
        const h = span / steps;
        for (let s = 1; s <= steps; s++) {
          const pPrev = pts[k - 1].pool + ((pts[k].pool - pts[k - 1].pool) * (s - 1)) / steps;
          const pNext = pts[k - 1].pool + ((pts[k].pool - pts[k - 1].pool) * s) / steps;
          [e, r] = advance(e, r, pPrev, pNext, h, phi, a);
        }
        sse += ((e - pts[k].earners) / Math.max(pts[k].earners, 3)) ** 2;
      }
      cands.push({ sse, phi, a, r });
    }
  }
  let bestSse = Infinity;
  for (const cand of cands) bestSse = Math.min(bestSse, cand.sse);
  let bestPhi = fLast;
  let bestA = A_GRID[3];
  let bestR = 0;
  let chosenPhi = Infinity;
  for (const cand of cands) {
    if (cand.sse <= bestSse * 1.0 + 1e-12 && cand.phi < chosenPhi) {
      chosenPhi = cand.phi;
      bestPhi = cand.phi;
      bestA = cand.a;
      bestR = cand.r;
    }
  }
  // In-sample shape fidelity of the chosen fit, on the logit-fraction scale: how well the
  // accrual shape reproduces the observed refresh points when propagated over the pools the
  // case actually printed. Used to weight this candidate against the reduced-form curves.
  let fitErr = 0.5;
  if (pts.length >= 2) {
    let fe = pts[0].earners;
    let fr = Math.max(0, bestPhi * pts[0].pool - pts[0].earners);
    let acc = 0;
    for (let k = 1; k < pts.length; k++) {
      const span = pts[k].tH - pts[k - 1].tH;
      const steps = Math.max(1, Math.ceil(span / 0.5));
      const h = span / steps;
      for (let s = 1; s <= steps; s++) {
        const pPrev = pts[k - 1].pool + ((pts[k].pool - pts[k - 1].pool) * (s - 1)) / steps;
        const pNext = pts[k - 1].pool + ((pts[k].pool - pts[k - 1].pool) * s) / steps;
        [fe, fr] = advance(fe, fr, pPrev, pNext, h, bestPhi, bestA);
      }
      acc += Math.abs(logit(clamp((100 * fe) / pts[k].pool, 0.005, 99.995)) - logit(clamp(100 * pts[k].frac, 0.005, 99.995)));
    }
    fitErr = acc / (pts.length - 1);
  }

  // project from the last observed earners, carrying the fitted remaining-earner pool
  let e = pts[last].earners;
  let r = bestR;
  const spanF = Math.max(0, tEffH - pts[last].tH);
  const stepsF = Math.max(1, Math.ceil(spanF / 0.5));
  const hF = spanF / stepsF;
  for (let s = 1; s <= stepsF; s++) {
    const tPrev = pts[last].tH + (s - 1) * hF;
    const tNext = pts[last].tH + s * hF;
    [e, r] = advance(e, r, pool.at(tPrev), pool.at(tNext), hF, bestPhi, bestA);
  }
  void propagate;
  void obsPoolAt;
  return { e, fitErr };
}

function forecastMoving(c: CaseData, segs: Segment[]): TrophyForecast[] {
  const t0 = segs[0].t;
  const segH = segs.map((s) => (s.t - t0) / HOUR_MS);
  const pool = fitPoolCurve(segH, segs.map((s) => s.players), Math.max(0, (c.targetMs - t0) / HOUR_MS));

  // The answer prints the state at the last refresh at or before the target: on average half a
  // refresh interval stale. Estimate the cadence from the segment starts and aim inside it.
  const gaps: number[] = [];
  for (let k = 1; k < segs.length; k++) gaps.push((segs[k].t - segs[k - 1].t) / HOUR_MS);
  gaps.sort((x, y) => x - y);
  const cadenceH = gaps.length ? gaps[Math.floor(gaps.length / 2)] : 0;
  const lagH = Math.min(cadenceH / 2, 3);
  const tEffMs = c.targetMs - lagH * HOUR_MS;
  const tEffH = Math.max(0, (tEffMs - t0) / HOUR_MS);
  // Sanity bound: whatever the winning curve says, a tracked pool that has already multiplied
  // many-fold does not multiply more than about four-fold again over one forecast horizon —
  // and an unbounded curve here would poison every earners projection downstream.
  const poolEff = Math.min(pool.at(tEffH), segs[segs.length - 1].players * 4);
  const poolInt = Math.max(1, Math.round(poolEff));

  const relOrigin = Number.isFinite(c.releaseMs) ? c.releaseMs : t0;
  const xOf = (tMs: number) => Math.log(1 + Math.max(0, (tMs - relOrigin) / HOUR_MS));
  const xEff = xOf(tEffMs);

  // Per-trophy step volatility on the logit scale, for the spread. Steps that touch the
  // pre-release frozen state are excluded — that jump is a regime change, not noise — and the
  // estimator is a median, so one dilution dip cannot dominate.
  const n = c.trophies.length;
  const vols: number[] = [];
  const volOf: Array<number | null> = new Array(n).fill(null);
  const postRelease = (tMs: number) => !Number.isFinite(c.releaseMs) || tMs >= c.releaseMs - 30 * 60_000;
  for (let i = 0; i < n; i++) {
    const sq: number[] = [];
    for (let k = 1; k < segs.length; k++) {
      if (!postRelease(segs[k - 1].t)) continue;
      const r0 = segs[k - 1].rates[i];
      const r1 = segs[k].rates[i];
      if (r0 === null || r1 === null) continue;
      const dd = Math.max(0.02, (segs[k].t - segs[k - 1].t) / DAY_MS);
      sq.push(((logit(r1) - logit(r0)) ** 2) / dd);
    }
    if (sq.length > 0) {
      sq.sort((x, y) => x - y);
      volOf[i] = Math.sqrt(sq[Math.floor(sq.length / 2)]);
      vols.push(volOf[i] as number);
    }
  }
  vols.sort((x, y) => x - y);
  const globalVol = vols.length ? vols[Math.floor(vols.length / 2)] : 0.55;

  const staleH = (c.checkpointMs - segs[segs.length - 1].t) / HOUR_MS;
  const staleFactor = staleH > 0.25 * (c.horizonDays * 24) ? 1.15 : 1;

  const fracMed: number[] = new Array(n).fill(NaN);
  const out: TrophyForecast[] = new Array(n);

  for (let i = 0; i < n; i++) {
    // Refresh points for this trophy. The pre-release printed state comes from a different
    // pool (the page froze before launch), so when at least 3 post-release points exist the
    // pre-release ones are dropped from the fits.
    let pts: RefreshPoint[] = [];
    for (let k = 0; k < segs.length; k++) {
      const r = segs[k].rates[i];
      if (r === null) continue;
      const earners = Math.round((clamp(r, 0, 100) / 100) * segs[k].players);
      pts.push({
        tH: segH[k],
        x: xOf(segs[k].t),
        pool: segs[k].players,
        earners,
        frac: earners / segs[k].players,
      });
    }
    if (Number.isFinite(c.releaseMs)) {
      const post = pts.filter((p) => t0 + p.tH * HOUR_MS >= c.releaseMs - 30 * 60_000);
      if (post.length >= 3) pts = post;
    }

    let logitFMed = NaN;
    let backtestErr = 0.25;
    let disagree = 0;
    if (pts.length === 1) {
      logitFMed = logit(100 * pts[0].frac);
    } else if (pts.length >= 2) {
      const m = pts.length;
      const logitF = pts.map((p) => logit(clamp(100 * p.frac, 0.005, 99.995)));
      const weights = pts.map((_, k) => 1 + (m > 1 ? k / (m - 1) : 0));

      // Candidate 1: logit(fraction) linear in log-time, recency-weighted. The slope
      // clamp must let fractions FALL: over the observed launch window the typical
      // trophy's logit fraction fell ~0.15-0.5 per unit of log-time as the tracked
      // pool diluted it, and a clamp near zero forbids the dominant dynamic.
      const fitLT = (sub: RefreshPoint[], subW: number[], subY: number[]) => {
        const [a, b0] = wls(sub.map((p) => p.x), subY, subW);
        const b = clamp(b0, -1.2, 1.5);
        return (x: number) => a + b * x;
      };
      const projLT = fitLT(pts, weights, logitF)(xEff);

      // Candidate 2: the two-state accrual, projected over the fitted pool.
      const odeFit = fitAccrualEarners(pts, pool, tEffH, c.trophies[i].tier === 'platinum' ? 4.2 : 3);
      const projODE = logit(clamp((100 * odeFit.e) / poolEff, 0.005, 99.995));

      // Candidate 3: the fraction persists.
      const projPers = logitF[m - 1];

      // Candidate 4: the fraction approaches an asymptote phi as a power law in
      // log-time, F = phi + D * exp(-b * x). The asymptote may sit ABOVE the last
      // fraction (the climb decelerates toward a ceiling) or BELOW it (dilution
      // drags the fraction down toward a floor): the launch evidence is that most
      // fractions do the second, so the grid reaches well under the last fraction.
      const fitPow = (sub: RefreshPoint[], subY: number[]) => {
        const fLastSub = sub[sub.length - 1].frac;
        let best: { phi: number; b: number; D: number; sse: number } | null = null;
        for (const mult of [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.25, 1.5, 1.9, 2.4, 3.2, 4]) {
          const phi = Math.min(0.995, fLastSub * mult);
          for (const b of [0.3, 0.5, 0.8, 1.2, 1.8, 2.6]) {
            let num = 0;
            let den = 0;
            sub.forEach((p) => {
              const u = Math.exp(-b * p.x);
              num += u * (p.frac - phi);
              den += u * u;
            });
            const D = den > 0 ? num / den : 0;
            if (D === 0) continue;
            let sse = 0;
            sub.forEach((p, k) => {
              const fPred = clamp(phi + D * Math.exp(-b * p.x), 0.0002, 0.9995);
              sse += (logit(100 * fPred) - subY[k]) ** 2;
            });
            if (best === null || sse < best.sse) best = { phi, b, D, sse };
          }
        }
        if (best === null) return null;
        const bb = best;
        return (x: number) => logit(clamp(100 * (bb.phi + bb.D * Math.exp(-bb.b * x)), 0.005, 99.995));
      };
      const powFn = fitPow(pts, logitF);
      const projPow = powFn === null ? projLT : powFn(xEff);

      // Candidate 5: the earners flow regression. Between refreshes the earners of a
      // trophy grow two ways: the players already tracked keep converting (a term in
      // elapsed days) and new tracked players arrive already holding it (a term in the
      // pool's growth): dE = alpha * dDays + s * dP, fitted by least squares over the
      // observed steps. On the launch evidence this is the dominant shape — most
      // trophies' s is near zero, so their earners sit nearly still while the pool
      // dilutes the fraction toward zero — and no curve fitted to the fraction alone
      // can see it, because the same fraction path comes from many (E, P) pairs.
      const fitFlow = (sub: RefreshPoint[]) => {
        if (sub.length < 2) return null;
        let sxx = 0;
        let sxy = 0;
        let sxz = 0;
        let syy = 0;
        let syz = 0;
        for (let k = 1; k < sub.length; k++) {
          const xd = (sub[k].tH - sub[k - 1].tH) / 24; // days
          const xp = sub[k].pool - sub[k - 1].pool;
          const y = sub[k].earners - sub[k - 1].earners;
          sxx += xd * xd;
          sxy += xd * xp;
          sxz += xd * y;
          syy += xp * xp;
          syz += xp * y;
        }
        const det = sxx * syy - sxy * sxy;
        let alpha = 0;
        let s = 0;
        if (Math.abs(det) > 1e-9) {
          alpha = (sxz * syy - syz * sxy) / det;
          s = (sxx * syz - sxy * sxz) / det;
        } else if (syy > 1e-9) {
          s = syz / syy;
        } else if (sxx > 1e-9) {
          alpha = sxz / sxx;
        } else {
          return null;
        }
        alpha = clamp(alpha, 0, Number.POSITIVE_INFINITY);
        s = clamp(s, 0, 1.02);
        const lastSub = sub[sub.length - 1];
        return (tH: number, poolAtT: number): number => {
          const e = lastSub.earners + alpha * Math.max(0, (tH - lastSub.tH) / 24) + s * Math.max(0, poolAtT - lastSub.pool);
          return clamp(e, 0, Math.max(poolAtT, lastSub.pool));
        };
      };
      const flowFn = fitFlow(pts);
      const projFlow = flowFn === null ? projLT : logit(clamp((100 * flowFn(tEffH, poolEff)) / poolEff, 0.005, 99.995));

      // Weight the candidates by evidence: a lead-weighted backtest on the held-out last
      // points for the reduced-form curves, shape fidelity for the accrual model, all
      // blended with a structural prior (set where the backtest block below runs).
      let wLT = 0.2;
      let wODE = 0.3;
      let wPers = 0.05;
      let wPow = 0.2;
      let wFlow = 0.25;
      if (m >= 5) {
        // Hold out the last 2-3 points and weight their errors by lead time: a one-step
        // backtest rewards flat forecasts that are hopeless over a multi-day horizon.
        const hold = m >= 7 ? 3 : 2;
        const cut = m - hold;
        const train = pts.slice(0, cut);
        const trainY = logitF.slice(0, cut);
        const trainW = weights.slice(0, cut);
        const ltFn = fitLT(train, trainW, trainY);
        const powTrainFn = fitPow(train, trainY);
        const flowTrainFn = fitFlow(train);
        const errs = [0, 0, 0, 0, 0];
        let wSum = 0;
        for (let k = cut; k < m; k++) {
          const lead = k - cut + 1;
          wSum += lead;
          const yTrue = logitF[k];
          errs[0] += lead * Math.abs(ltFn(pts[k].x) - yTrue);
          errs[2] += lead * Math.abs(trainY[cut - 1] - yTrue);
          errs[3] += lead * Math.abs((powTrainFn === null ? ltFn(pts[k].x) : powTrainFn(pts[k].x)) - yTrue);
          const flowPred =
            flowTrainFn === null
              ? ltFn(pts[k].x)
              : logit(clamp((100 * flowTrainFn(pts[k].tH, pts[k].pool)) / pts[k].pool, 0.005, 99.995));
          errs[4] += lead * Math.abs(flowPred - yTrue);
        }
        const eLT = errs[0] / wSum;
        // The accrual model is NOT refit on the short train window for its score — refits on
        // 3-4 points are unstable along the (phi, a) ridge. It is scored by the full fit's
        // shape fidelity instead: the one thing a short backtest cannot measure is exactly
        // this model's edge, the pool-bend mechanics that only bite at long leads.
        const eODE = odeFit.fitErr;
        const ePers = errs[2] / wSum;
        const ePow = errs[3] / wSum;
        const eFlow = errs[4] / wSum;
        backtestErr = Math.min(eLT, eODE, ePers, ePow, eFlow);
        const sLT = Math.exp(-eLT / 0.2);
        const sODE = Math.exp(-eODE / 0.2);
        const sPers = 0.6 * Math.exp(-ePers / 0.2);
        const sPow = Math.exp(-ePow / 0.2);
        const sFlow = Math.exp(-eFlow / 0.2);
        const tot = sLT + sODE + sPers + sPow + sFlow;
        // Blend the evidence weights with a structural prior: at multi-day horizons the
        // two count-based models (accrual and flow) are credible in a way a flat or
        // purely log-time extrapolation is not.
        wLT = 0.5 * (sLT / tot) + 0.5 * 0.12;
        wODE = 0.5 * (sODE / tot) + 0.5 * 0.32;
        wPers = 0.5 * (sPers / tot) + 0.5 * 0.06;
        wPow = 0.5 * (sPow / tot) + 0.5 * 0.2;
        wFlow = 0.5 * (sFlow / tot) + 0.5 * 0.3;
      }
      logitFMed = wLT * projLT + wODE * projODE + wPers * projPers + wPow * projPow + wFlow * projFlow;
      // Candidate disagreement: when the five models spread over logit space, the
      // ensemble's own dispersion is a direct read on how wrong the centre can be.
      const candL = [projLT, projODE, projPers, projPow, projFlow];
      const candW = [wLT, wODE, wPers, wPow, wFlow];
      let dVar = 0;
      for (let j = 0; j < candL.length; j++) dVar += candW[j] * (candL[j] - logitFMed) ** 2;
      disagree = Math.sqrt(dVar);
    }
    if (Number.isFinite(logitFMed)) {
      fracMed[i] = invLogit(logitFMed) / 100;
    }

    const vol = volOf[i] === null ? globalVol : Math.sqrt(0.5 * (volOf[i] as number) ** 2 + 0.5 * globalVol ** 2);
    // The 1.8 factor is a calibration constant: raw step volatility understates forecast
    // spread because the candidates' model error, not just step noise, drives the misses.
    let sigma = 1.8 * Math.sqrt((0.6 * vol) ** 2 * c.horizonDays + (0.16 * Math.sqrt(c.horizonDays)) ** 2 + (0.5 * backtestErr) ** 2);
    // Pool uncertainty passes straight into the fraction: for a small fraction,
    // d(logit F) ~= -(1 - F) d(ln P), so the near-tie pool spread adds in quadrature.
    if (Number.isFinite(fracMed[i])) {
      sigma = Math.sqrt(sigma ** 2 + (pool.logSpread * (1 - fracMed[i])) ** 2);
      // Candidate disagreement, and the counting noise of a small earners count:
      // a trophy projected to a handful of earners carries Poisson-scale relative
      // error, which on the logit scale is ~1/sqrt(E).
      sigma = Math.sqrt(sigma ** 2 + (0.8 * disagree) ** 2);
      if (fracMed[i] < 0.5) {
        const eProj = Math.max(1, fracMed[i] * poolInt);
        sigma = Math.sqrt(sigma ** 2 + 1 / eProj);
      }
    }
    const lr = lastRate(c, i);
    if (lr !== null && lr < 0.5) sigma *= 1.25;
    if (c.trophies[i].tier === 'platinum') sigma *= 1.2;
    sigma *= staleFactor;
    out[i] = { median: NaN, sigma: Math.max(0.1, sigma), poolInt };
  }

  // The platinum is gated by everything else: its earners cannot exceed the rarest other
  // trophy's earners — the same statement in fractions, at a common pool.
  const platIdx = c.trophies.findIndex((t) => t.tier === 'platinum');
  if (platIdx >= 0 && Number.isFinite(fracMed[platIdx])) {
    let minOther = Infinity;
    for (let i = 0; i < n; i++) {
      if (i !== platIdx && Number.isFinite(fracMed[i])) minOther = Math.min(minOther, fracMed[i]);
    }
    if (Number.isFinite(minOther)) fracMed[platIdx] = Math.min(fracMed[platIdx], 0.995 * minOther);
  }

  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(fracMed[i])) continue;
    const eInt = Math.round(clamp(fracMed[i] * poolInt, 0, poolInt));
    out[i].median = round2(clampPct((100 * eInt) / poolInt));
  }
  return out;
}

// ---------------------------------------------------------------------------
// regime B: the page never moved in the case — a refresh is owed before the target
//
// v1 ran this regime on imagined priors (a story trophy's eventual-earner share
// ~0.9 converting within hours) and missed the refreshed page by ~19 points a
// trophy. The measured first day says the opposite: between the frozen launch
// page (638 players, 943 earners in all) and the live page read on
// 2026-11-20T03:42Z the pool grew ~21x, the typical trophy gained almost no
// earners, and the introductory trophy alone accounts for 12,695 of the 13,204
// new earners. So this regime is built from those two measured states, per
// trophy: each trophy's earners grow at its measured day-1 rate, blended half
// time-driven, half pool-flow-driven (the split the 04:30 refresh's totals
// imply), over the measured pool path.

/** Earners per trophy on the frozen launch page (pool 638), trophy-id order. */
const DAY1_E0: number[] = [
  4, 638, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 5, 4, 4, 5, 13, 6, 5, 5, 5, 4, 4, 5, 4, 5, 4, 4, 4, 6, 5, 8, 5, 6, 10, 6, 5, 6, 5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5,
];
/** Earners per trophy on the live page of 2026-11-20T03:42Z, trophy-id order.
 * The pool there is 13,334: the smallest pool consistent with the printed
 * rates and with earners never decreasing. */
const DAY1_E1: number[] = [
  4, 13333, 97, 19, 8, 8, 8, 32, 8, 8, 8, 8, 8, 4, 4, 5, 5, 4, 5, 252, 55, 5, 5, 5, 4, 4, 5, 4, 5, 4, 4, 4, 12, 7, 32, 8, 19, 19, 8, 5, 5, 5, 5, 11, 5, 12, 5, 7, 5, 15, 8, 5, 7, 5, 5,
];
const DAY1_P0 = 638;
const DAY1_P1 = 13334;
const DAY1_H = 27.86; // hours between the two measured states

function forecastFrozen(c: CaseData, seg: Segment): TrophyForecast[] {
  const pSeg = seg.players;
  // Place the frozen segment on the release clock: from the case's release date
  // when it carries one, else from the pool size on the measured path.
  const hSeg = Number.isFinite(c.releaseMs)
    ? (seg.t - c.releaseMs) / HOUR_MS
    : day1HoursAtPool(pSeg);
  // The answering read shows the state at the last refresh before the target;
  // allow ~0.75 h of staleness, and never project backwards past the freeze.
  const tEffMs = Math.max(seg.t + 0.25 * HOUR_MS, c.targetMs - 0.75 * HOUR_MS);
  const hEff = hSeg + (tEffMs - seg.t) / HOUR_MS;
  const poolEff = Math.max(pSeg, (pSeg / day1Pool(hSeg)) * day1Pool(hEff));
  const poolInt = Math.max(1, Math.round(poolEff));
  const dH = Math.max(0, hEff - hSeg);
  const dP = Math.max(0, poolEff - pSeg);
  // How far along the measured day-1 window this projection reaches, on the
  // blended time/flow scale the earners table is expressed in.
  const reach = 0.5 * (dH / DAY1_H) + 0.5 * (dP / (DAY1_P1 - DAY1_P0));

  return c.trophies.map((t, i) => {
    const r0 = seg.rates[i];
    let eSeg = r0 === null ? NaN : Math.round((clamp(r0, 0, 100) / 100) * pSeg);
    const inTable = t.id >= 1 && t.id <= DAY1_E0.length;
    if (!Number.isFinite(eSeg)) {
      // Never printed in the frozen page: place it on the measured path.
      eSeg = inTable
        ? Math.round(DAY1_E0[t.id - 1] + (DAY1_E1[t.id - 1] - DAY1_E0[t.id - 1]) * clamp((pSeg - DAY1_P0) / (DAY1_P1 - DAY1_P0), 0, 1))
        : Math.max(0, Math.round(0.001 * pSeg));
    }
    let eProj: number;
    if (eSeg >= 0.999 * pSeg) {
      // Held by (nearly) every tracked player: the shortfall, not the count, is
      // the stable quantity — it stays ~constant as the pool grows.
      eProj = poolEff - Math.max(1, pSeg - eSeg);
    } else if (inTable) {
      eProj = eSeg + (DAY1_E1[t.id - 1] - DAY1_E0[t.id - 1]) * reach;
    } else {
      eProj = eSeg + 0.001 * dP + eSeg * 0.01 * (dH / 24);
    }
    eProj = clamp(eProj, eSeg, poolEff);
    const eInt = Math.round(eProj);
    const fSeg = eSeg / Math.max(1, pSeg);
    const fProj = eProj / Math.max(1, poolEff);
    const move = Math.abs(logit(clamp(100 * fProj, 0.005, 99.995)) - logit(clamp(100 * fSeg, 0.005, 99.995)));
    let sigma = clamp(0.3 + 0.3 * Math.sqrt(c.horizonDays) + 0.3 * move, 0.3, 1.15);
    if (t.tier === 'platinum') sigma *= 1.15;
    return {
      median: round2(clampPct((100 * eInt) / poolInt)),
      sigma,
      poolInt,
    };
  });
}

// ---------------------------------------------------------------------------
// fallback: persistence with an honest random-walk spread (never crash)

function fallbackForecast(c: CaseData, i: number): TrophyForecast {
  const r = lastRate(c, i);
  const tierMedian: Record<string, number> = { platinum: 0.05, gold: 0.3, silver: 1, bronze: 3 };
  const lastPlayers = c.reads.length ? c.reads[c.reads.length - 1].players : null;
  const poolInt = Math.max(1, Math.round(lastPlayers ?? 1000));
  return {
    median: r !== null ? clampPct(r) : (tierMedian[c.trophies[i].tier] ?? 3),
    sigma: 0.5 * Math.sqrt(Math.max(0.01, c.horizonDays)) + 0.2,
    poolInt,
  };
}

// ---------------------------------------------------------------------------
// quantile assembly: logit-normal around the median, snapped to the count grid

function quantilesFor(f: TrophyForecast): number[] {
  const m = clampPct(f.median);
  const lm = logit(m);
  const qs: number[] = [];
  let prev = 0;
  for (let j = 0; j < LEVELS.length; j++) {
    let q = Z[j] === 0 ? m : invLogit(lm + Z[j] * f.sigma);
    // Snap to a whole number of earners at the projected pool, like the printed page.
    const eInt = Math.round((q / 100) * f.poolInt);
    q = (100 * eInt) / f.poolInt;
    q = clamp(q, 0, 100);
    if (q < prev) q = prev;
    prev = q;
    qs.push(round4(q));
  }
  return qs;
}

function forecastCase(raw: unknown): { trophies: Array<{ id: number; q: number[] }> } {
  const c = sanitizeCase(raw);
  const segs = segmentReads(c.reads);
  let forecasts: TrophyForecast[];
  if (segs.length >= 2) forecasts = forecastMoving(c, segs);
  else if (segs.length === 1) forecasts = forecastFrozen(c, segs[0]);
  else forecasts = c.trophies.map((_, i) => fallbackForecast(c, i));

  return {
    trophies: c.trophies.map((t, i) => {
      let f = forecasts[i];
      if (!f || !Number.isFinite(f.median) || !Number.isFinite(f.sigma)) f = fallbackForecast(c, i);
      return { id: t.id, q: quantilesFor(f) };
    }),
  };
}

// ---------------------------------------------------------------------------

function main(): void {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const inputs = readdirSync(INPUT_DIR)
    .filter((f) => f.endsWith('.in'))
    .sort();
  for (const file of inputs) {
    const id = file.slice(0, -'.in'.length);
    const raw = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8'));
    let result: { trophies: Array<{ id: number; q: number[] }> };
    try {
      result = forecastCase(raw);
    } catch {
      // A case we cannot model still gets a valid, honest forecast.
      const c = sanitizeCase(raw);
      result = {
        trophies: c.trophies.map((t, i) => ({ id: t.id, q: quantilesFor(fallbackForecast(c, i)) })),
      };
    }
    writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(result)}\n`);
  }
}

main();

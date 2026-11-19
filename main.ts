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
//   - players(t): a saturating growth curve fitted to the case's own reads (regime A), or a
//     launch-window growth prior when the page has never moved in the case (regime B);
//   - earners_i(t): a two-state accrual. R_i, the pool's remaining eventual earners of trophy i,
//     grows by phi_i of every new tracked player and converts into earners at rate a_i, where
//     phi_i is the share of players who will ever earn trophy i and a_i reflects how early and
//     how easy it is. Earners never decrease: dilution lowers the fraction, never the count.
//     In regime A both are fitted per trophy to the refresh points the case actually shows;
//     in regime B they come from tier/description priors.
// The median is snapped to the count grid (a whole number of earners at the projected pool,
// printed to 2 decimals like the page), and every quantile is snapped the same way. The spread
// lives on the logit scale, estimated from the case's own step volatility where the case moves,
// and from prior parameter uncertainty where it does not. Everything is a deterministic function
// of the case file: no clock reads, no randomness, so two runs write identical bytes.

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
// trophy typing: priors for the frozen-page regime, ceilings for the fitted one

type TrophyKind = 'platinum' | 'story' | 'mission' | 'collect' | 'grind' | 'gold' | 'silver' | 'misc';

interface Prior {
  phi: number; // share of the tracked pool that eventually earns it (week-one scale)
  a: number; // accrual speed, per hour
}

function trophyKind(t: TrophyMeta): { kind: TrophyKind; chapter: number } {
  if (t.tier === 'platinum') return { kind: 'platinum', chapter: 0 };
  const d = t.description.toLowerCase();
  const ch = /complete chapter\s+(\d+)/.exec(d);
  if (ch) return { kind: 'story', chapter: parseInt(ch[1], 10) };
  if (/(collect|find all|photograph|gather|discover all)/.test(d)) return { kind: 'collect', chapter: 0 };
  if (/(earn \$|complete all|win every|own every|purchase all)/.test(d)) return { kind: 'grind', chapter: 0 };
  if (d.startsWith('complete ')) return { kind: 'mission', chapter: 0 };
  if (t.tier === 'gold') return { kind: 'gold', chapter: 0 };
  if (t.tier === 'silver') return { kind: 'silver', chapter: 0 };
  return { kind: 'misc', chapter: 0 };
}

function priorFor(t: TrophyMeta): Prior {
  const { kind, chapter } = trophyKind(t);
  switch (kind) {
    case 'platinum':
      return { phi: 0.16, a: 0.022 };
    case 'story': {
      const n = Math.max(1, chapter);
      return { phi: clamp(0.96 - 0.035 * n, 0.66, 0.93), a: clamp(0.32 - 0.02 * n, 0.13, 0.3) };
    }
    case 'mission':
      return { phi: 0.88, a: 0.24 };
    case 'collect':
      return { phi: 0.4, a: 0.055 };
    case 'grind':
      return { phi: 0.32, a: 0.045 };
    case 'gold':
      return { phi: 0.3, a: 0.05 };
    case 'silver':
      return { phi: 0.45, a: 0.07 };
    default:
      return { phi: 0.58, a: 0.1 };
  }
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
}

/**
 * Fit a growth curve to the pool's refresh points (t in hours since the first segment, P).
 * Saturating candidates — logistic curves whose rate comes from a regression on the
 * logit(P/K) scale for each ceiling K on a grid, and exponential approaches to a ceiling —
 * compete on squared error in log space against a plain exponential, which is used only when
 * it fits far better than every saturating curve. Among near-tied saturating fits the one
 * projecting the smallest pool at the horizon wins: the ceiling is barely identified before
 * the bend, and over-projecting the pool is the catastrophic direction. The projection is
 * floored at the last observed pool.
 */
function fitPoolCurve(ts: number[], ps: number[], horizonH: number): PoolCurve {
  const n = ts.length;
  const lastP = ps[n - 1];
  if (n === 1) return { at: () => lastP };
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
  // Parsimony toward saturation: the ceiling is barely identified before the bend, and the
  // fits that bend soonest and latest look alike inside the window while diverging wildly
  // outside it. Among saturating fits within 30% of the best SSE, take the one projecting
  // the SMALLEST pool at the horizon — over-projecting the pool is the catastrophic error.
  let chosen: ((t: number) => number) | null = null;
  if (satCands.length > 0) {
    const bestSatSse = Math.min(...satCands.map((cand) => cand.sse));
    if (bestSatSse <= expSse / 0.6) {
      let bestAtH = Infinity;
      for (const cand of satCands) {
        if (cand.sse > bestSatSse * 1.3) continue;
        const v = cand.at(Math.max(horizonH, ts[n - 1]));
        if (v < bestAtH) {
          bestAtH = v;
          chosen = cand.at;
        }
      }
    }
  }
  if (chosen === null) chosen = expF;
  return {
    at: (t: number) => {
      const v = chosen(Math.max(0, t));
      return Number.isFinite(v) ? clamp(v, lastP, Math.max(lastP, Math.min(2_000_000, lastP * 12))) : lastP;
    },
  };
}

/** Launch-window pool prior for a page that has never moved: logistic from the frozen pool. */
function launchPool(P0: number, releaseMs: number): (tMs: number) => number {
  const K = 90_000;
  const r = Math.LN2 / 1.7; // early doubling about every 1.7 hours
  return (tMs: number) => {
    if (!Number.isFinite(releaseMs) || tMs <= releaseMs) return P0;
    const hours = (tMs - releaseMs) / HOUR_MS;
    const v = K / (1 + ((K - P0) / Math.max(1, P0)) * Math.exp(-r * hours));
    return clamp(v, P0, K);
  };
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
  for (const f of [1, 1.12, 1.28, 1.5, 1.8, 2.2, 2.7, 3.3, 4.2]) {
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
      acc += Math.abs(logit(clamp((100 * fe) / pts[k].pool, 0.02, 99.95)) - logit(clamp(100 * pts[k].frac, 0.02, 99.95)));
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
    if (pts.length === 1) {
      logitFMed = logit(100 * pts[0].frac);
    } else if (pts.length >= 2) {
      const m = pts.length;
      const logitF = pts.map((p) => logit(clamp(100 * p.frac, 0.02, 99.95)));
      const weights = pts.map((_, k) => 1 + (m > 1 ? k / (m - 1) : 0));

      // Candidate 1: logit(fraction) linear in log-time, recency-weighted.
      const fitLT = (sub: RefreshPoint[], subW: number[], subY: number[]) => {
        const [a, b0] = wls(sub.map((p) => p.x), subY, subW);
        const b = clamp(b0, -0.02, 1.5);
        return (x: number) => a + b * x;
      };
      const projLT = fitLT(pts, weights, logitF)(xEff);

      // Candidate 2: the two-state accrual, projected over the fitted pool.
      const odeFit = fitAccrualEarners(pts, pool, tEffH, c.trophies[i].tier === 'platinum' ? 4.2 : 3);
      const projODE = logit(clamp((100 * odeFit.e) / poolEff, 0.02, 99.95));

      // Candidate 3: the fraction persists.
      const projPers = logitF[m - 1];

      // Candidate 4: the fraction approaches a ceiling phi as a power law in log-time,
      // F = phi - C * exp(-b * x). This is the shape earners-versus-pool takes once pool
      // growth bends: the climb decelerates but does not stop at the last observed slope.
      const fitPow = (sub: RefreshPoint[], subY: number[]) => {
        const fLastSub = sub[sub.length - 1].frac;
        let best: { phi: number; b: number; C: number; sse: number } | null = null;
        for (const mult of [1.05, 1.2, 1.4, 1.7, 2.1, 2.6, 3.2, 4]) {
          const phi = Math.min(0.995, fLastSub * mult);
          if (phi <= fLastSub) continue;
          for (const b of [0.3, 0.5, 0.8, 1.2, 1.8, 2.6]) {
            let num = 0;
            let den = 0;
            sub.forEach((p) => {
              const u = Math.exp(-b * p.x);
              num += u * (phi - p.frac);
              den += u * u;
            });
            const C = den > 0 ? num / den : 0;
            if (C <= 0) continue;
            let sse = 0;
            sub.forEach((p, k) => {
              const fPred = clamp(phi - C * Math.exp(-b * p.x), 0.0002, 0.9995);
              sse += (logit(100 * fPred) - subY[k]) ** 2;
            });
            if (best === null || sse < best.sse) best = { phi, b, C, sse };
          }
        }
        if (best === null) return null;
        const bb = best;
        return (x: number) => logit(clamp(100 * (bb.phi - bb.C * Math.exp(-bb.b * x)), 0.02, 99.95));
      };
      const powFn = fitPow(pts, logitF);
      const projPow = powFn === null ? projLT : powFn(xEff);

      // Weight the candidates by evidence: a lead-weighted backtest on the held-out last
      // points for the reduced-form curves, shape fidelity for the accrual model, all
      // blended with a structural prior (set where the backtest block below runs).
      let wLT = 0.3;
      let wODE = 0.3;
      let wPers = 0.1;
      let wPow = 0.3;
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
        const errs = [0, 0, 0, 0];
        let wSum = 0;
        for (let k = cut; k < m; k++) {
          const lead = k - cut + 1;
          wSum += lead;
          const yTrue = logitF[k];
          errs[0] += lead * Math.abs(ltFn(pts[k].x) - yTrue);
          errs[2] += lead * Math.abs(trainY[cut - 1] - yTrue);
          errs[3] += lead * Math.abs((powTrainFn === null ? ltFn(pts[k].x) : powTrainFn(pts[k].x)) - yTrue);
        }
        const eLT = errs[0] / wSum;
        // The accrual model is NOT refit on the short train window for its score — refits on
        // 3-4 points are unstable along the (phi, a) ridge. It is scored by the full fit's
        // shape fidelity instead: the one thing a short backtest cannot measure is exactly
        // this model's edge, the pool-bend mechanics that only bite at long leads.
        const eODE = odeFit.fitErr;
        const ePers = errs[2] / wSum;
        const ePow = errs[3] / wSum;
        backtestErr = Math.min(eLT, eODE, ePers, ePow);
        const sLT = Math.exp(-eLT / 0.2);
        const sODE = Math.exp(-eODE / 0.2);
        const sPers = 0.6 * Math.exp(-ePers / 0.2);
        const sPow = Math.exp(-ePow / 0.2);
        const tot = sLT + sODE + sPers + sPow;
        // Blend the evidence weights with a structural prior: at multi-day horizons the
        // earners-vs-pool model and the ceiling curve are credible in a way a flat or
        // purely log-time extrapolation is not.
        wLT = 0.5 * (sLT / tot) + 0.5 * 0.15;
        wODE = 0.5 * (sODE / tot) + 0.5 * 0.5;
        wPers = 0.5 * (sPers / tot) + 0.5 * 0.1;
        wPow = 0.5 * (sPow / tot) + 0.5 * 0.25;
      }
      logitFMed = wLT * projLT + wODE * projODE + wPers * projPers + wPow * projPow;
    }
    if (Number.isFinite(logitFMed)) {
      fracMed[i] = invLogit(logitFMed) / 100;
    }

    const vol = volOf[i] === null ? globalVol : Math.sqrt(0.5 * (volOf[i] as number) ** 2 + 0.5 * globalVol ** 2);
    // The 1.8 factor is a calibration constant: raw step volatility understates forecast
    // spread because the candidates' model error, not just step noise, drives the misses.
    let sigma = 1.8 * Math.sqrt((0.6 * vol) ** 2 * c.horizonDays + (0.16 * Math.sqrt(c.horizonDays)) ** 2 + (0.5 * backtestErr) ** 2);
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

function forecastFrozen(c: CaseData, seg: Segment): TrophyForecast[] {
  const P0 = seg.players;
  const freezeMs = seg.t;
  const poolAt = launchPool(P0, c.releaseMs);
  const tEffMs = Math.max(freezeMs + 0.25 * HOUR_MS, c.targetMs - 0.5 * HOUR_MS);
  const poolInt = Math.max(1, Math.round(poolAt(tEffMs)));

  const stepH = 0.25;
  const steps = Math.max(1, Math.round((tEffMs - freezeMs) / HOUR_MS / stepH));

  return c.trophies.map((t, i) => {
    const prior = priorFor(t);
    const r0 = seg.rates[i];
    let e = r0 === null ? NaN : Math.round((clamp(r0, 0, 100) / 100) * P0);
    if (!Number.isFinite(e)) {
      // Never printed: start from a small tier-typical count and let the prior run.
      e = Math.max(0, Math.round(prior.phi * 0.05 * P0));
    }
    // Two-state accrual: R = remaining eventual earners in the pool. Arrivals add phi of each
    // new player to R; R converts into earners at rate a. Earners never decrease.
    let cur = e;
    let rest = Math.max(0, prior.phi * P0 - e);
    let pPrev = poolAt(freezeMs);
    for (let s = 1; s <= steps; s++) {
      const tMs = freezeMs + s * stepH * HOUR_MS;
      const p = poolAt(tMs);
      let conv = rest * (1 - Math.exp(-prior.a * stepH * 0.5));
      cur += conv;
      rest -= conv;
      rest += prior.phi * Math.max(0, p - pPrev);
      conv = rest * (1 - Math.exp(-prior.a * stepH * 0.5));
      cur += conv;
      rest -= conv;
      pPrev = p;
      if (cur > p) cur = p;
    }
    const eInt = Math.round(clamp(cur, 0, poolInt));
    let sigma = 0.45 + 0.08 * Math.sqrt(c.horizonDays);
    if (t.tier === 'platinum') sigma *= 1.2;
    if (r0 !== null && r0 >= 99.99) sigma *= 1.15;
    if (r0 !== null && r0 < 0.5) sigma *= 1.2;
    return {
      median: round2(clampPct((100 * eInt) / poolInt)),
      sigma: Math.min(0.75, sigma),
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

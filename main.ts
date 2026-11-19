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
//   - earners_i(t): a saturating accrual, dE/dt = a * (phi * players(t) - E), whose ceiling phi is
//     the share of the tracked pool that will ever earn trophy i and whose speed a reflects how
//     early and how easy the trophy is. In regime A both are fitted per trophy to the refresh
//     points the case actually shows; in regime B they come from tier/description priors.
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
 * Fit a saturating pool curve to refresh points (t in hours since the first segment, P).
 * Candidates: log-linear (pure exponential), log-in-log-time, and exponential approach to a
 * ceiling K on a grid. The winner is picked by squared error in log space, with a small
 * preference for saturating shapes, and the projection is floored at the last observed pool.
 */
function fitPoolCurve(ts: number[], ps: number[]): PoolCurve {
  const lastP = ps[ps.length - 1];
  const lnP = ps.map((p) => Math.log(Math.max(1, p)));
  const candidates: Array<{ sse: number; order: number; at: (t: number) => number }> = [];

  { // exponential approach to a ceiling K
    let best: { sse: number; at: (t: number) => number } | null = null;
    for (const mult of [1.3, 1.8, 2.6, 4, 6, 9, 14, 22, 34, 52]) {
      const K = lastP * mult;
      if (K <= Math.max(...ps) * 1.02) continue;
      const ys = ps.map((p) => Math.log(K - p));
      const [a, b] = ols(ts, ys);
      if (b >= -0.0005) continue; // K - P must shrink
      let sse = 0;
      for (let i = 0; i < ts.length; i++) {
        const pred = Math.log(Math.max(1, K - Math.exp(a + b * ts[i])));
        sse += (pred - lnP[i]) ** 2;
      }
      if (best === null || sse < best.sse) {
        best = { sse, at: (t: number) => K - Math.exp(a + b * t) };
      }
    }
    if (best !== null) candidates.push({ sse: best.sse, order: 0, at: best.at });
  }
  { // log in log-time
    const xs = ts.map((t) => Math.log(1 + t / 6));
    const [a, b0] = ols(xs, lnP);
    const b = Math.max(0, b0);
    let sse = 0;
    for (let i = 0; i < ts.length; i++) sse += (a + b * xs[i] - lnP[i]) ** 2;
    candidates.push({ sse, order: 1, at: (t: number) => Math.exp(a + b * Math.log(1 + t / 6)) });
  }
  { // pure exponential
    const [a, b0] = ols(ts, lnP);
    const b = Math.max(0, b0);
    let sse = 0;
    for (let i = 0; i < ts.length; i++) sse += (a + b * ts[i] - lnP[i]) ** 2;
    candidates.push({ sse, order: 2, at: (t: number) => Math.exp(a + b * t) });
  }

  let winner = candidates[0];
  for (const cand of candidates) {
    if (cand.sse * (1 + 0.02 * cand.order) < winner.sse * (1 + 0.02 * winner.order)) winner = cand;
  }
  return {
    at: (t: number) => {
      const v = winner.at(Math.max(0, t));
      return Number.isFinite(v) ? clamp(v, lastP, 2_000_000) : lastP;
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

function forecastMoving(c: CaseData, segs: Segment[]): TrophyForecast[] {
  const t0 = segs[0].t;
  const segH = segs.map((s) => (s.t - t0) / HOUR_MS);
  const pool = fitPoolCurve(segH, segs.map((s) => s.players));

  // The answer prints the state at the last refresh at or before the target: on average half a
  // refresh interval stale. Estimate the cadence from the segment starts and aim inside it.
  const gaps: number[] = [];
  for (let k = 1; k < segs.length; k++) gaps.push((segs[k].t - segs[k - 1].t) / HOUR_MS);
  gaps.sort((x, y) => x - y);
  const cadenceH = gaps.length ? gaps[Math.floor(gaps.length / 2)] : 0;
  const lagH = Math.min(cadenceH / 2, 3);
  const tEffH = Math.max(0, (c.targetMs - t0) / HOUR_MS - lagH);
  const poolEff = pool.at(tEffH);
  const poolInt = Math.max(1, Math.round(poolEff));

  // Per-trophy step volatility on the logit scale, for the spread.
  const n = c.trophies.length;
  const vols: number[] = [];
  const volOf: Array<number | null> = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    let cnt = 0;
    for (let k = 1; k < segs.length; k++) {
      const r0 = segs[k - 1].rates[i];
      const r1 = segs[k].rates[i];
      if (r0 === null || r1 === null) continue;
      const dd = Math.max(0.02, (segs[k].t - segs[k - 1].t) / DAY_MS);
      acc += ((logit(r1) - logit(r0)) ** 2) / dd;
      cnt++;
    }
    if (cnt > 0) {
      volOf[i] = Math.sqrt(acc / cnt);
      vols.push(volOf[i] as number);
    }
  }
  vols.sort((x, y) => x - y);
  const globalVol = vols.length ? vols[Math.floor(vols.length / 2)] : 0.55;

  const staleH = (c.checkpointMs - segs[segs.length - 1].t) / HOUR_MS;
  const staleFactor = staleH > 0.25 * (c.horizonDays * 24) ? 1.15 : 1;

  const earnersMed: number[] = new Array(n).fill(NaN);
  const out: TrophyForecast[] = new Array(n);

  for (let i = 0; i < n; i++) {
    // Refresh observations of this trophy's earners: whole counts at each printed state.
    const obsT: number[] = [];
    const obsE: number[] = [];
    for (let k = 0; k < segs.length; k++) {
      const r = segs[k].rates[i];
      if (r === null) continue;
      obsT.push(segH[k]);
      obsE.push(Math.round((clamp(r, 0, 100) / 100) * segs[k].players));
    }

    let eMed = NaN;
    if (obsE.length >= 2) {
      const fLast = obsE[obsE.length - 1] / segs[segs.length - 1].players;
      // Candidate 1: the earned fraction persists; earners grow only with the pool.
      let bestSse = 0;
      for (let k = 1; k < obsE.length; k++) {
        const pred = fLast * pool.at(obsT[k]);
        bestSse += ((pred - obsE[k]) / Math.max(obsE[k], 3)) ** 2;
      }
      let bestE = fLast * poolEff;
      // Candidate 2: saturating accrual toward a ceiling fraction phi at speed a.
      const phiSet = new Set<number>();
      for (const f of [1, 1.12, 1.28, 1.5, 1.8, 2.2, 2.7, 3.3, 4.2]) {
        phiSet.add(clamp(fLast * f, 0, 0.999));
      }
      phiSet.add(clamp(fLast + 0.05, 0, 0.999));
      for (const phi of phiSet) {
        for (const a of A_GRID) {
          let e = obsE[0];
          let sse = 0;
          for (let k = 1; k < obsE.length; k++) {
            const dt = obsT[k] - obsT[k - 1];
            e = phi * pool.at(obsT[k]) - (phi * pool.at(obsT[k - 1]) - e) * Math.exp(-a * dt);
            sse += ((e - obsE[k]) / Math.max(obsE[k], 3)) ** 2;
          }
          if (sse < bestSse) {
            bestSse = sse;
            let ef = obsE[0];
            for (let k = 1; k < obsE.length; k++) {
              const dt = obsT[k] - obsT[k - 1];
              ef = phi * pool.at(obsT[k]) - (phi * pool.at(obsT[k - 1]) - ef) * Math.exp(-a * dt);
            }
            const dtF = tEffH - obsT[obsT.length - 1];
            bestE = phi * poolEff - (phi * pool.at(obsT[obsT.length - 1]) - ef) * Math.exp(-a * dtF);
          }
        }
      }
      eMed = bestE;
    } else if (obsE.length === 1) {
      const segIdx = segs.findIndex((s) => s.rates[i] !== null);
      const f = obsE[0] / segs[Math.max(0, segIdx)].players;
      eMed = f * poolEff;
    }
    earnersMed[i] = eMed;

    const vol = volOf[i] === null ? globalVol : Math.sqrt(0.5 * (volOf[i] as number) ** 2 + 0.5 * globalVol ** 2);
    let sigma = Math.sqrt(vol * vol * c.horizonDays + (0.16 * Math.sqrt(c.horizonDays)) ** 2);
    const lr = lastRate(c, i);
    if (lr !== null && lr < 0.5) sigma *= 1.25;
    if (c.trophies[i].tier === 'platinum') sigma *= 1.2;
    sigma *= staleFactor;
    out[i] = { median: NaN, sigma: Math.max(0.1, sigma), poolInt };
  }

  // The platinum is gated by everything else: its earners cannot exceed the rarest other
  // trophy's earners. Cap it just under that minimum.
  const platIdx = c.trophies.findIndex((t) => t.tier === 'platinum');
  if (platIdx >= 0 && Number.isFinite(earnersMed[platIdx])) {
    let minOther = Infinity;
    for (let i = 0; i < n; i++) {
      if (i !== platIdx && Number.isFinite(earnersMed[i])) minOther = Math.min(minOther, earnersMed[i]);
    }
    if (Number.isFinite(minOther)) earnersMed[platIdx] = Math.min(earnersMed[platIdx], 0.98 * minOther);
  }

  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(earnersMed[i])) continue;
    const eInt = Math.round(clamp(earnersMed[i], 0, poolInt));
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
    let cur = e;
    for (let s = 1; s <= steps; s++) {
      const tMs = freezeMs + s * stepH * HOUR_MS;
      const p = poolAt(tMs);
      cur += prior.a * (prior.phi * p - cur) * stepH;
      if (cur < 0) cur = 0;
      if (cur > p) cur = p;
    }
    const eInt = Math.round(clamp(cur, 0, poolInt));
    let sigma = 0.3 + 0.1 * Math.sqrt(c.horizonDays);
    if (t.tier === 'platinum') sigma *= 1.2;
    if (r0 !== null && r0 >= 99.99) sigma *= 1.15;
    if (r0 !== null && r0 < 0.5) sigma *= 1.2;
    return {
      median: round2(clampPct((100 * eInt) / poolInt)),
      sigma: Math.min(0.6, sigma),
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

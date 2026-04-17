// Default air-writing template bank.
// Each template is a normalised 32-point [x, y] trajectory (coords in [0, 1]).
// Matched against live trajectories via DTW in AirWriter.recognise.
// To add a new template: pick a distinctive *single-stroke* shape,
// sample ~32 evenly-spaced points from stroke start → end, normalise
// x/y into [0, 1], and add an entry to DEFAULT_AIR_TEMPLATES.
//
// DTW quality tips:
// - Stick to single-stroke shapes. Multi-stroke shapes (like an X) look
//   like a teleport to DTW and will mis-match.
// - Shapes should be distinctive in direction and extent — a small
//   check-mark and a big slash look similar after normalisation.

function linear(from: [number, number], to: [number, number], n: number): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    out.push([from[0] + t * (to[0] - from[0]), from[1] + t * (to[1] - from[1])]);
  }
  return out;
}

function concat(...segs: [number, number][][]): [number, number][] {
  const out: [number, number][] = [];
  for (const s of segs) out.push(...s);
  return resample(out, 32);
}

function resample(pts: [number, number][], n: number): [number, number][] {
  if (pts.length < 2) return pts;
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * (pts.length - 1);
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, pts.length - 1);
    const frac = t - lo;
    out.push([
      pts[lo][0] + frac * (pts[hi][0] - pts[lo][0]),
      pts[lo][1] + frac * (pts[hi][1] - pts[lo][1]),
    ]);
  }
  return out;
}

// check-mark: short down-right, then long up-right → affirmation
const YES: [number, number][] = concat(
  linear([0.0, 0.5], [0.35, 1.0], 12),
  linear([0.35, 1.0], [1.0, 0.0], 20)
);

// question-mark: curve over the top, then down to the dot → clarifying
const QUESTION: [number, number][] = concat(
  linear([0.1, 0.25], [0.5, 0.0], 8),
  linear([0.5, 0.0], [0.9, 0.25], 8),
  linear([0.9, 0.25], [0.5, 0.55], 8),
  linear([0.5, 0.55], [0.5, 1.0], 8)
);

// zig-zag wave across the top → greeting
const HI: [number, number][] = concat(
  linear([0.0, 0.0], [0.25, 1.0], 8),
  linear([0.25, 1.0], [0.5, 0.0], 8),
  linear([0.5, 0.0], [0.75, 1.0], 8),
  linear([0.75, 1.0], [1.0, 0.0], 8)
);

// straight vertical line bottom→top → "help" (raise hand / SOS mental model)
const HELP: [number, number][] = linear([0.5, 1.0], [0.5, 0.0], 32);

// horizontal line left→right → "done" (close / finish)
const DONE: [number, number][] = linear([0.0, 0.5], [1.0, 0.5], 32);

// plus-sign-ish as a single stroke: long down, backtrack up, then across → "more"
// mimics drawing "+"  as one continuous stroke (down, back, right)
const MORE: [number, number][] = concat(
  linear([0.5, 0.0], [0.5, 1.0], 12),
  linear([0.5, 1.0], [0.5, 0.5], 6),
  linear([0.5, 0.5], [1.0, 0.5], 14)
);

// single wave (down-up-down-up smooth) → "water" (fluid/ocean mental model)
const WATER: [number, number][] = concat(
  linear([0.0, 0.5], [0.2, 0.9], 6),
  linear([0.2, 0.9], [0.4, 0.1], 8),
  linear([0.4, 0.1], [0.6, 0.9], 8),
  linear([0.6, 0.9], [0.8, 0.1], 8),
  linear([0.8, 0.1], [1.0, 0.5], 2)
);

// square/box (traced as one stroke) → "stop"
// start top-left, go right, down, left, up — closing the box
const STOP: [number, number][] = concat(
  linear([0.0, 0.0], [1.0, 0.0], 8),
  linear([1.0, 0.0], [1.0, 1.0], 8),
  linear([1.0, 1.0], [0.0, 1.0], 8),
  linear([0.0, 1.0], [0.0, 0.0], 8)
);

export const DEFAULT_AIR_TEMPLATES: Map<string, [number, number][]> = new Map([
  ["yes", YES],
  ["?", QUESTION],
  ["hi", HI],
  ["help", HELP],
  ["done", DONE],
  ["more", MORE],
  ["water", WATER],
  ["stop", STOP],
]);

import type { Affect, GestureName, MemoryBucket } from "../types";

// ── Affect classification (ported from backend/sensing/face_mesh.py) ────────

interface AffectVector {
  MAR: number;
  EAR: number;
  BRI: number;
  LCP: number;
}

export function classifyAffect(v: AffectVector): Affect {
  // BRI is relative (browMid.y - eyeCenter.y) / interOcular — more negative = brows raised higher
  // LCP is vertical offset of lip corners from mouth center, normalised by inter-ocular,
  //   relative to calibrated neutral — positive = corners pulled UP (smile), negative = DOWN (frown)
  // MAR is absolute ratio — higher = mouth more open
  // EAR is absolute ratio — lower = eyes more closed / squinting
  if (v.BRI < -0.35 && v.MAR > 0.4) return "SURPRISED";
  // FRUSTRATED: a clear frown, OR brows lowered + squinting — either signals displeasure
  if (v.LCP < -0.018) return "FRUSTRATED";
  if (v.BRI > -0.2 && v.EAR < 0.18) return "FRUSTRATED";
  if (v.LCP > 0.012) return "HAPPY";
  return "NEUTRAL";
}

// Face landmark indices (MediaPipe 478-point mesh)
const MOUTH_TOP = 13, MOUTH_BOTTOM = 14, MOUTH_LEFT = 61, MOUTH_RIGHT = 291;
const EYE_TOP = 159, EYE_BOTTOM = 145, EYE_LEFT = 33, EYE_RIGHT = 133;
const BROW_LEFT = 70, BROW_RIGHT = 300;
const CORNER_LEFT = 61, CORNER_RIGHT = 291;

function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

export function computeAffectVector(
  landmarks: { x: number; y: number }[],
  neutralLCP: number
): AffectVector {
  const MAR =
    dist(landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM]) /
    (dist(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT]) + 1e-6);

  const EAR =
    dist(landmarks[EYE_TOP], landmarks[EYE_BOTTOM]) /
    (dist(landmarks[EYE_LEFT], landmarks[EYE_RIGHT]) + 1e-6);

  const eyeCenter = {
    x: (landmarks[EYE_LEFT].x + landmarks[EYE_RIGHT].x) / 2,
    y: (landmarks[EYE_LEFT].y + landmarks[EYE_RIGHT].y) / 2,
  };
  const interOcular = dist(landmarks[EYE_LEFT], landmarks[EYE_RIGHT]);
  const browMid = {
    x: (landmarks[BROW_LEFT].x + landmarks[BROW_RIGHT].x) / 2,
    y: (landmarks[BROW_LEFT].y + landmarks[BROW_RIGHT].y) / 2,
  };
  // MediaPipe y increases downward, so browMid.y < eyeCenter.y when brows are above eyes.
  // Raising brows moves them toward y=0, making this value more negative.
  const BRI = (browMid.y - eyeCenter.y) / (interOcular + 1e-6);

  // Lip-corner pull: average y of the two corners vs. mouth vertical centre,
  // normalised by inter-ocular distance, relative to calibrated neutral.
  // MediaPipe y increases downward, so corners rising above the mouth centre → negative raw,
  // which we flip so smile = positive. Subtracting the calibrated neutral removes per-face bias.
  const mouthCentreY = (landmarks[MOUTH_TOP].y + landmarks[MOUTH_BOTTOM].y) / 2;
  const cornerAvgY = (landmarks[CORNER_LEFT].y + landmarks[CORNER_RIGHT].y) / 2;
  const rawLCP = (mouthCentreY - cornerAvgY) / (interOcular + 1e-6);
  const LCP = rawLCP - neutralLCP;

  return { MAR, EAR, BRI, LCP };
}

// ── Gesture classification (ported from backend/sensing/gesture.py) ─────────

interface Point3D {
  x: number;
  y: number;
  z: number;
}

function norm3(a: Point3D): number {
  return Math.sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2);
}

function sub3(a: Point3D, b: Point3D): Point3D {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function scale3(a: Point3D, s: number): Point3D {
  return { x: a.x * s, y: a.y * s, z: a.z * s };
}

export function classifyGesture(landmarks: Point3D[]): GestureName | null {
  const wrist = landmarks[0];
  const palmWidth =
    norm3(sub3(landmarks[5], landmarks[17])) + 1e-6;

  const p = landmarks.map((lm) => scale3(sub3(lm, wrist), 1 / palmWidth));

  const thumbTip = p[4];
  const indexTip = p[8];
  const middleTip = p[12];
  const ringTip = p[16];
  const pinkyTip = p[20];
  const indexMcp = p[5];

  const fingersCurled = [
    [indexTip, p[5]],
    [middleTip, p[9]],
    [ringTip, p[13]],
  ].every(([tip, mcp]) => norm3(tip) < norm3(mcp));

  // Check POINTING before THUMBS_UP — pointing with a raised thumb would otherwise
  // satisfy fingersCurled on a noisy frame and fire the wrong label first.
  const indexExtended = norm3(indexTip) > norm3(indexMcp) * 1.3;
  const othersCurled = [middleTip, ringTip, pinkyTip].every(
    (tip) => norm3(tip) < 0.7
  );
  if (indexExtended && othersCurled) return "POINTING";

  if (thumbTip.y < -0.3 && fingersCurled) return "THUMBS_UP";
  if (thumbTip.y > 0.3 && fingersCurled) return "THUMBS_DOWN";

  const allExtended = [indexTip, middleTip, ringTip, pinkyTip, thumbTip].every(
    (tip) => norm3(tip) > 0.7
  );
  if (allExtended) return "WAVING";

  return null;
}

// ── Gaze region mapping (ported from backend/sensing/gaze.py) ────────────────

const LEFT_IRIS_CENTER = 468;
const RIGHT_IRIS_CENTER = 473;

interface GazeRegion {
  bounds: [number, number, number, number]; // x_min, y_min, x_max, y_max
  bucket: MemoryBucket;
}

const GAZE_REGIONS: GazeRegion[] = [
  // Centre checked first (most specific region)
  { bounds: [0.3, 0.3, 0.7, 0.7], bucket: "social" },
  { bounds: [0.0, 0.0, 0.5, 0.5], bucket: "family" },
  { bounds: [0.5, 0.0, 1.0, 0.5], bucket: "medical" },
  { bounds: [0.0, 0.5, 0.5, 1.0], bucket: "hobbies" },
  { bounds: [0.5, 0.5, 1.0, 1.0], bucket: "daily_routine" },
];

function regionFor(x: number, y: number): MemoryBucket | null {
  for (const { bounds, bucket } of GAZE_REGIONS) {
    if (x >= bounds[0] && x <= bounds[2] && y >= bounds[1] && y <= bounds[3]) {
      return bucket;
    }
  }
  return null;
}

export class GazeTracker {
  private currentRegion: MemoryBucket | null = null;
  private dwellStart = 0;
  private dwellThresholdMs: number;

  constructor(dwellThresholdMs = 1500) {
    this.dwellThresholdMs = dwellThresholdMs;
  }

  process(landmarks: { x: number; y: number }[]): MemoryBucket | null {
    if (landmarks.length <= RIGHT_IRIS_CENTER) return null;

    const gazeX =
      (landmarks[LEFT_IRIS_CENTER].x + landmarks[RIGHT_IRIS_CENTER].x) / 2;
    const gazeY =
      (landmarks[LEFT_IRIS_CENTER].y + landmarks[RIGHT_IRIS_CENTER].y) / 2;

    const bucket = regionFor(gazeX, gazeY);

    if (bucket !== this.currentRegion) {
      this.currentRegion = bucket;
      this.dwellStart = performance.now();
      return null;
    }

    if (
      bucket !== null &&
      performance.now() - this.dwellStart >= this.dwellThresholdMs
    ) {
      this.currentRegion = null;
      this.dwellStart = 0;
      return bucket;
    }

    return null;
  }

  reset() {
    this.currentRegion = null;
    this.dwellStart = 0;
  }
}

// ── Head-pose tracker (shake / sharp-nod-dissatisfied) ──────────────────────

export type HeadSignal = "HEAD_SHAKE" | "HEAD_NOD_DISSATISFIED";

const NOSE_TIP = 1;

interface NosePoint {
  x: number;
  y: number;
  t: number;
}

export interface HeadDebug {
  dx: number;        // current x displacement from neutral
  dy: number;        // current y displacement from neutral
  maxAbsDx: number;  // peak |dx| within the window
  maxAbsDy: number;  // peak |dy| within the window
  crossings: number; // side crossings within the window (deadband-filtered)
}

export class HeadPoseTracker {
  private neutralX: number | null = null;
  private neutralY: number | null = null;
  private history: NosePoint[] = [];
  private lastEmitTs = 0;
  private lastDebug: HeadDebug = {
    dx: 0,
    dy: 0,
    maxAbsDx: 0,
    maxAbsDy: 0,
    crossings: 0,
  };

  private static WINDOW_MS = 1000;
  private static REFRACTORY_MS = 2000;
  private static SHAKE_AMPLITUDE = 0.015;
  private static SHAKE_MIN_CROSSINGS = 3;
  // Per-frame jitter below this magnitude is ignored when counting side
  // crossings, so micro-fidgets near neutral can't rack up false crossings.
  private static SHAKE_DEADBAND = 0.005;
  private static NOD_DROP = 0.06;
  private static NOD_WINDOW_MS = 600;
  // Reject "nod" when horizontal motion exceeds this — it's a shake/sway.
  private static NOD_MAX_HORIZONTAL = 0.015;
  // Recovery: head must come back to within this of neutral.
  private static NOD_RECOVERY = 0.015;
  // The drop must start from near-neutral (not from a tilted resting pose).
  private static NOD_START_THRESHOLD = 0.015;
  // Minimum frames between drop start and peak — guards against single-frame
  // landmark glitches that look like an instantaneous jerk.
  private static NOD_MIN_DROP_FRAMES = 3;
  // Minimum frames between peak and recovery — same reason, going up.
  private static NOD_MIN_RECOVERY_FRAMES = 2;

  calibrate(landmarks: { x: number; y: number }[]): void {
    if (!landmarks[NOSE_TIP]) return;
    this.neutralX = landmarks[NOSE_TIP].x;
    this.neutralY = landmarks[NOSE_TIP].y;
    this.history = [];
    this.lastEmitTs = 0;
  }

  process(landmarks: { x: number; y: number }[]): HeadSignal | null {
    if (!landmarks[NOSE_TIP]) return null;
    if (this.neutralX === null || this.neutralY === null) return null;

    const now = performance.now();
    const nose = landmarks[NOSE_TIP];
    this.history.push({ x: nose.x, y: nose.y, t: now });
    const cutoff = now - HeadPoseTracker.WINDOW_MS;
    this.history = this.history.filter((p) => p.t >= cutoff);

    this.updateDebug(nose);

    if (now - this.lastEmitTs < HeadPoseTracker.REFRACTORY_MS) return null;
    if (this.history.length < 6) return null;

    const shake = this.detectShake();
    if (shake) {
      this.lastEmitTs = now;
      return shake;
    }

    const nod = this.detectNod(now);
    if (nod) {
      this.lastEmitTs = now;
      return nod;
    }

    return null;
  }

  private updateDebug(nose: { x: number; y: number }): void {
    if (this.neutralX === null || this.neutralY === null) return;
    let maxAbsDx = 0;
    let maxAbsDy = 0;
    let crossings = 0;
    let prevSide = 0;
    for (const p of this.history) {
      const dx = p.x - this.neutralX;
      const dy = p.y - this.neutralY;
      const absDx = Math.abs(dx);
      maxAbsDx = Math.max(maxAbsDx, absDx);
      maxAbsDy = Math.max(maxAbsDy, Math.abs(dy));
      if (absDx < HeadPoseTracker.SHAKE_DEADBAND) continue;
      const side = dx > 0 ? 1 : -1;
      if (prevSide !== 0 && side !== prevSide) crossings += 1;
      prevSide = side;
    }
    this.lastDebug = {
      dx: nose.x - this.neutralX,
      dy: nose.y - this.neutralY,
      maxAbsDx,
      maxAbsDy,
      crossings,
    };
  }

  get debug(): HeadDebug {
    return this.lastDebug;
  }

  private detectShake(): HeadSignal | null {
    if (this.neutralX === null) return null;
    let crossings = 0;
    let prevSide = 0;
    let maxAbs = 0;
    for (const p of this.history) {
      const dx = p.x - this.neutralX;
      const absDx = Math.abs(dx);
      maxAbs = Math.max(maxAbs, absDx);
      // Only commit to a side once the displacement clears the deadband —
      // otherwise sub-millimeter jitter near neutral fakes crossings.
      if (absDx < HeadPoseTracker.SHAKE_DEADBAND) continue;
      const side = dx > 0 ? 1 : -1;
      if (prevSide !== 0 && side !== prevSide) crossings += 1;
      prevSide = side;
    }
    if (
      crossings >= HeadPoseTracker.SHAKE_MIN_CROSSINGS &&
      maxAbs >= HeadPoseTracker.SHAKE_AMPLITUDE
    ) {
      return "HEAD_SHAKE";
    }
    return null;
  }

  private detectNod(now: number): HeadSignal | null {
    if (this.neutralX === null || this.neutralY === null) return null;
    const windowStart = now - HeadPoseTracker.NOD_WINDOW_MS;
    const recent = this.history.filter((p) => p.t >= windowStart);
    if (recent.length < 6) return null;

    // Reject if there's significant horizontal motion — that's a shake/sway.
    let maxAbsDx = 0;
    for (const p of recent) {
      maxAbsDx = Math.max(maxAbsDx, Math.abs(p.x - this.neutralX));
    }
    if (maxAbsDx > HeadPoseTracker.NOD_MAX_HORIZONTAL) return null;

    // Find the peak (lowest head position) within the window.
    let maxDrop = 0;
    let peakIdx = -1;
    for (let i = 0; i < recent.length; i++) {
      const drop = recent[i].y - this.neutralY;
      if (drop > maxDrop) {
        maxDrop = drop;
        peakIdx = i;
      }
    }
    if (maxDrop < HeadPoseTracker.NOD_DROP || peakIdx < 0) return null;

    // Find a near-neutral start before the peak — a nod is a deliberate
    // motion *from* neutral, not a recovery from an already-tilted pose.
    let startIdx = -1;
    for (let i = peakIdx - 1; i >= 0; i--) {
      if (
        recent[i].y - this.neutralY <=
        HeadPoseTracker.NOD_START_THRESHOLD
      ) {
        startIdx = i;
        break;
      }
    }
    if (
      startIdx < 0 ||
      peakIdx - startIdx < HeadPoseTracker.NOD_MIN_DROP_FRAMES
    ) {
      return null;
    }

    // Recovery: head must return near neutral after the peak.
    let recoveryIdx = -1;
    for (let i = peakIdx + 1; i < recent.length; i++) {
      if (recent[i].y - this.neutralY < HeadPoseTracker.NOD_RECOVERY) {
        recoveryIdx = i;
        break;
      }
    }
    if (
      recoveryIdx < 0 ||
      recoveryIdx - peakIdx < HeadPoseTracker.NOD_MIN_RECOVERY_FRAMES
    ) {
      return null;
    }

    return "HEAD_NOD_DISSATISFIED";
  }

  reset(): void {
    this.neutralX = null;
    this.neutralY = null;
    this.history = [];
    this.lastEmitTs = 0;
  }

  get calibrated(): boolean {
    return this.neutralX !== null && this.neutralY !== null;
  }
}

// ── Air-writing DTW (ported from backend/sensing/air_writing.py) ─────────────

const INDEX_TIP = 8;
const VELOCITY_START = 15;
const VELOCITY_END = 5;
const STROKE_GAP_MS = 200;
const RESAMPLE_N = 32;

function normaliseTrajectory(pts: [number, number][]): [number, number][] {
  if (pts.length < 2) return pts;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of pts) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  const scaleX = maxX - minX + 1e-6;
  const scaleY = maxY - minY + 1e-6;
  const norm = pts.map(([x, y]) => [(x - minX) / scaleX, (y - minY) / scaleY] as [number, number]);

  // Resample to RESAMPLE_N points via linear interpolation
  const resampled: [number, number][] = [];
  for (let i = 0; i < RESAMPLE_N; i++) {
    const t = (i / (RESAMPLE_N - 1)) * (norm.length - 1);
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, norm.length - 1);
    const frac = t - lo;
    resampled.push([
      norm[lo][0] + frac * (norm[hi][0] - norm[lo][0]),
      norm[lo][1] + frac * (norm[hi][1] - norm[lo][1]),
    ]);
  }
  return resampled;
}

function dtwDistance(a: [number, number][], b: [number, number][]): number {
  const n = a.length, m = b.length;
  const dtw: number[][] = Array.from({ length: n + 1 }, () =>
    Array(m + 1).fill(Infinity)
  );
  dtw[0][0] = 0;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = Math.sqrt(
        (a[i - 1][0] - b[j - 1][0]) ** 2 + (a[i - 1][1] - b[j - 1][1]) ** 2
      );
      dtw[i][j] = cost + Math.min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1]);
    }
  }
  return dtw[n][m];
}

export class AirWriter {
  private trajectory: [number, number][] = [];
  private inStroke = false;
  private strokeEndTime = 0;
  private prevPt: [number, number] | null = null;
  private buffer: string[] = [];
  private templates: Map<string, [number, number][]>;

  constructor(templates: Map<string, [number, number][]> = new Map()) {
    this.templates = templates;
  }

  processHandLandmarks(
    landmarks: { x: number; y: number }[],
    frameWidth: number,
    frameHeight: number
  ): string | null {
    const tip: [number, number] = [
      landmarks[INDEX_TIP].x * frameWidth,
      landmarks[INDEX_TIP].y * frameHeight,
    ];

    let velocity = 0;
    if (this.prevPt) {
      velocity = Math.sqrt(
        (tip[0] - this.prevPt[0]) ** 2 + (tip[1] - this.prevPt[1]) ** 2
      );
    }
    this.prevPt = tip;

    if (velocity > VELOCITY_START) {
      this.inStroke = true;
      this.trajectory.push(tip);
      this.strokeEndTime = 0;
      return null;
    }

    if (this.inStroke && velocity < VELOCITY_END) {
      if (this.strokeEndTime === 0) {
        this.strokeEndTime = performance.now();
      }
      return this.checkStrokeEnd();
    }

    return null;
  }

  private checkStrokeEnd(): string | null {
    if (!this.inStroke || this.strokeEndTime === 0) return null;
    if (performance.now() - this.strokeEndTime >= STROKE_GAP_MS) {
      const char = this.recognise(this.trajectory);
      this.trajectory = [];
      this.inStroke = false;
      this.strokeEndTime = 0;
      if (char) this.buffer.push(char);
      return char;
    }
    return null;
  }

  private recognise(trajectory: [number, number][]): string | null {
    if (trajectory.length < 5) {
      return null;
    }
    if (this.templates.size === 0) {
      console.debug("[AirWriter] stroke completed but template bank is empty");
      return null;
    }
    const query = normaliseTrajectory(trajectory);
    let bestChar: string | null = null;
    let bestDist = Infinity;
    for (const [char, template] of this.templates) {
      const d = dtwDistance(query, template);
      if (d < bestDist) {
        bestDist = d;
        bestChar = char;
      }
    }
    // Reject poor matches so we don't pass garbage to the LLM.
    // Threshold is empirical — tune once real users test this.
    const MATCH_THRESHOLD = 8.0;
    if (bestDist > MATCH_THRESHOLD) {
      console.debug(
        `[AirWriter] no template matched (best='${bestChar}', dist=${bestDist.toFixed(2)})`
      );
      return null;
    }
    return bestChar;
  }

  getText(): string {
    const text = this.buffer.join("");
    this.buffer = [];
    return text;
  }

  noHand(): string | null {
    this.prevPt = null;
    return this.checkStrokeEnd();
  }
}

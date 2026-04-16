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
  // LCP is relative to calibrated neutral — positive = corners pulled up (smile)
  // MAR is absolute ratio — higher = mouth more open
  // EAR is absolute ratio — lower = eyes more closed
  if (v.BRI < -0.35 && v.MAR > 0.4) return "SURPRISED";
  if (v.EAR < 0.12 && v.LCP < -0.005) return "FRUSTRATED";
  if (v.LCP > 0.005) return "HAPPY";
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

  const LCP =
    (landmarks[CORNER_LEFT].x + landmarks[CORNER_RIGHT].x) / 2 - neutralLCP;

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

  if (thumbTip.y < -0.3 && fingersCurled) return "THUMBS_UP";
  if (thumbTip.y > 0.3 && fingersCurled) return "THUMBS_DOWN";

  const indexExtended = norm3(indexTip) > norm3(indexMcp) * 1.3;
  const othersCurled = [middleTip, ringTip, pinkyTip].every(
    (tip) => norm3(tip) < 0.5
  );
  if (indexExtended && othersCurled) return "POINTING";

  const allExtended = [indexTip, middleTip, ringTip, pinkyTip].every(
    (tip) => norm3(tip) > 0.5
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
    if (trajectory.length < 5 || this.templates.size === 0) return null;
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

import type { Matrix } from "@mediapipe/tasks-vision";
import type { Affect, GestureName, HeadDebug, HeadSignal, MemoryBucket } from "../types";

const SIGMA_K = 2.0;
const CALIBRATION_DURATION_MS = 5000;
const CALIBRATION_WARMUP_MS   = 1000;
const OUTLIER_TRIM_FRACTION   = 0.10;

const AFFECT_BLENDSHAPES = [
  "mouthSmileLeft", "mouthSmileRight",
  "browDownLeft", "browDownRight",
  "eyeSquintLeft", "eyeSquintRight",
  "jawOpen", "browInnerUp",
] as const;
type AffectBlendshape = typeof AFFECT_BLENDSHAPES[number];

interface Stats { mean: number; std: number }

interface Baseline {
  affect: Record<string, Stats>;
  gaze: { x: number; y: number };
  head: { pitch: number; yaw: number; roll: number };
  faceBboxSize: number;  // normalised face size — proxy for distance
}

function trimmedStats(values: number[]): Stats {
  if (values.length === 0) return { mean: 0, std: 0 };
  const sorted = [...values].sort((a, b) => a - b);
  const trim = Math.floor(sorted.length * OUTLIER_TRIM_FRACTION);
  const kept = sorted.slice(trim, sorted.length - trim);
  if (kept.length === 0) return { mean: 0, std: 0 };
  const mean = kept.reduce((s, v) => s + v, 0) / kept.length;
  const variance = kept.reduce((s, v) => s + (v - mean) ** 2, 0) / kept.length;
  const std = Math.max(Math.sqrt(variance), 0.01);
  return { mean, std };
}

function trimmedMean(values: number[]): number {
  return trimmedStats(values).mean;
}

export class Calibrator {
  private startTs = 0;
  private active = false;
  private done = false;

  private affectSamples: Record<string, number[]> = {};
  private gazeSamples: { x: number; y: number }[] = [];
  private headSamples: { pitch: number; yaw: number; roll: number }[] = [];
  private bboxSamples: number[] = [];

  private baseline: Baseline | null = null;

  start(): void {
    this.startTs = performance.now();
    this.active = true;
    this.done = false;
    this.baseline = null;
    this.affectSamples = {};
    for (const name of AFFECT_BLENDSHAPES) this.affectSamples[name] = [];
    this.gazeSamples = [];
    this.headSamples = [];
    this.bboxSamples = [];
  }

  cancel(): void {
    this.active = false;
    this.done = false;
    this.baseline = null;
  }

  get isActive(): boolean { return this.active; }
  get isReady(): boolean  { return this.done && this.baseline !== null; }

  // 0 → 1 over the calibration window (excluding warm-up).
  get progress(): number {
    if (!this.active) return this.done ? 1 : 0;
    const elapsed = performance.now() - this.startTs - CALIBRATION_WARMUP_MS;
    if (elapsed <= 0) return 0;
    return Math.min(1, elapsed / (CALIBRATION_DURATION_MS - CALIBRATION_WARMUP_MS));
  }

  // Feed a frame's signals during calibration. After the window elapses,
  // the baseline is computed and `isReady` becomes true.
  addSample(args: {
    blendshapes: Record<string, number>;
    gaze: { x: number; y: number } | null;
    head: { pitch: number; yaw: number; roll: number } | null;
    faceBboxSize: number | null;
  }): void {
    if (!this.active) return;
    const elapsed = performance.now() - this.startTs;

    if (elapsed < CALIBRATION_WARMUP_MS) return;

    if (elapsed >= CALIBRATION_DURATION_MS) {
      this.finalise();
      return;
    }

    for (const name of AFFECT_BLENDSHAPES) {
      const v = args.blendshapes[name];
      if (typeof v === "number") this.affectSamples[name].push(v);
    }
    if (args.gaze) this.gazeSamples.push(args.gaze);
    if (args.head) this.headSamples.push(args.head);
    if (typeof args.faceBboxSize === "number") this.bboxSamples.push(args.faceBboxSize);
  }

  private finalise(): void {
    const affect: Record<string, Stats> = {};
    for (const name of AFFECT_BLENDSHAPES) {
      affect[name] = trimmedStats(this.affectSamples[name] ?? []);
    }
    const gaze = {
      x: trimmedMean(this.gazeSamples.map((g) => g.x)),
      y: trimmedMean(this.gazeSamples.map((g) => g.y)),
    };
    const head = {
      pitch: trimmedMean(this.headSamples.map((h) => h.pitch)),
      yaw:   trimmedMean(this.headSamples.map((h) => h.yaw)),
      roll:  trimmedMean(this.headSamples.map((h) => h.roll)),
    };
    // Floor at a small positive value so we never divide by zero when scaling.
    const faceBboxSize = Math.max(trimmedMean(this.bboxSamples), 0.01);

    this.baseline = { affect, gaze, head, faceBboxSize };
    this.active = false;
    this.done = true;
  }

  getBaseline(): Baseline | null { return this.baseline; }
}

const AFFECT_FALLBACK_THRESHOLD = 0.4;

function isAbove(
  bs: Record<string, number>,
  name: AffectBlendshape,
  baseline: Baseline | null,
): boolean {
  const v = bs[name] ?? 0;
  if (baseline) {
    const stats = baseline.affect[name];
    if (!stats) return false;
    return v - stats.mean > SIGMA_K * stats.std;
  }
  return v > AFFECT_FALLBACK_THRESHOLD;
}

export function classifyAffect(
  bs: Record<string, number>,
  baseline: Baseline | null = null,
): Affect {
  const smileL  = isAbove(bs, "mouthSmileLeft",  baseline);
  const smileR  = isAbove(bs, "mouthSmileRight", baseline);
  const browDL  = isAbove(bs, "browDownLeft",    baseline);
  const browDR  = isAbove(bs, "browDownRight",   baseline);
  const squintL = isAbove(bs, "eyeSquintLeft",   baseline);
  const squintR = isAbove(bs, "eyeSquintRight",  baseline);
  const jawOpen = isAbove(bs, "jawOpen",         baseline);
  const browIn  = isAbove(bs, "browInnerUp",     baseline);

  if (jawOpen && browIn)  return "SURPRISED";
  if (browDL || browDR)   return "FRUSTRATED";
  if (squintL && squintR) return "FRUSTRATED";
  if (smileL && smileR)   return "HAPPY";
  return "NEUTRAL";
}

export function mapGestureLabel(label: string): GestureName | null {
  switch (label) {
    case "Thumb_Up":    return "THUMBS_UP";
    case "Thumb_Down":  return "THUMBS_DOWN";
    case "Pointing_Up": return "POINTING_UP";
    case "Closed_Fist": return "CLOSED_FIST";
    case "Open_Palm":   return "OPEN_PALM";
    case "Victory":     return "VICTORY";
    case "ILoveYou":    return "I_LOVE_YOU";
    default:            return null;
  }
}

// Bucket layout matches the 5 regions on the AAC interface:
//   family / medical (top), social (centre), hobbies / daily_routine (bottom).
const GAZE_INVERT_Y = import.meta.env.VITE_GAZE_INVERT_Y === "true";
const GAZE_LATERAL_DELTA  = 0.12;
const GAZE_VERTICAL_DELTA = 0.12;

export function worldGazeXY(
  matrix: Matrix,
  bs: Record<string, number>,
): { x: number; y: number } {
  const eyeR = ((bs.eyeLookInLeft  ?? 0) + (bs.eyeLookOutRight ?? 0)) / 2;
  const eyeL = ((bs.eyeLookOutLeft ?? 0) + (bs.eyeLookInRight  ?? 0)) / 2;
  const eyeU = ((bs.eyeLookUpLeft  ?? 0) + (bs.eyeLookUpRight  ?? 0)) / 2;
  const eyeD = ((bs.eyeLookDownLeft ?? 0) + (bs.eyeLookDownRight ?? 0)) / 2;

  const lx = eyeR - eyeL;
  const ly = eyeU - eyeD;
  const lz = 1.0;

  const d = matrix.data;
  const cx = d[0]*lx + d[4]*ly + d[8]*lz;
  const cy = d[1]*lx + d[5]*ly + d[9]*lz;
  const cz = d[2]*lx + d[6]*ly + d[10]*lz;

  const fwd = Math.abs(cz) > 0.01 ? cz : 0.01;
  const y = GAZE_INVERT_Y ? -(cy / fwd) : (cy / fwd);
  return { x: cx / fwd, y };
}

function deflectionToRegion(dx: number, dy: number): MemoryBucket | null {
  const ax = Math.abs(dx), ay = Math.abs(dy);
  if (ax < GAZE_LATERAL_DELTA && ay < GAZE_VERTICAL_DELTA) return "social";
  if (dx < -GAZE_LATERAL_DELTA && dy >  GAZE_VERTICAL_DELTA) return "family";
  if (dx >  GAZE_LATERAL_DELTA && dy >  GAZE_VERTICAL_DELTA) return "medical";
  if (dx < -GAZE_LATERAL_DELTA && dy < -GAZE_VERTICAL_DELTA) return "hobbies";
  if (dx >  GAZE_LATERAL_DELTA && dy < -GAZE_VERTICAL_DELTA) return "daily_routine";
  return null;
}

export class GazeTracker {
  private currentBucket: MemoryBucket | null = null;
  private dwellStart = 0;
  private dwellThresholdMs: number;
  private _activeZone: MemoryBucket | null = null;
  private _lastSeenAt = 0;
  private static ACTIVE_ZONE_TIMEOUT_MS = 500;

  constructor(dwellThresholdMs = 1500) {
    this.dwellThresholdMs = dwellThresholdMs;
  }

  get activeZone(): MemoryBucket | null {
    if (performance.now() - this._lastSeenAt > GazeTracker.ACTIVE_ZONE_TIMEOUT_MS) {
      this._activeZone = null;
    }
    return this._activeZone;
  }

  process(
    matrix: Matrix | null,
    bs: Record<string, number>,
    baseline: Baseline | null,
  ): MemoryBucket | null {
    if (!matrix) return null;

    const { x, y } = worldGazeXY(matrix, bs);
    const dx = baseline ? x - baseline.gaze.x : x;
    const dy = baseline ? y - baseline.gaze.y : y;

    const bucket = deflectionToRegion(dx, dy);
    if (bucket !== null) {
      this._activeZone = bucket;
      this._lastSeenAt = performance.now();
    }

    if (bucket !== this.currentBucket) {
      this.currentBucket = bucket;
      this.dwellStart = performance.now();
      return null;
    }

    if (bucket !== null &&
        performance.now() - this.dwellStart >= this.dwellThresholdMs) {
      this.currentBucket = null;
      this.dwellStart = 0;
      return bucket;
    }

    return null;
  }

  reset() {
    this.currentBucket = null;
    this._activeZone = null;
    this.dwellStart = 0;
    this._lastSeenAt = 0;
  }
}

interface AnglePoint { pitch: number; yaw: number; t: number }

const RAD2DEG = 180 / Math.PI;

export function extractAngles(
  data: number[],
): { pitch: number; yaw: number; roll: number } {
  const r20 = data[2], r21 = data[6], r22 = data[10];
  const r10 = data[1], r00 = data[0];
  return {
    pitch: Math.atan2(r21, r22),
    yaw:   Math.atan2(-r20, Math.sqrt(r21 * r21 + r22 * r22)),
    roll:  Math.atan2(r10, r00),
  };
}

const WINDOW_MS       = 1200;
const REFRACTORY_MS   = 2000;
const NOD_WINDOW_MS   = 1000;
// Hard cap covers backgrounded-tab catch-up where many frames arrive at once.
const HISTORY_MAX     = 100;

const SHAKE_RANGE_RAD     = 0.30;
const SHAKE_DEADBAND_RAD  = 0.05;
const SHAKE_MIN_REVERSALS = 3;

const NOD_AMPLITUDE_RAD = 0.12;
const NOD_SHARP_RAD     = 0.25;
const NOD_RECOVERY_RAD  = 0.12;
const NOD_MAX_YAW_RAD   = 0.25;

export class HeadPoseTracker {
  private history: AnglePoint[] = [];
  private lastEmitTs = 0;
  private lastDebug: HeadDebug = { pitch: 0, yaw: 0, roll: 0, crossings: 0 };

  process(matrix: Matrix, baseline: Baseline | null): HeadSignal | null {
    const raw = extractAngles(matrix.data);
    const pitch = baseline ? raw.pitch - baseline.head.pitch : raw.pitch;
    const yaw   = baseline ? raw.yaw   - baseline.head.yaw   : raw.yaw;
    const roll  = baseline ? raw.roll  - baseline.head.roll  : raw.roll;
    const now = performance.now();

    this.history.push({ pitch, yaw, t: now });
    const cutoff = now - WINDOW_MS;
    let drop = 0;
    while (drop < this.history.length && this.history[drop].t < cutoff) drop++;
    if (this.history.length - drop > HISTORY_MAX) {
      drop = this.history.length - HISTORY_MAX;
    }
    if (drop > 0) this.history.splice(0, drop);

    this.updateDebug(pitch, yaw, roll);

    if (now - this.lastEmitTs < REFRACTORY_MS) return null;
    if (this.history.length < 6) return null;

    const shake = this.detectShake();
    if (shake) { this.lastEmitTs = now; return shake; }

    const nod = this.detectNod(now);
    if (nod) { this.lastEmitTs = now; return nod; }

    return null;
  }

  private updateDebug(pitch: number, yaw: number, roll: number): void {
    let crossings = 0;
    let prevDir = 0;
    for (let i = 1; i < this.history.length; i++) {
      const diff = this.history[i].yaw - this.history[i - 1].yaw;
      if (Math.abs(diff) < SHAKE_DEADBAND_RAD) continue;
      const dir = diff > 0 ? 1 : -1;
      if (prevDir !== 0 && dir !== prevDir) crossings++;
      prevDir = dir;
    }
    this.lastDebug = {
      pitch: +(pitch * RAD2DEG).toFixed(1),
      yaw:   +(yaw   * RAD2DEG).toFixed(1),
      roll:  +(roll  * RAD2DEG).toFixed(1),
      crossings,
    };
  }

  private detectShake(): HeadSignal | null {
    const yaws = this.history.map((p) => p.yaw);
    const range = Math.max(...yaws) - Math.min(...yaws);
    if (range < SHAKE_RANGE_RAD) return null;

    let reversals = 0, prevDir = 0;
    for (let i = 1; i < yaws.length; i++) {
      const diff = yaws[i] - yaws[i - 1];
      if (Math.abs(diff) < SHAKE_DEADBAND_RAD) continue;
      const dir = diff > 0 ? 1 : -1;
      if (prevDir !== 0 && dir !== prevDir) reversals++;
      prevDir = dir;
    }
    return reversals >= SHAKE_MIN_REVERSALS ? "HEAD_SHAKE" : null;
  }

  private detectNod(now: number): HeadSignal | null {
    const recent = this.history.filter((p) => p.t >= now - NOD_WINDOW_MS);
    if (recent.length < 6) return null;

    const yawRange = Math.max(...recent.map((p) => Math.abs(p.yaw)));
    if (yawRange > NOD_MAX_YAW_RAD) return null;

    const pitches = recent.map((p) => p.pitch);
    const startPitch = pitches[0];
    const maxDev = Math.max(...pitches.map((p) => Math.abs(p - startPitch)));
    if (maxDev < NOD_AMPLITUDE_RAD) return null;

    const lastPitch = pitches[pitches.length - 1];
    if (Math.abs(lastPitch - startPitch) >= NOD_RECOVERY_RAD) return null;

    return maxDev >= NOD_SHARP_RAD ? "HEAD_NOD_DISSATISFIED" : "HEAD_NOD";
  }

  get debug(): HeadDebug { return this.lastDebug; }

  reset(): void {
    this.history = [];
    this.lastEmitTs = 0;
  }
}

export function faceBboxSize(landmarks: { x: number; y: number }[]): number | null {
  if (!landmarks || landmarks.length < 3) return null;
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  for (const p of landmarks) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const w = maxX - minX;
  const h = maxY - minY;
  if (w <= 0 || h <= 0) return null;
  return Math.sqrt(w * h);
}

const INDEX_TIP = 8;
const VELOCITY_START = 15;
const VELOCITY_END = 5;
const STROKE_GAP_MS = 200;

export class AirWriter {
  private trajectory: [number, number][] = [];
  private inStroke = false;
  private strokeEndTime = 0;
  private prevPt: [number, number] | null = null;
  private pendingStroke: [number, number][] | null = null;

  processHandLandmarks(
    landmarks: { x: number; y: number }[],
    frameWidth: number,
    frameHeight: number
  ): void {
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
      return;
    }

    if (this.inStroke && velocity < VELOCITY_END) {
      if (this.strokeEndTime === 0) {
        this.strokeEndTime = performance.now();
      }
      this.checkStrokeEnd();
    }
  }

  private checkStrokeEnd(): void {
    if (!this.inStroke || this.strokeEndTime === 0) return;
    if (performance.now() - this.strokeEndTime >= STROKE_GAP_MS) {
      if (this.trajectory.length >= 5) {
        this.pendingStroke = [...this.trajectory];
      }
      this.trajectory = [];
      this.inStroke = false;
      this.strokeEndTime = 0;
    }
  }

  get strokeActive(): boolean {
    return this.inStroke;
  }

  getCompletedStroke(): [number, number][] | null {
    const s = this.pendingStroke;
    this.pendingStroke = null;
    return s;
  }

  noHand(): void {
    if (this.inStroke && this.strokeEndTime === 0) {
      this.strokeEndTime = performance.now();
    }
    this.prevPt = null;
    this.checkStrokeEnd();
  }
}

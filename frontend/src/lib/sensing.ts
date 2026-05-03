import type { Affect, GestureName, MemoryBucket } from "../types";

// ── Affect classification via MediaPipe blendshapes ──────────────────────────

export function classifyAffect(bs: Record<string, number>): Affect {
  const smileLeft   = bs["mouthSmileLeft"]  ?? 0;
  const smileRight  = bs["mouthSmileRight"] ?? 0;
  const browDownL   = bs["browDownLeft"]    ?? 0;
  const browDownR   = bs["browDownRight"]   ?? 0;
  const squintL     = bs["eyeSquintLeft"]   ?? 0;
  const squintR     = bs["eyeSquintRight"]  ?? 0;
  const jawOpen     = bs["jawOpen"]         ?? 0;
  const browInnerUp = bs["browInnerUp"]     ?? 0;

  if (jawOpen > 0.4 && browInnerUp > 0.5) return "SURPRISED";
  if (browDownL > 0.4 || browDownR > 0.4) return "FRUSTRATED";
  if (squintL > 0.5 && squintR > 0.5)     return "FRUSTRATED";
  if (smileLeft > 0.5 && smileRight > 0.5) return "HAPPY";
  return "NEUTRAL";
}

// ── Gesture label mapping from MediaPipe GestureRecognizer ───────────────────

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

// ── Air-writing stroke collector (recognition via Gemini Vision) ─────────────

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

  // Returns the completed stroke trajectory and clears it (call once per frame).
  getCompletedStroke(): [number, number][] | null {
    const s = this.pendingStroke;
    this.pendingStroke = null;
    return s;
  }

  // Kept for API compatibility — always returns "".
  getText(): string {
    return "";
  }

  noHand(): void {
    if (this.inStroke && this.strokeEndTime === 0) {
      this.strokeEndTime = performance.now();
    }
    this.prevPt = null;
    this.checkStrokeEnd();
  }
}

import type { Matrix } from "@mediapipe/tasks-vision";
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

// ── Gaze tracker — world-space gaze via head rotation × eye blendshapes ──────
//
// Old approach: absolute iris X/Y position in frame → grid region.
//   Problem: head shifting in frame changes the bucket even if eyes didn't move.
//
// New approach:
//   1. Eye direction in face-local space from blendshapes (head-relative).
//   2. Rotate into camera space using the facial transformation matrix.
//   3. Perspective-project to a 2-D screen gaze point.
//   4. Map that point to the 5 memory buckets with a dwell timer.
//
// Bucket layout (matches the 5 regions on the AAC interface):
//
//   family     │  medical
//   (top-left) │  (top-right)
//   ───────────┼───────────
//   hobbies    │  daily_routine
//   (bot-left) │  (bot-right)
//            social
//           (centre)
//
// If top/bottom buckets appear swapped on your device, set VITE_GAZE_INVERT_Y=true.
const GAZE_INVERT_Y = import.meta.env.VITE_GAZE_INVERT_Y === "true";
const GAZE_CENTER      = 0.10;  // radius around origin treated as "social"
const GAZE_LATERAL     = 0.12;  // |x| must exceed this for left/right split
const GAZE_VERTICAL    = 0.12;  // |y| must exceed this for top/bottom split

function worldGazeXY(
  matrix: Matrix,
  bs: Record<string, number>,
): { x: number; y: number } {
  // Eye direction in face-local space.
  // MediaPipe "In" = toward nose, "Out" = away from nose.
  // viewer-right  = InLeft  + OutRight
  // viewer-left   = OutLeft + InRight
  const eyeR = ((bs.eyeLookInLeft  ?? 0) + (bs.eyeLookOutRight ?? 0)) / 2;
  const eyeL = ((bs.eyeLookOutLeft ?? 0) + (bs.eyeLookInRight  ?? 0)) / 2;
  const eyeU = ((bs.eyeLookUpLeft  ?? 0) + (bs.eyeLookUpRight  ?? 0)) / 2;
  const eyeD = ((bs.eyeLookDownLeft ?? 0) + (bs.eyeLookDownRight ?? 0)) / 2;

  // Face-local gaze vector (+X right, +Y up, +Z forward toward camera).
  const lx = eyeR - eyeL;
  const ly = eyeU - eyeD;
  const lz = 1.0; // canonical forward direction

  // Rotate to camera space using the 3×3 submatrix of the column-major 4×4.
  // R[row][col] = data[col*4 + row]
  const d = matrix.data;
  const cx = d[0]*lx + d[4]*ly + d[8]*lz;
  const cy = d[1]*lx + d[5]*ly + d[9]*lz;
  const cz = d[2]*lx + d[6]*ly + d[10]*lz;

  // Perspective-project onto screen plane.
  const fwd = Math.abs(cz) > 0.01 ? cz : 0.01;
  const y = GAZE_INVERT_Y ? -(cy / fwd) : (cy / fwd);
  return { x: cx / fwd, y };
}

function gazeToRegion(x: number, y: number): MemoryBucket | null {
  const ax = Math.abs(x), ay = Math.abs(y);
  if (ax < GAZE_CENTER && ay < GAZE_CENTER) return "social";
  if (ax < GAZE_LATERAL && ay < GAZE_VERTICAL) return "social"; // near-centre
  if (x < -GAZE_LATERAL && y >  GAZE_VERTICAL) return "family";
  if (x >  GAZE_LATERAL && y >  GAZE_VERTICAL) return "medical";
  if (x < -GAZE_LATERAL && y < -GAZE_VERTICAL) return "hobbies";
  if (x >  GAZE_LATERAL && y < -GAZE_VERTICAL) return "daily_routine";
  return null; // edge zone — don't fire
}

export class GazeTracker {
  private currentBucket: MemoryBucket | null = null;
  private dwellStart = 0;
  private dwellThresholdMs: number;
  private _activeZone: MemoryBucket | null = null;

  constructor(dwellThresholdMs = 1500) {
    this.dwellThresholdMs = dwellThresholdMs;
  }

  // Current zone the user is looking at right now — updates every frame.
  // Use this to highlight the zone map immediately.
  get activeZone(): MemoryBucket | null {
    return this._activeZone;
  }

  process(
    matrix: Matrix | null,
    bs: Record<string, number>,
  ): MemoryBucket | null {
    const { x, y } = matrix
      ? worldGazeXY(matrix, bs)
      : { x: 0, y: 0 };

    const bucket = matrix ? gazeToRegion(x, y) : null;
    this._activeZone = bucket; // always reflect current zone

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
  }
}

// ── Head-pose tracker using facial transformation matrix ────────────────────
//
// MediaPipe FaceLandmarker produces a 4×4 column-major transformation matrix
// that encodes the 3-D rotation of the canonical face model in camera space.
// We decompose it to Euler angles (ZYX convention) — no calibration step needed
// because the angles are always relative to the canonical neutral pose.
//
// Signals emitted:
//   HEAD_SHAKE           — yaw oscillates ±N° (left/right), "no"
//   HEAD_NOD             — gentle pitch dip + recovery, "yes"
//   HEAD_NOD_DISSATISFIED — sharp/large pitch dip + recovery, discomfort

export type HeadSignal = "HEAD_SHAKE" | "HEAD_NOD" | "HEAD_NOD_DISSATISFIED";

export interface HeadDebug {
  pitch: number;     // degrees — nod angle
  yaw: number;       // degrees — shake angle
  roll: number;      // degrees — tilt angle
  crossings: number; // yaw direction reversals in current window
}

interface AnglePoint { pitch: number; yaw: number; t: number }

const RAD2DEG = 180 / Math.PI;

function extractAngles(data: Float32Array): { pitch: number; yaw: number; roll: number } {
  // Column-major 4×4: R[row][col] = data[col*4 + row]
  // ZYX Euler (R = Rz·Ry·Rx):
  //   pitch (X, nod)   = atan2(R[2][1], R[2][2]) = atan2(data[6],  data[10])
  //   yaw   (Y, shake) = atan2(−R[2][0], √(R[2][1]²+R[2][2]²))
  //   roll  (Z, tilt)  = atan2(R[1][0], R[0][0])  = atan2(data[1],  data[0])
  const r20 = data[2], r21 = data[6], r22 = data[10];
  const r10 = data[1], r00 = data[0];
  return {
    pitch: Math.atan2(r21, r22),
    yaw:   Math.atan2(-r20, Math.sqrt(r21 * r21 + r22 * r22)),
    roll:  Math.atan2(r10, r00),
  };
}

// Thresholds (radians unless noted)
const WINDOW_MS       = 1200;
const REFRACTORY_MS   = 2000;
const NOD_WINDOW_MS   = 1000;

const SHAKE_RANGE_RAD    = 0.30;  // total yaw swing needed (~17°)
const SHAKE_DEADBAND_RAD = 0.05;  // ignore jitter below ~3°
const SHAKE_MIN_REVERSALS = 3;

const NOD_AMPLITUDE_RAD   = 0.15; // ~8.6° — min pitch deviation for any nod
const NOD_SHARP_RAD       = 0.28; // ~16° — above this = DISSATISFIED
const NOD_RECOVERY_RAD    = 0.15; // must return within ~8.6° of start pitch
const NOD_MAX_YAW_RAD     = 0.25; // reject if too much lateral (~14°)

export class HeadPoseTracker {
  private history: AnglePoint[] = [];
  private lastEmitTs = 0;
  private lastDebug: HeadDebug = { pitch: 0, yaw: 0, roll: 0, crossings: 0 };

  // No-op — angles are self-calibrating relative to the canonical face model.
  // Kept so existing callers (calibrateHeadPose button) don't break.
  calibrate(_landmarks: unknown): void {}

  process(matrix: Matrix): HeadSignal | null {
    const { pitch, yaw, roll } = extractAngles(matrix.data);
    const now = performance.now();

    this.history.push({ pitch, yaw, t: now });
    this.history = this.history.filter((p) => p.t >= now - WINDOW_MS);

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

    // Reject if there's significant lateral motion — it's a shake, not a nod.
    const yawRange = Math.max(...recent.map((p) => Math.abs(p.yaw)));
    if (yawRange > NOD_MAX_YAW_RAD) return null;

    const pitches = recent.map((p) => p.pitch);
    const startPitch = pitches[0];
    const maxDev = Math.max(...pitches.map((p) => Math.abs(p - startPitch)));
    if (maxDev < NOD_AMPLITUDE_RAD) return null;

    // Must recover back near the start pitch.
    const lastPitch = pitches[pitches.length - 1];
    if (Math.abs(lastPitch - startPitch) >= NOD_RECOVERY_RAD) return null;

    return maxDev >= NOD_SHARP_RAD ? "HEAD_NOD_DISSATISFIED" : "HEAD_NOD";
  }

  get debug(): HeadDebug { return this.lastDebug; }

  reset(): void {
    this.history = [];
    this.lastEmitTs = 0;
  }

  // Always true — no manual calibration step required with the matrix approach.
  get calibrated(): boolean { return true; }
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

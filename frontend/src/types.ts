export type Affect = "HAPPY" | "FRUSTRATED" | "NEUTRAL" | "SURPRISED";
export type GestureName = "THUMBS_UP" | "THUMBS_DOWN" | "POINTING" | "WAVING";
export type MemoryBucket = "family" | "medical" | "hobbies" | "daily_routine" | "social";
export type HeadSignal = "HEAD_SHAKE" | "HEAD_NOD_DISSATISFIED";

export interface HeadDebug {
  dx: number;
  dy: number;
  maxAbsDx: number;
  maxAbsDy: number;
  crossings: number;
}

export interface SensingState {
  affect: Affect | null;
  gestureTag: GestureName | null;
  gazeBucket: MemoryBucket | null;
  airWrittenText: string;
  headSignal: HeadSignal | null;
  headCalibrated: boolean;
  headDebug: HeadDebug;
}

export interface Persona {
  id: string;
  name: string;
  condition: string;
  style: string;
}

export type ResolvedSource =
  | "voice_only"
  | "air_only"
  | "agree"
  | "conflict_air"
  | "conflict_voice"
  | "none";

export interface ResolvedIntent {
  text: string;
  source: ResolvedSource;
  voice_text: string | null;
  air_text: string | null;
}

export interface ChatRequest {
  user_id: string;
  query: string;
  affect_override: Affect | null;
  gesture_tag: GestureName | null;
  gaze_bucket: MemoryBucket | null;
  air_written_text: string | null;
  head_signal?: HeadSignal | null;
  voice_text?: string | null;
  resolved_intent?: ResolvedIntent | null;
}

export interface TurnaroundRequest {
  user_id: string;
  turn_id?: number;
  head_signal?: HeadSignal | null;
}

export interface LatencyLog {
  t_sensing: number;
  t_intent: number;
  t_retrieval: number;
  t_generation: number;
  t_total: number;
}

export interface EvalScores {
  groundedness: number;
  hallucination_rate: number;
  no_evidence: boolean;
  t_total_s: number;
  slo_target_s: number;
  slo_passed: boolean;
  slo_margin_s: number;
  multimodal_alignment: number;
  affect_alignment: number;
  gesture_alignment: number;
  gaze_alignment: number;
}

export type CandidateStrategy =
  | "broad"
  | "focused"
  | "serendipitous"
  | "side_index";

export interface Candidate {
  text: string;
  strategy: CandidateStrategy | string;
  grounded_buckets: string[];
}

export interface ChatResponse {
  user_id: string;
  query: string;
  response: string;
  candidates: Candidate[];
  affect: string;
  llm_tier: string;
  retrieval_mode: string;
  latency: LatencyLog;
  guardrail_passed: boolean;
  run_id: string | null;
  turn_id: number;
  eval_scores?: EvalScores | null;
}

export interface ChatMessage {
  role: "partner" | "aac_user";
  content: string;
  latency?: LatencyLog;
  affect?: string;
  runId?: string | null;
  turnId?: number;
  rephrased?: boolean;
  isTurnaround?: boolean;
  evalScores?: EvalScores | null;
  candidates?: Candidate[];
  // picked becomes true after the user clicks one — also locks in `content` to the picked text
  picked?: boolean;
  pickedIdx?: number;
  // Candidates from prior regeneration rounds — rendered struck-through above the active picker
  rejectedRounds?: Candidate[][];
}

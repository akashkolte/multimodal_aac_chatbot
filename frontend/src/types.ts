export type Affect = "HAPPY" | "FRUSTRATED" | "NEUTRAL" | "SURPRISED";
export type GestureName = "THUMBS_UP" | "THUMBS_DOWN" | "POINTING" | "WAVING";
export type MemoryBucket = "family" | "medical" | "hobbies" | "daily_routine" | "social";

export interface SensingState {
  affect: Affect | null;
  gestureTag: GestureName | null;
  gazeBucket: MemoryBucket | null;
  airWrittenText: string;
}

export interface Persona {
  id: string;
  name: string;
  condition: string;
  style: string;
}

export interface ChatRequest {
  user_id: string;
  query: string;
  affect_override: Affect | null;
  gesture_tag: GestureName | null;
  gaze_bucket: MemoryBucket | null;
  air_written_text: string | null;
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

export interface ChatResponse {
  user_id: string;
  query: string;
  response: string;
  affect: string;
  llm_tier: string;
  retrieval_mode: string;
  latency: LatencyLog;
  guardrail_passed: boolean;
  eval_scores: EvalScores | null;
}

export interface ChatMessage {
  role: "partner" | "aac_user";
  content: string;
  latency?: LatencyLog;
  affect?: string;
  evalScores?: EvalScores | null;
}

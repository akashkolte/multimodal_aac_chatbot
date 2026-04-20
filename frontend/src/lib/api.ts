import type {
  ChatRequest,
  ChatResponse,
  Persona,
  TurnaroundRequest,
} from "../types";

const API_BASE = "";

export async function fetchUsers(): Promise<Persona[]> {
  const res = await fetch(`${API_BASE}/users`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const data = await res.json();
  return data.users;
}

export async function sendChat(req: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function sendTurnaround(
  req: TurnaroundRequest
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat/turnaround`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function resetSession(userId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE}/session/reset?user_id=${encodeURIComponent(userId)}`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error(`API error: ${res.status}`);
}

export type StreamEvent =
  | { type: "candidate_start"; idx: number; strategy: string; grounded_buckets: string[] }
  | { type: "token"; idx: number; delta: string }
  | { type: "candidate_done"; idx: number; text: string }
  | { type: "candidate_error"; idx: number; error: string }
  | { type: "complete"; response: ChatResponse }
  | { type: "error"; message: string };

async function readSSE(
  res: Response,
  onEvent: (evt: StreamEvent) => void,
): Promise<void> {
  if (!res.body) throw new Error("no response body");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const emitFrame = (frame: string) => {
    const line = frame.split("\n").find((l) => l.startsWith("data:"));
    if (!line) return;
    const json = line.slice(5).trim();
    if (!json) return;
    try {
      onEvent(JSON.parse(json) as StreamEvent);
    } catch (e) {
      console.warn("SSE parse failed", e, json.slice(0, 200));
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE frames are separated by blank lines.
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) emitFrame(part);
  }
  // Server closed cleanly but the final frame didn't end with \n\n —
  // emit whatever remains so the terminal event isn't dropped.
  if (buffer.trim()) emitFrame(buffer);
}

export async function streamChat(
  req: ChatRequest,
  onEvent: (evt: StreamEvent) => void,
): Promise<void> {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  await readSSE(res, onEvent);
}

export async function streamRegenerate(
  args: { user_id: string; turn_id: number; rejected_texts: string[] },
  onEvent: (evt: StreamEvent) => void,
): Promise<void> {
  const res = await fetch(`${API_BASE}/chat/regenerate/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  await readSSE(res, onEvent);
}

export async function sendRegenerate(args: {
  user_id: string;
  turn_id: number;
  rejected_texts: string[];
}): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat/regenerate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function sendPick(args: {
  run_id: string;
  user_id: string;
  picked_idx: number;
}): Promise<void> {
  const res = await fetch(`${API_BASE}/chat/pick`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
}

export async function submitRating(args: {
  run_id: string;
  user_id: string;
  authenticity: number;
  rater_id?: string;
  notes?: string;
}): Promise<void> {
  const res = await fetch(`${API_BASE}/feedback/rating`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) return false;
    const data = await res.json();
    return data.models_ready === true;
  } catch {
    return false;
  }
}

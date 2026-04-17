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

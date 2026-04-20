import { DEFAULT_AIR_TEMPLATES } from "./airTemplates";

// Canonical AAC tokens that carry high signal when someone air-writes them —
// short, action-oriented, and hard to confuse for casual chat. When the
// voice transcript and the air-written text disagree, these tokens win.
const AAC_PRIORITY_TOKENS: ReadonlySet<string> = new Set(
  ["help", "stop", "water", "done", "more"].filter((t) =>
    DEFAULT_AIR_TEMPLATES.has(t)
  )
);

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

function normalise(s: string | null | undefined): string {
  return (s ?? "").trim().toLowerCase();
}

function tokens(s: string): Set<string> {
  return new Set(
    s
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 1)
  );
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let inter = 0;
  for (const tok of a) if (b.has(tok)) inter++;
  const union = a.size + b.size - inter;
  return union === 0 ? 0 : inter / union;
}

export function resolveIntent(
  voiceRaw: string | null,
  airRaw: string | null
): ResolvedIntent {
  const voice = normalise(voiceRaw);
  const air = normalise(airRaw);

  if (!voice && !air) {
    return { text: "", source: "none", voice_text: null, air_text: null };
  }
  if (voice && !air) {
    return {
      text: voice,
      source: "voice_only",
      voice_text: voice,
      air_text: null,
    };
  }
  if (!voice && air) {
    return { text: air, source: "air_only", voice_text: null, air_text: air };
  }

  // Both present.
  const voiceTokens = tokens(voice);
  const airTokens = tokens(air);
  const overlap = jaccard(voiceTokens, airTokens);

  // Air-text appears as a substring of the voice transcript (or vice versa) —
  // user probably said the word while also writing it. Treat as agreement.
  const substringHit =
    voice.includes(air) || air.includes(voice) || overlap >= 0.5;

  if (substringHit) {
    // Prefer the longer / richer form (usually voice), but mark source as agree.
    const winner = voice.length >= air.length ? voice : air;
    return {
      text: winner,
      source: "agree",
      voice_text: voice,
      air_text: air,
    };
  }

  // Genuine conflict. AAC priority tokens (help/stop/water/done/more) dominate.
  if (AAC_PRIORITY_TOKENS.has(air)) {
    return {
      text: air,
      source: "conflict_air",
      voice_text: voice,
      air_text: air,
    };
  }

  // Otherwise voice wins — higher information density.
  return {
    text: voice,
    source: "conflict_voice",
    voice_text: voice,
    air_text: air,
  };
}

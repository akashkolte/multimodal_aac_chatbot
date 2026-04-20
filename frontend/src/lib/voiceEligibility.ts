// Personas for whom a live-mic button makes sense.
// Gate reflects each persona's real-world speech access, not the in-universe
// voice of the character: we hide the mic whenever the modelled access method
// is non-verbal (locked-in, letterboard, dictation-to-assistant, etc.), even
// if the character can "speak" in their canon.
export const VOICE_CAPABLE_PERSONAS: ReadonlySet<string> = new Set([
  "abed_nadir",
  "allie_calhoun",
  "forrest_gump",
  "gabby_giffords",
  "michael_j_fox",
  "raymond_babbitt",
  "walter_jr_white",
]);

export function isVoiceCapable(userId: string | null): boolean {
  return !!userId && VOICE_CAPABLE_PERSONAS.has(userId);
}

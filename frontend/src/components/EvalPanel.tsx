import { memo, useState } from "react";
import { submitRating } from "../lib/api";
import type { EvalScores } from "../types";

interface Props {
  runId: string;
  userId: string;
  latencyTotal: number;
  sloTarget?: number;
  evalScores?: EvalScores | null;
}

function gradeClass(score: number): string {
  if (score >= 0.75) return "good";
  if (score >= 0.4) return "mid";
  return "bad";
}

function fmt(score: number): string {
  return (score * 100).toFixed(0) + "%";
}

function buildTip(parts: {
  title: string;
  question: string;
  how?: string;
  thisTurn?: string;
  fallback?: string;
}): string {
  const header = `${parts.title} — ${parts.question}`;
  if (parts.fallback) return `${header}\n\n${parts.fallback}`;
  const sections = [parts.how, parts.thisTurn].filter(Boolean);
  return sections.length ? `${header}\n\n${sections.join("\n\n")}` : header;
}

function groundednessTip(s: EvalScores): string {
  const title = "GROUNDED";
  if (s.no_evidence) {
    return buildTip({
      title,
      question: "Did the response stick to the retrieved memories?",
      fallback:
        "Not scored: no memories were retrieved this turn (e.g. a 'how are you feeling?' question that skips retrieval).",
    });
  }
  const total = s.sentences_total ?? 0;
  const grounded = s.sentences_grounded ?? 0;
  const thr = s.nli_threshold ?? 0.5;
  return buildTip({
    title,
    question: "Did the response stick to the retrieved memories, or hallucinate?",
    how:
      `How: each sentence in the response is checked against each retrieved chunk with an NLI model. ` +
      `A sentence counts as grounded if at least one chunk entails it with probability ≥ ${thr.toFixed(2)}.`,
    thisTurn:
      `This turn: ${grounded}/${total} sentences grounded → ${fmt(s.groundedness)}. ` +
      `Hallucination = ${fmt(s.hallucination_rate)} (${total - grounded} unsupported).`,
  });
}

function relevanceTip(s: EvalScores): string {
  return buildTip({
    title: "RELEVANT",
    question: "Did the response actually address the partner's question?",
    how:
      "How: cosine similarity between the BGE embedding of the query and the embedding of the response. " +
      "Higher = more semantically on-topic.",
    thisTurn: `This turn: ${(s.relevance ?? 0).toFixed(3)} → ${fmt(s.relevance ?? 0)}.`,
  });
}

function affectTip(s: EvalScores): string {
  const question = "Does the response tone match the detected facial expression?";
  const ex = s.explain?.affect;
  if (!ex) {
    return buildTip({ title: "AFFECT", question });
  }
  return buildTip({
    title: "AFFECT",
    question,
    how:
      "How: response sentiment is computed from positive vs negative word counts, " +
      "then compared to the affect target.",
    thisTurn:
      `This turn: detected ${ex.target}, response sentiment = ${ex.sentiment.toFixed(2)} ` +
      `(${ex.pos_words} positive word${ex.pos_words === 1 ? "" : "s"}, ` +
      `${ex.neg_words} negative) → ${fmt(s.affect_alignment)}.`,
  });
}

function gestureTip(s: EvalScores): string {
  const title = "GESTURE";
  const question = "Does the response opener acknowledge the detected hand gesture?";
  const ex = s.explain?.gesture;
  if (!ex) {
    return buildTip({ title, question, fallback: "No gesture detected this turn — defaults to 0." });
  }
  if (!ex.has_pattern) {
    return buildTip({
      title,
      question,
      fallback: `Detected ${ex.tag}, but this gesture has no opener pattern to test — partial credit (50%).`,
    });
  }
  return buildTip({
    title,
    question,
    how: `How: regex check on the first words of the response (e.g. THUMBS_UP expects 'yes/sure/absolutely…').`,
    thisTurn:
      `This turn: detected ${ex.tag}, opener ${ex.matched ? "matched" : "did not match"} ` +
      `→ ${ex.matched ? "100%" : "0%"}.`,
  });
}

function gazeTip(s: EvalScores): string {
  const title = "GAZE";
  const question = "Did the retrieved memories come from the topic the user was looking at?";
  const ex = s.explain?.gaze;
  if (!ex) {
    return buildTip({ title, question, fallback: "No gaze bucket detected this turn — defaults to 0." });
  }
  if (ex.total_chunks === 0) {
    return buildTip({
      title,
      question,
      fallback: `User was looking at: ${ex.bucket}. No chunks retrieved this turn — defaults to 0.`,
    });
  }
  return buildTip({
    title,
    question,
    how: `How: fraction of retrieved chunks whose 'bucket' label matches the gaze target.`,
    thisTurn:
      `This turn: user looking at ${ex.bucket}, ${ex.matched_chunks}/${ex.total_chunks} ` +
      `retrieved chunks matched → ${fmt(s.gaze_alignment)}.`,
  });
}

function diversityTip(s: EvalScores): string {
  const title = "DIVERSITY";
  const question = "How different are the candidate responses the picker showed?";
  const n = s.n_candidates ?? 0;
  const d = s.candidate_diversity ?? 0;
  if (n < 2) {
    return buildTip({ title, question, fallback: `Only ${n} candidate this turn — not meaningful.` });
  }
  return buildTip({
    title,
    question,
    how:
      "How: average pairwise cosine distance between BGE embeddings of the candidate texts. " +
      "High = varied alternatives. Low = three paraphrases of the same answer (the 'aloha' problem).",
    thisTurn: `This turn: ${n} candidates, mean pairwise distance = ${d.toFixed(3)} → ${fmt(d)}.`,
  });
}

function sloTip(
  s: EvalScores | null | undefined,
  fallbackLatency: number,
  fallbackTarget: number,
  fallbackPassed: boolean,
): string {
  const latency = s?.t_total_s ?? fallbackLatency;
  const target = s?.slo_target_s ?? fallbackTarget;
  const passed = s?.slo_passed ?? fallbackPassed;
  const margin = s?.slo_margin_s;
  const sign = (margin ?? 0) >= 0 ? "+" : "";
  const m = margin !== undefined ? `${sign}${margin.toFixed(2)}s` : "";
  return buildTip({
    title: "LATENCY",
    question: "Did the response arrive within the SLO budget?",
    thisTurn:
      `Target: < ${target.toFixed(1)}s. ` +
      `This turn: ${latency.toFixed(2)}s${m ? ` (${m} margin)` : ""} — ${passed ? "passed ✓" : "failed ✗"}.`,
  });
}

function EvalPanelImpl({
  runId,
  userId,
  latencyTotal,
  sloTarget = 6.0,
  evalScores,
}: Props) {
  const [value, setValue] = useState<number | null>(null);
  const [hover, setHover] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const sloPassed = evalScores
    ? evalScores.slo_passed
    : latencyTotal > 0 && latencyTotal < sloTarget;
  const effectiveLatency = evalScores?.t_total_s ?? latencyTotal;
  const showDiversity =
    evalScores && (evalScores.n_candidates ?? 0) >= 2;
  const showRelevance = evalScores && evalScores.relevance !== undefined;

  async function rate(stars: number) {
    if (submitting || value !== null) return;
    setSubmitting(true);
    try {
      await submitRating({
        run_id: runId,
        user_id: userId,
        authenticity: stars,
      });
      setValue(stars);
    } catch (e) {
      console.error("rating submit failed", e);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="eval-panel">
      <div className="eval-row">
        {effectiveLatency > 0 && (
          <span
            className="tip"
            data-tip={sloTip(evalScores, effectiveLatency, sloTarget, sloPassed)}
          >
            <span className={`slo-badge ${sloPassed ? "pass" : "fail"}`}>
              {effectiveLatency.toFixed(2)}s {sloPassed ? "✓" : "✗"}
            </span>
          </span>
        )}
        {evalScores && (
          <>
            <span className="tip" data-tip={groundednessTip(evalScores)}>
              <span
                className={`eval-pill ${
                  evalScores.no_evidence ? "muted" : gradeClass(evalScores.groundedness)
                }`}
              >
                grounded {evalScores.no_evidence ? "—" : fmt(evalScores.groundedness)}
              </span>
            </span>
            {showRelevance && (
              <span className="tip" data-tip={relevanceTip(evalScores)}>
                <span className={`eval-pill ${gradeClass(evalScores.relevance ?? 0)}`}>
                  relevant {fmt(evalScores.relevance ?? 0)}
                </span>
              </span>
            )}
            <span className="tip" data-tip={affectTip(evalScores)}>
              <span className={`eval-pill ${gradeClass(evalScores.affect_alignment)}`}>
                affect {fmt(evalScores.affect_alignment)}
              </span>
            </span>
            <span className="tip" data-tip={gestureTip(evalScores)}>
              <span className={`eval-pill ${gradeClass(evalScores.gesture_alignment)}`}>
                gesture {fmt(evalScores.gesture_alignment)}
              </span>
            </span>
            <span className="tip" data-tip={gazeTip(evalScores)}>
              <span className={`eval-pill ${gradeClass(evalScores.gaze_alignment)}`}>
                gaze {fmt(evalScores.gaze_alignment)}
              </span>
            </span>
            {showDiversity && (
              <span className="tip" data-tip={diversityTip(evalScores)}>
                <span className={`eval-pill ${gradeClass(evalScores.candidate_diversity ?? 0)}`}>
                  diversity {fmt(evalScores.candidate_diversity ?? 0)}
                </span>
              </span>
            )}
          </>
        )}
        <div className="tip star-rating" data-tip="Rate how authentic this response felt as the persona (1 = off, 5 = spot on). Logged to ratings.jsonl.">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              className={`star ${star <= (hover || (value ?? 0)) ? "active" : ""}`}
              onMouseEnter={() => setHover(star)}
              onMouseLeave={() => setHover(0)}
              onClick={() => rate(star)}
              disabled={value !== null || submitting}
            >
              ★
            </button>
          ))}
          {value !== null && <span className="star-label">{value}/5</span>}
        </div>
      </div>
    </div>
  );
}

export const EvalPanel = memo(EvalPanelImpl);

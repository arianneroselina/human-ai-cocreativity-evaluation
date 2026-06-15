"use client";

import { useState } from "react";
import { Button } from "@/components/shadcn_ui/button";
import { Textarea } from "@/components/shadcn_ui/textarea";
import LikertRow, { type Likert } from "@/components/ui/likertRow";
import { delay } from "@/lib/utils";
import { Poem } from "@/lib/evaluation/types";

type RatingResult = {
  sessionId: string;
  poemId: string;
  fluency: Likert;
  themeAlignment: Likert;
  meaningfulness: Likert;
  poeticness: Likert;
  overallQuality: Likert;
  comment: string;
};

type EvaluationWorkbenchProps = {
  poems: Poem[];
};

export default function EvaluationWorkbench({ poems }: EvaluationWorkbenchProps) {
  const [evaluatorId, setEvaluatorId] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [ratedPoemIds, setRatedPoemIds] = useState<string[]>([]);
  const [isLoadingSession, setIsLoadingSession] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);

  const [currentIndex, setCurrentIndex] = useState(0);
  const [ratings, setRatings] = useState<RatingResult[]>([]);

  const [fluency, setFluency] = useState<Likert | null>(null);
  const [themeAlignment, setThemeAlignment] = useState<Likert | null>(null);
  const [meaningfulness, setMeaningfulness] = useState<Likert | null>(null);
  const [poeticness, setPoeticness] = useState<Likert | null>(null);
  const [overallQuality, setOverallQuality] = useState<Likert | null>(null);
  const [comment, setComment] = useState("");

  const [isChangingPoem, setIsChangingPoem] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const currentPoem = poems[currentIndex];
  const isFinished = hasStarted && currentIndex >= poems.length;
  const canSubmit = fluency && themeAlignment && meaningfulness && poeticness && overallQuality;

  async function handleStart() {
    let cleanEvaluatorCode = evaluatorId.trim();
    if (!cleanEvaluatorCode || !["1", "2"].includes(cleanEvaluatorCode)) {
      alert("Please choose your evaluator ID.");
      return;
    }

    try {
      setIsLoadingSession(true);

      const response = await fetch("/api/evaluation-session", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          evaluatorId: cleanEvaluatorCode,
        }),
      });

      if (!response.ok) {
        throw new Error("Could not load evaluation session.");
      }

      const data: {
        sessionId: string;
        ratedPoemIds: string[];
      } = await response.json();

      const ratedSet = new Set(data.ratedPoemIds);

      const nextIndex = poems.findIndex((poem) => !ratedSet.has(poem.id));

      setSessionId(data.sessionId);
      setRatedPoemIds(data.ratedPoemIds);
      setCurrentIndex(nextIndex === -1 ? poems.length : nextIndex);
      setHasStarted(true);
    } catch (error) {
      console.error(error);
      alert("Could not load your evaluation session.");
    } finally {
      setIsLoadingSession(false);
    }
  }

  function resetForm() {
    setFluency(null);
    setThemeAlignment(null);
    setMeaningfulness(null);
    setPoeticness(null);
    setOverallQuality(null);
    setComment("");
  }

  async function handleSubmit() {
    if (!currentPoem || !sessionId) return;

    if (!fluency || !themeAlignment || !meaningfulness || !poeticness || !overallQuality) {
      alert("Please rate all four metrics before continuing.");
      return;
    }

    const newRating: RatingResult = {
      sessionId,
      poemId: currentPoem.id,
      fluency,
      themeAlignment,
      meaningfulness,
      poeticness,
      overallQuality,
      comment,
    };

    try {
      setIsSubmitting(true);

      const response = await fetch("/api/ratings", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(newRating),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.error ?? "Failed to submit rating.");
      }

      setRatings((previous) => [...previous, newRating]);

      setIsChangingPoem(true);
      await delay(300);

      const newRatedPoemIds = [...ratedPoemIds, currentPoem.id];
      const newRatedSet = new Set(newRatedPoemIds);

      setRatedPoemIds(newRatedPoemIds);

      const nextIndex = poems.findIndex(
        (poem, index) => index > currentIndex && !newRatedSet.has(poem.id)
      );

      setCurrentIndex(nextIndex === -1 ? poems.length : nextIndex);
      resetForm();

      window.scrollTo({
        top: 0,
        behavior: "smooth",
      });

      requestAnimationFrame(() => {
        setIsChangingPoem(false);
      });
    } catch (error) {
      console.error(error);
      alert("Could not save your rating. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  }

  if (!hasStarted) {
    return (
      <div className="mx-auto max-w-xl rounded-3xl border bg-card p-8 shadow-sm">
        <h1 className="text-2xl font-semibold">Start evaluation</h1>

        <p className="mt-2 text-muted-foreground">Please choose your evaluator ID to continue.</p>

        <div className="mt-6 space-y-4">
          <select
            value={evaluatorId}
            onChange={(event) => setEvaluatorId(event.target.value)}
            className="h-11 w-full rounded-xl border bg-background px-4"
          >
            <option value="">Select evaluator ID</option>
            <option value="1">Evaluator 1</option>
            <option value="2">Evaluator 2</option>
          </select>

          <Button
            className="h-11 w-full rounded-xl"
            onClick={handleStart}
            disabled={isLoadingSession || !evaluatorId.trim()}
          >
            {isLoadingSession ? "Loading..." : "Continue"}
          </Button>
        </div>
      </div>
    );
  }

  if (isFinished) {
    return (
      <div className="rounded-3xl border border-border bg-card p-8 shadow-sm">
        <h2 className="text-2xl font-semibold">Evaluation complete</h2>

        <p className="mt-2 text-muted-foreground">You rated {ratings.length} poems.</p>

        <pre className="mt-6 max-h-96 overflow-auto rounded-2xl bg-muted p-4 text-sm">
          {JSON.stringify(ratings, null, 2)}
        </pre>
      </div>
    );
  }

  if (!currentPoem) {
    return null;
  }

  return (
    <div
      key={currentPoem.id}
      className={`grid gap-6 transition-all duration-300 lg:grid-cols-[minmax(0,1fr)_460px] ${
        isChangingPoem ? "translate-y-3 opacity-0" : "translate-y-0 opacity-100"
      }`}
    >
      <section className="overflow-hidden rounded-3xl border bg-card shadow-sm">
        <div className="border-b bg-muted/30 px-6 py-5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-xs font-medium uppercase tracking-[0.2em] text-muted-foreground">
                Anonymous poem
              </p>
              <h2 className="mt-1 text-2xl font-semibold">
                Poem {currentIndex + 1}/{poems.length}
              </h2>
            </div>
          </div>
        </div>

        <div className="space-y-5 p-6">
          <div className="rounded-2xl border bg-background p-4">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Topic
            </p>
            <p className="mt-1 text-lg font-semibold">{currentPoem.topic}</p>
          </div>

          <div className="rounded-2xl border bg-background p-6">
            <div className="mx-auto max-w-3xl whitespace-pre-line text-lg leading-9">
              {currentPoem.text}
            </div>
          </div>
        </div>
      </section>

      <aside className="rounded-3xl border bg-card p-6 shadow-sm lg:sticky lg:top-6 lg:self-start">
        <div className="mb-6">
          <p className="text-xs font-medium uppercase tracking-[0.2em] text-muted-foreground">
            Evaluation
          </p>
          <h3 className="mt-1 text-2xl font-semibold">Rating</h3>
        </div>

        <div className="space-y-6">
          <LikertRow
            label="How fluent and understandable is the poem?"
            value={fluency}
            onChange={setFluency}
            left="Not fluent"
            right="Very fluent"
          />

          <LikertRow
            label="How well does the poem fit the given topic?"
            value={themeAlignment}
            onChange={setThemeAlignment}
            left="Does not fit"
            right="Fits very well"
          />

          <LikertRow
            label="How meaningful is the poem?"
            value={meaningfulness}
            onChange={setMeaningfulness}
            left="Not meaningful"
            right="Very meaningful"
          />

          <LikertRow
            label="How poetic or aesthetically interesting is the poem?"
            value={poeticness}
            onChange={setPoeticness}
            left="Not poetic"
            right="Very poetic"
          />

          <LikertRow
            label="Overall, how good is the poem?"
            value={overallQuality}
            onChange={setOverallQuality}
            left="Very poor"
            right="Excellent"
          />

          <div className="space-y-2">
            <label className="text-sm font-medium">Comment</label>
            <Textarea
              value={comment}
              onChange={(event) => setComment(event.target.value)}
              placeholder="Optional"
              className="min-h-24 resize-none"
            />
          </div>

          <Button
            className="h-11 w-full rounded-xl"
            onClick={handleSubmit}
            disabled={!canSubmit || isSubmitting}
          >
            {isSubmitting ? (isChangingPoem ? "Loading next poem..." : "Submitting...") : "Submit"}
          </Button>
        </div>
      </aside>
    </div>
  );
}

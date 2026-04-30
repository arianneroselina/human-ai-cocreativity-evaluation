"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/shadcn_ui/button";
import { Textarea } from "@/components/shadcn_ui/textarea";
import LikertRow, { type Likert } from "./likertRow";
import { dummyPoems, type DummyPoem } from "./dummyPoems";

type RatingResult = {
  poemId: string;
  clarity: Likert;
  creativity: Likert;
  relevance: Likert;
  overallQuality: Likert;
  comment: string;
};

function shuffleArray<T>(array: T[]) {
  return [...array].sort(() => Math.random() - 0.5);
}

export default function EvaluationWorkbench() {
  const [randomizedPoems, setRandomizedPoems] = useState<DummyPoem[]>([]);
  const [hasStarted, setHasStarted] = useState(false);

  const [currentIndex, setCurrentIndex] = useState(0);
  const [ratings, setRatings] = useState<RatingResult[]>([]);

  const [clarity, setClarity] = useState<Likert | null>(null);
  const [creativity, setCreativity] = useState<Likert | null>(null);
  const [relevance, setRelevance] = useState<Likert | null>(null);
  const [overallQuality, setOverallQuality] = useState<Likert | null>(null);
  const [comment, setComment] = useState("");

  useEffect(() => {
    setRandomizedPoems(shuffleArray(dummyPoems));
    setHasStarted(true);
  }, []);

  const currentPoem = randomizedPoems[currentIndex];
  const isFinished = hasStarted && currentIndex >= randomizedPoems.length;
  const canSubmit = clarity && creativity && relevance && overallQuality;

  function resetForm() {
    setClarity(null);
    setCreativity(null);
    setRelevance(null);
    setOverallQuality(null);
    setComment("");
  }

  function handleSubmit() {
    if (!currentPoem) return;

    if (!clarity || !creativity || !relevance || !overallQuality) {
      alert("Please rate all four metrics before continuing.");
      return;
    }

    const newRating: RatingResult = {
      poemId: currentPoem.id,
      clarity,
      creativity,
      relevance,
      overallQuality,
      comment,
    };

    setRatings((previous) => [...previous, newRating]);
    setCurrentIndex((previous) => previous + 1);
    resetForm();
  }

  if (!hasStarted) {
    return (
      <div className="rounded-3xl border border-border bg-card p-8 shadow-sm">
        <p className="text-muted-foreground">Preparing randomized poems...</p>
      </div>
    );
  }

  if (isFinished) {
    return (
      <div className="rounded-3xl border border-border bg-card p-8 shadow-sm">
        <h2 className="text-2xl font-semibold">Evaluation complete</h2>

        <p className="mt-2 text-muted-foreground">
          You rated {ratings.length} poems.
        </p>

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
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_460px]">
      <section className="overflow-hidden rounded-3xl border bg-card shadow-sm">
        <div className="border-b bg-muted/30 px-6 py-5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-xs font-medium uppercase tracking-[0.2em] text-muted-foreground">
                Anonymous poem
              </p>
              <h2 className="mt-1 text-2xl font-semibold">
                Poem {currentIndex + 1}/{randomizedPoems.length}
              </h2>
            </div>

            <div className="rounded-full border bg-background px-3 py-1 text-xs text-muted-foreground">
              Source hidden
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
            label="How clear and understandable is the poem?"
            value={clarity}
            onChange={setClarity}
            left="Not clear"
            right="Very clear"
          />

          <LikertRow
            label="How creative or original is the poem?"
            value={creativity}
            onChange={setCreativity}
            left="Not creative"
            right="Very creative"
          />

          <LikertRow
            label="How well does the poem fit the given topic?"
            value={relevance}
            onChange={setRelevance}
            left="Not relevant"
            right="Very relevant"
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
            disabled={!canSubmit}
          >
            Submit
          </Button>
        </div>
      </aside>
    </div>
  );
}

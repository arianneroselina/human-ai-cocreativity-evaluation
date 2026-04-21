"use client";

import { useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, Equal, Trophy } from "lucide-react";
import { evaluationPairs, evaluationSummary } from "@/data/mockEvaluation";
import { Button } from "@/components/shadcn_ui/button";
import ComparisonCard from "@/components/evaluation/comparisonCard";
import TopicSidebar from "@/components/evaluation/topicSidebar";

export default function EvaluationWorkbench() {
  const [selectedTopicId, setSelectedTopicId] = useState(evaluationPairs[0].topicId);
  const [selectedPairIndex, setSelectedPairIndex] = useState(0);

  const topicPairs = useMemo(
    () => evaluationPairs.filter((pair) => pair.topicId === selectedTopicId),
    [selectedTopicId]
  );

  const activePair = topicPairs[Math.min(selectedPairIndex, topicPairs.length - 1)] ?? topicPairs[0];

  const goToPreviousPair = () => setSelectedPairIndex((current) => Math.max(current - 1, 0));
  const goToNextPair = () =>
    setSelectedPairIndex((current) => Math.min(current + 1, topicPairs.length - 1));

  const chooseTopic = (topicId: string) => {
    setSelectedTopicId(topicId);
    setSelectedPairIndex(0);
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[280px_minmax(0,1fr)]">
      <TopicSidebar selectedTopicId={selectedTopicId} onSelectTopic={chooseTopic} />

      <section className="space-y-6">
        <div className="rounded-2xl border border-border bg-card p-5 shadow-sm">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="text-xs font-medium uppercase tracking-[0.2em] text-muted-foreground">
                Dummy evaluation workspace
              </p>
              <h2 className="mt-2 text-2xl font-semibold">Anonymous poem comparison</h2>
              <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                This first step is UI-only. The platform shows two poems from the same topic side by
                side, hides who created them, and lets you shape the evaluator experience before any
                real study logic is added.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-xl border border-border bg-background px-3 py-2">
                <p className="text-xs text-muted-foreground">Participants</p>
                <p className="text-lg font-semibold">{evaluationSummary.participants}</p>
              </div>
              <div className="rounded-xl border border-border bg-background px-3 py-2">
                <p className="text-xs text-muted-foreground">Topics</p>
                <p className="text-lg font-semibold">{evaluationSummary.topics}</p>
              </div>
              <div className="rounded-xl border border-border bg-background px-3 py-2">
                <p className="text-xs text-muted-foreground">Poems</p>
                <p className="text-lg font-semibold">{evaluationSummary.totalPoems}</p>
              </div>
              <div className="rounded-xl border border-border bg-background px-3 py-2">
                <p className="text-xs text-muted-foreground">Visible pairs</p>
                <p className="text-lg font-semibold">{evaluationSummary.visiblePairs}</p>
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-card p-5 shadow-sm">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Current topic</p>
              <h3 className="text-xl font-semibold">{activePair?.topicLabel}</h3>
              <p className="mt-1 text-sm text-muted-foreground">
                Pair {selectedPairIndex + 1} of {topicPairs.length}
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button variant="outline" onClick={goToPreviousPair} disabled={selectedPairIndex === 0}>
                <ChevronLeft className="h-4 w-4" /> Previous
              </Button>
              <Button
                variant="outline"
                onClick={goToNextPair}
                disabled={selectedPairIndex >= topicPairs.length - 1}
              >
                Next <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <div className="mt-5 grid gap-4 rounded-xl border border-dashed border-border bg-background/60 p-4 md:grid-cols-3">
            <button className="rounded-xl border border-border bg-card px-4 py-3 text-left hover:bg-accent">
              <div className="flex items-center gap-2 font-medium">
                <Trophy className="h-4 w-4" /> Text A is better
              </div>
              <p className="mt-1 text-sm text-muted-foreground">
                Placeholder button for later evaluator submission.
              </p>
            </button>
            <button className="rounded-xl border border-border bg-card px-4 py-3 text-left hover:bg-accent">
              <div className="flex items-center gap-2 font-medium">
                <Equal className="h-4 w-4" /> Tie / equally strong
              </div>
              <p className="mt-1 text-sm text-muted-foreground">Useful when both texts feel comparable.</p>
            </button>
            <button className="rounded-xl border border-border bg-card px-4 py-3 text-left hover:bg-accent">
              <div className="flex items-center gap-2 font-medium">
                <Trophy className="h-4 w-4" /> Text B is better
              </div>
              <p className="mt-1 text-sm text-muted-foreground">
                Kept symmetrical so the UI stays unbiased.
              </p>
            </button>
          </div>
        </div>

        {activePair && (
          <div className="grid gap-6 xl:grid-cols-2">
            <ComparisonCard label="Text A" poem={activePair.left} />
            <ComparisonCard label="Text B" poem={activePair.right} />
          </div>
        )}
      </section>
    </div>
  );
}

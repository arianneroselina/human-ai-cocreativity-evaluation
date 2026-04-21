import Link from "next/link";
import { ArrowRight, EyeOff, GitCompareArrows, Shapes } from "lucide-react";
import { Button } from "@/components/shadcn_ui/button";
import EvaluationWorkbench from "@/components/evaluation/evaluationWorkbench";

export default function Page() {
  return (
    <div className="mx-auto max-w-7xl space-y-8 px-4 py-8 sm:px-6 lg:px-8">
      <section className="rounded-3xl border border-border bg-card p-8 shadow-sm">
        <div className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr] lg:items-center">
          <div>
            <p className="text-sm font-medium uppercase tracking-[0.2em] text-muted-foreground">
              Human-AI Co-Creativity Evaluation
            </p>
            <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">
              Prepare the poem evaluation platform UI with dummy data first
            </h1>
            <p className="mt-4 max-w-3xl text-base leading-7 text-muted-foreground">
              This repo is now shaped around the evaluation side of your study: anonymous pairwise
              comparison of poems from the same topic. No real participant flow yet, no submission
              backend yet, just a clean interface you can iterate on.
            </p>

            <div className="mt-6 flex flex-wrap gap-3">
              <Button asChild>
                <Link href="#workbench">
                  Open workbench <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>
          </div>

          <div className="grid gap-3">
            <div className="rounded-2xl border border-border bg-background p-4">
              <div className="flex items-center gap-2 font-medium">
                <GitCompareArrows className="h-4 w-4" /> Pairwise comparison
              </div>
              <p className="mt-2 text-sm text-muted-foreground">
                Each screen compares two poems with the same topic.
              </p>
            </div>
            <div className="rounded-2xl border border-border bg-background p-4">
              <div className="flex items-center gap-2 font-medium">
                <EyeOff className="h-4 w-4" /> Anonymous presentation
              </div>
              <p className="mt-2 text-sm text-muted-foreground">
                Creation source and participant identity stay hidden from evaluators.
              </p>
            </div>
            <div className="rounded-2xl border border-border bg-background p-4">
              <div className="flex items-center gap-2 font-medium">
                <Shapes className="h-4 w-4" /> UI-first foundation
              </div>
              <p className="mt-2 text-sm text-muted-foreground">
                Dummy poems, dummy pairings, and placeholder actions make iteration fast.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section id="workbench">
        <EvaluationWorkbench />
      </section>
    </div>
  );
}

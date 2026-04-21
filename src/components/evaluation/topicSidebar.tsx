"use client";

import { topicOverview } from "@/data/mockEvaluation";
import { cn } from "@/lib/utils";

type Props = {
  selectedTopicId: string;
  onSelectTopic: (topicId: string) => void;
};

export default function TopicSidebar({ selectedTopicId, onSelectTopic }: Props) {
  return (
    <aside className="rounded-2xl border border-border bg-card p-4 shadow-sm">
      <div>
        <p className="text-xs font-medium uppercase tracking-[0.2em] text-muted-foreground">
          Topics
        </p>
        <h2 className="mt-2 text-lg font-semibold">Evaluation set</h2>
      </div>

      <div className="mt-4 space-y-2">
        {topicOverview.map((topic, index) => {
          const isSelected = selectedTopicId === topic.id;
          return (
            <button
              key={topic.id}
              type="button"
              onClick={() => onSelectTopic(topic.id)}
              className={cn(
                "w-full rounded-xl border px-3 py-3 text-left transition",
                isSelected
                  ? "border-primary bg-primary/5"
                  : "border-border bg-background hover:bg-accent"
              )}
            >
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-xs text-muted-foreground">Topic {index + 1}</p>
                  <p className="font-medium">{topic.label}</p>
                </div>
                <div className="text-right text-xs text-muted-foreground">
                  <p>{topic.poemCount} poems</p>
                  <p>{topic.pairCount} pairs</p>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </aside>
  );
}

import type { PoemEntry } from "@/data/mockEvaluation";

type Props = {
  label: "Text A" | "Text B";
  poem: PoemEntry;
};

export default function ComparisonCard({ label, poem }: Props) {
  return (
    <article className="flex h-full flex-col rounded-2xl border border-border bg-card shadow-sm">
      <div className="border-b border-border p-4">
        <p className="text-xs font-medium uppercase tracking-[0.2em] text-muted-foreground">
          {label}
        </p>
        <h3 className="mt-2 text-xl font-semibold">{poem.title}</h3>
        <p className="mt-1 text-sm text-muted-foreground">
          Topic: {poem.topicLabel} · Author information hidden
        </p>
      </div>

      <div className="flex-1 p-4">
        <div className="whitespace-pre-line text-sm leading-7 text-foreground/90">{poem.text}</div>
      </div>
    </article>
  );
}

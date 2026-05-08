import type { WorkflowFeedbackSummary } from "@/lib/research-dashboard/feedbackSummaries";

export default function FeedbackSummaryCards({
                                               summaries,
                                             }: {
  summaries: WorkflowFeedbackSummary[];
}) {
  if (summaries.length === 0) {
    return (
      <div className="rounded-xl border border-yellow-200 bg-yellow-50 p-4 text-sm text-yellow-900">
        No feedback summaries found. Run{" "}
        <code className="font-mono">make process-data</code> with{" "}
        <code className="font-mono">OPENAI_API_KEY</code> set.
      </div>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      {summaries.map((item) => (
        <article
          key={item.workflow}
          className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm"
        >
          <div className="mb-3 flex items-start justify-between gap-4">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">
                {item.workflowLabel}
              </h3>

              <p className="mt-1 font-mono text-xs text-gray-400">
                {item.workflow}
              </p>
            </div>

            <span className="shrink-0 rounded-full bg-gray-100 px-3 py-1 text-xs text-gray-700">
              {item.commentCount} comments
            </span>
          </div>

          <p className="text-sm leading-6 text-gray-700">{item.summary}</p>
        </article>
      ))}
    </div>
  );
}

import fs from "node:fs";
import path from "node:path";

export type WorkflowFeedbackSummary = {
  workflow: string;
  workflowLabel: string;
  commentCount: number;
  summary: string;
};

const SUMMARY_PATH = path.join(
  process.cwd(),
  "data/processed/dashboard_tables/workflow_feedback_summaries.json",
);

export function getWorkflowFeedbackSummaries(): WorkflowFeedbackSummary[] {
  if (!fs.existsSync(SUMMARY_PATH)) {
    return [];
  }

  try {
    const raw = fs.readFileSync(SUMMARY_PATH, "utf-8");
    const parsed = JSON.parse(raw);

    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed.map((item) => ({
      workflow: String(item.workflow ?? "unknown"),
      workflowLabel: String(item.workflowLabel ?? item.workflow ?? "Unknown"),
      commentCount: Number(item.commentCount ?? 0),
      summary: String(item.summary ?? ""),
    }));
  } catch {
    return [];
  }
}

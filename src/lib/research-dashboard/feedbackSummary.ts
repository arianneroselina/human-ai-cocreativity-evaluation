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
  "data/runtime/workflow_feedback_summaries.csv",
);

function parseCsvLine(line: string): string[] {
  const values: string[] = [];
  let current = "";
  let insideQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const nextChar = line[i + 1];

    if (char === '"' && insideQuotes && nextChar === '"') {
      current += '"';
      i += 1;
      continue;
    }

    if (char === '"') {
      insideQuotes = !insideQuotes;
      continue;
    }

    if (char === "," && !insideQuotes) {
      values.push(current);
      current = "";
      continue;
    }

    current += char;
  }

  values.push(current);
  return values;
}

function parseCsv(content: string): Record<string, string>[] {
  const lines = content
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0);

  if (lines.length === 0) {
    return [];
  }

  const headers = parseCsvLine(lines[0]);

  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    const row: Record<string, string> = {};

    headers.forEach((header, index) => {
      row[header] = values[index] ?? "";
    });

    return row;
  });
}

export function getWorkflowFeedbackSummaries(): WorkflowFeedbackSummary[] {
  if (!fs.existsSync(SUMMARY_PATH)) {
    return [];
  }

  try {
    const raw = fs.readFileSync(SUMMARY_PATH, "utf-8");
    const rows = parseCsv(raw);

    return rows.map((row) => ({
      workflow: row.workflow || "unknown",
      workflowLabel: row.workflowLabel || row.workflow || "Unknown",
      commentCount: Number(row.commentCount || 0),
      summary: row.summary || "",
    }));
  } catch {
    return [];
  }
}

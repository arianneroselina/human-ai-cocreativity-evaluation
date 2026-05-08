import fs from "fs";
import path from "path";
import Papa from "papaparse";
import { getPoemTaskById } from "@/data/tasks";
import { parseNumber } from "@/lib/utils";
import { Poem } from "@/lib/evaluation/types";

function parseDate(value: unknown): Date | null {
  if (!value || typeof value !== "string") return null;

  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function parseBoolean(value: unknown): boolean {
  return value === true || value === "t" || value === "true" || value === "1";
}

function loadParticipantId(folderPath: string): number | null {
  const sessionPath = path.join(folderPath, "Session.csv");

  if (!fs.existsSync(sessionPath)) return null;

  const csvContent = fs.readFileSync(sessionPath, "utf-8");
  const parsed = Papa.parse(csvContent, {
    header: true,
    skipEmptyLines: true,
  });

  const firstRow = parsed.data[0] as any;

  return parseNumber(firstRow?.participantId);
}

export function loadPoems(): Poem[] {
  const inputsDir = path.join(process.cwd(), "inputs");
  const poems: Poem[] = [];

  // Get all subfolders in inputs/
  const subfolders = fs
    .readdirSync(inputsDir, { withFileTypes: true })
    .filter((dirent) => dirent.isDirectory())
    .map((dirent) => dirent.name);

  subfolders.forEach((folder) => {
    const folderPath = path.join(inputsDir, folder);
    const csvPath = path.join(folderPath, "Round.csv");

    if (!fs.existsSync(csvPath)) return;

    const participantId = loadParticipantId(folderPath);

    const csvContent = fs.readFileSync(csvPath, "utf-8");
    const parsed = Papa.parse(csvContent, {
      header: true,
      skipEmptyLines: true,
    });

    parsed.data.forEach((row: any) => {
      if (!row.id || !row.workflow || !row.taskId) return;

      const roundIndex = parseNumber(row.index);

      poems.push({
        id: row.id,

        sessionId: row.sessionId ?? null,
        participantId,
        roundIndex,

        taskId: row.taskId,
        topic: getPoemTaskById(row.taskId).title,
        text: row.text?.trim() ?? "",
        workflow: row.workflow as "human" | "ai" | "human_ai" | "ai_human",

        isEmpty: !row.text || row.text.trim() === "",
        timeMs: parseNumber(row.timeMs),
        wordCount: parseNumber(row.wordCount),
        charCount: parseNumber(row.charCount),
        passed: parseBoolean(row.passed),

        startedAt: parseDate(row.startedAt),
        submittedAt: parseDate(row.submittedAt),
      });
    });
  });

  return poems;
}

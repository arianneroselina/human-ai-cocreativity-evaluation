import fs from "fs";
import path from "path";
import Papa from "papaparse";
import { getPoemTaskById } from "@/data/tasks";
import { parseNumber } from "@/lib/utils";

export type Poem = {
  id: string;
  taskId: string;
  topic: string;
  text: string;
  workflow: string;
  timeMs: number | null;
  wordCount: number | null;
  charCount: number | null;
  passed: boolean;
};

export function loadPoems(): Poem[] {
  const inputsDir = path.join(process.cwd(), "inputs");
  const poems: Poem[] = [];

  // Get all subfolders in inputs/
  const subfolders = fs
    .readdirSync(inputsDir, { withFileTypes: true })
    .filter((dirent) => dirent.isDirectory())
    .map((dirent) => dirent.name);

  subfolders.forEach((folder) => {
    const csvPath = path.join(inputsDir, folder, "Round.csv");
    if (fs.existsSync(csvPath)) {
      const csvContent = fs.readFileSync(csvPath, "utf-8");
      const parsed = Papa.parse(csvContent, { header: true, skipEmptyLines: true });

      parsed.data.forEach((row: any) => {
        if (row.workflow && row.taskId) {
          poems.push({
            id: row.id,
            taskId: row.taskId,
            topic: getPoemTaskById(row.taskId).title,
            text: row.text.trim(),
            workflow: row.workflow as "human" | "ai" | "human_ai" | "ai_human",
            timeMs: parseNumber(row.timeMs),
            wordCount: parseNumber(row.wordCount),
            charCount: parseNumber(row.charCount),
            passed: row.passed === "t" || row.passed === "true",
          });
        }
      });
    }
  });

  return poems;
}

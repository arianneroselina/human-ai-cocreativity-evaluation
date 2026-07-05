import "server-only";

import fs from "fs";
import path from "path";

export type AnalysisFile = {
  name: string;
  url: string;
  sizeKb: number;
};

const ANALYSIS_DIR = path.join(process.cwd(), "public", "research-dashboard", "analysis");

export function getAnalysisFiles(): AnalysisFile[] {
  if (!fs.existsSync(ANALYSIS_DIR)) {
    return [];
  }

  return fs
    .readdirSync(ANALYSIS_DIR, { withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => {
      const filePath = path.join(ANALYSIS_DIR, entry.name);
      const stats = fs.statSync(filePath);

      return {
        name: entry.name,
        url: `/research-dashboard/analysis/${entry.name}`,
        sizeKb: Math.round((stats.size / 1024) * 10) / 10,
      };
    })
    .sort((a, b) => a.name.localeCompare(b.name));
}

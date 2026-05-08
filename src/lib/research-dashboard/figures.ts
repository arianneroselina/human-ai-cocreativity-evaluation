import "server-only";

import fs from "fs";
import path from "path";

export type DashboardFigure = {
  slug: string;
  title: string;
  description: string;
  pngUrl: string;
  pdfUrl: string;
  svgUrl: string;
};

const MANIFEST_PATH = path.join(
  process.cwd(),
  "public",
  "research-dashboard",
  "figures",
  "manifest.json"
);

export function getDashboardFigures(): DashboardFigure[] {
  if (!fs.existsSync(MANIFEST_PATH)) {
    return [];
  }

  const content = fs.readFileSync(MANIFEST_PATH, "utf-8");
  return JSON.parse(content) as DashboardFigure[];
}

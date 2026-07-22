"use client";

import { useMemo } from "react";
import type { DashboardFigure } from "@/lib/research-dashboard/figures";
import FigureGallery from "./FigureGallery";
import FoldableSection from "@/app/research-dashboard/components/FoldableSection";

export type FigureGroup = {
  id: string;
  title: string;
  description: string;
  figures: DashboardFigure[];
};

type FigureDashboardProps = {
  figureGroups: FigureGroup[];
  favoriteNumbers: string[];
  interestingNumbers: string[];
};

export default function FigureDashboard({
  figureGroups,
  favoriteNumbers,
  interestingNumbers,
}: FigureDashboardProps) {
  return (
    <div className="space-y-8 pt-5">
      {figureGroups.map((group, index) => (
        <FoldableSection
          key={group.id}
          id={group.id}
          title={group.title}
          badge={`${group.figures.length} figures`}
          description={group.description}
          defaultOpen={index < 2}
        >
          <FigureGallery
            figures={group.figures}
            favoriteNumbers={favoriteNumbers}
            interestingNumbers={interestingNumbers}
          />
        </FoldableSection>
      ))}
    </div>
  );
}

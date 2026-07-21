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

function matchesFigureNumber(figure: DashboardFigure, numbers: string[]) {
  return numbers.some((number) => figure.slug.startsWith(number));
}

export default function FigureDashboard({
  figureGroups,
  favoriteNumbers,
  interestingNumbers,
}: FigureDashboardProps) {
  const allFigures = useMemo(() => {
    const figuresBySlug = new Map<string, DashboardFigure>();

    for (const group of figureGroups) {
      for (const figure of group.figures) {
        figuresBySlug.set(figure.slug, figure);
      }
    }

    return Array.from(figuresBySlug.values());
  }, [figureGroups]);

  const favoriteFigures = allFigures.filter((figure) =>
    matchesFigureNumber(figure, favoriteNumbers)
  );

  const interestingFigures = allFigures.filter(
    (figure) =>
      matchesFigureNumber(figure, interestingNumbers) &&
      !matchesFigureNumber(figure, favoriteNumbers)
  );

  return (
    <div className="space-y-8">
      <FoldableSection
        id="favorite-figures"
        title="Favorite Figures"
        badge={`${favoriteFigures.length} figures`}
        description="The most relevant figures selected for the thesis."
        defaultOpen
      >
        {favoriteFigures.length > 0 ? (
          <FigureGallery
            figures={favoriteFigures}
            favoriteNumbers={favoriteNumbers}
            interestingNumbers={interestingNumbers}
          />
        ) : (
          <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50 px-5 py-8 text-center text-sm text-gray-500">
            No favorite figures configured.
          </div>
        )}
      </FoldableSection>

      <FoldableSection
        id="interesting-figures"
        title="Potentially Interesting Figures"
        badge={`${interestingFigures.length} figures`}
        description="Additional figures that may be useful depending on the thesis focus."
        defaultOpen
      >
        {interestingFigures.length > 0 ? (
          <FigureGallery
            figures={interestingFigures}
            favoriteNumbers={favoriteNumbers}
            interestingNumbers={interestingNumbers}
          />
        ) : (
          <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50 px-5 py-8 text-center text-sm text-gray-500">
            No potentially interesting figures configured.
          </div>
        )}
      </FoldableSection>

      <div className="border-t border-gray-300 pt-5">
        <div className="p-2 mb-5">
          <h2 className="text-xl font-semibold text-gray-900">
            All Figures
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Complete collection grouped by analysis category.
          </p>
        </div>

        <div className="space-y-8">
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
      </div>
    </div>
  );
}

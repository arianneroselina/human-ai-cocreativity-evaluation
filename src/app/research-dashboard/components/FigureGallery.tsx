"use client";

import { useEffect, useRef, useState } from "react";
import type { DashboardFigure } from "@/lib/research-dashboard/figures";
import { ExpandIcon, DownloadIcon, StarIcon, CircleQuestionMarkIcon } from "lucide-react";

type FigureGalleryProps = {
  figures: DashboardFigure[];
  favoriteNumbers?: string[];
  interestingNumbers?: string[];
};

function DownloadMenu({ figure }: { figure: DashboardFigure }) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }

    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleEscape);

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, []);

  return (
    <div ref={menuRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        className="flex h-8 w-8 items-center justify-center rounded-md border bg-white text-gray-600 hover:bg-gray-50"
        title="Download"
        aria-label={`Download ${figure.title}`}
      >
        <DownloadIcon />
      </button>

      {open && (
        <div className="absolute right-0 z-20 mt-2 w-36 overflow-hidden rounded-lg border border-gray-200 bg-white text-sm shadow-lg">
          <a
            href={figure.pngUrl}
            download
            onClick={() => setOpen(false)}
            className="block px-3 py-2 hover:bg-gray-50"
          >
            PNG
          </a>

          <a
            href={figure.pdfUrl}
            download
            onClick={() => setOpen(false)}
            className="block px-3 py-2 hover:bg-gray-50"
          >
            PDF
          </a>
        </div>
      )}
    </div>
  );
}

export default function FigureGallery({
  figures,
  favoriteNumbers = [],
  interestingNumbers = [],
}: FigureGalleryProps) {
  const [selectedFigure, setSelectedFigure] = useState<DashboardFigure | null>(null);

  useEffect(() => {
    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setSelectedFigure(null);
      }
    }

    window.addEventListener("keydown", handleEscape);

    return () => {
      window.removeEventListener("keydown", handleEscape);
    };
  }, []);

  function isFavorite(figure: DashboardFigure) {
    return favoriteNumbers.some((number) => figure.slug.startsWith(number));
  }

  function isInteresting(figure: DashboardFigure) {
    return interestingNumbers.some((number) => figure.slug.startsWith(number));
  }

  if (figures.length === 0) {
    return (
      <div className="rounded-xl border border-yellow-200 bg-yellow-50 p-4 text-sm text-yellow-900">
        No generated figures found. Run <code className="font-mono">make generate-figures</code> or{" "}
        <code className="font-mono">make process-data</code>.
      </div>
    );
  }

  return (
    <>
      <div className="grid grid-cols-[repeat(auto-fit,minmax(380px,1fr))] gap-6">
        {figures.map((figure) => {
          const favorite = isFavorite(figure);
          const interesting = isInteresting(figure);

          return (
            <section
              key={figure.slug}
              className="flex flex-col rounded-2xl border border-gray-200 bg-white p-5 shadow-sm"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <h3 className="text-lg font-semibold text-gray-900">{figure.title}</h3>

                  <p className="mt-1 text-sm leading-relaxed text-gray-500">{figure.description}</p>
                </div>

                <div className="flex shrink-0 items-center gap-2">
                  {favorite && (
                    <div
                      className="flex h-8 w-8 items-center justify-center rounded-md border border-yellow-300 bg-yellow-50 text-yellow-500"
                      title="Favorite figure"
                      aria-label="Favorite figure"
                    >
                      <StarIcon />
                    </div>
                  )}
                  {interesting && (
                    <div
                      className="flex h-8 w-8 items-center justify-center rounded-md border border-blue-300 bg-blue-50 text-blue-600"
                      title="Potentially interesting figure"
                      aria-label="Potentially interesting figure"
                    >
                      <CircleQuestionMarkIcon />
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={() => setSelectedFigure(figure)}
                    className="flex h-8 w-8 items-center justify-center rounded-md border bg-white text-gray-600 hover:bg-gray-50"
                    title="Open fullscreen"
                    aria-label={`Open ${figure.title} fullscreen`}
                  >
                    <ExpandIcon />
                  </button>

                  <DownloadMenu figure={figure} />
                </div>
              </div>

              <button
                type="button"
                onClick={() => setSelectedFigure(figure)}
                className="mt-4 flex min-h-[320px] flex-1 items-center justify-center overflow-hidden rounded-xl border bg-gray-50 p-3 hover:bg-gray-100"
              >
                <img
                  src={figure.pngUrl}
                  alt={figure.title}
                  className="max-h-[430px] w-full object-contain"
                />
              </button>
            </section>
          );
        })}
      </div>

      {selectedFigure && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
          onClick={() => setSelectedFigure(null)}
          role="dialog"
          aria-modal="true"
          aria-label={selectedFigure.title}
        >
          <div
            className="flex max-h-[95vh] w-full max-w-7xl flex-col rounded-2xl bg-white p-4 shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="mb-3 flex items-start justify-between gap-4">
              <div className="min-w-0">
                <h2 className="text-xl font-semibold text-gray-900">{selectedFigure.title}</h2>

                <p className="mt-1 text-sm text-gray-500">{selectedFigure.description}</p>
              </div>

              <div className="flex shrink-0 items-center gap-2">
                {isFavorite(selectedFigure) && (
                  <div
                    className="flex h-8 w-8 items-center justify-center rounded-md border border-yellow-300 bg-yellow-50 text-yellow-500"
                    title="Favorite figure"
                    aria-label="Favorite figure"
                  >
                    <StarIcon />
                  </div>
                )}
                {isInteresting(selectedFigure) && (
                  <div
                    className="flex h-8 w-8 items-center justify-center rounded-md border border-blue-300 bg-blue-50 text-blue-600"
                    title="Potentially interesting figure"
                    aria-label="Potentially interesting figure"
                  >
                    <CircleQuestionMarkIcon />
                  </div>
                )}

                <DownloadMenu figure={selectedFigure} />

                <button
                  type="button"
                  onClick={() => setSelectedFigure(null)}
                  className="rounded-lg border px-3 py-1.5 text-sm hover:bg-gray-50"
                >
                  Close
                </button>
              </div>
            </div>

            <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto rounded-xl border bg-gray-50 p-4">
              <img
                src={selectedFigure.pngUrl}
                alt={selectedFigure.title}
                className="max-h-[78vh] w-auto max-w-full object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}

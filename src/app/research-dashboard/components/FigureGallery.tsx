"use client";

import { useEffect, useRef, useState } from "react";
import type { DashboardFigure } from "@/lib/research-dashboard/figures";

function DownloadIcon() {
  return (
    <svg
      aria-hidden="true"
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="M7 10l5 5 5-5" />
      <path d="M12 15V3" />
    </svg>
  );
}

function ExpandIcon() {
  return (
    <svg
      aria-hidden="true"
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M15 3h6v6" />
      <path d="M10 14 21 3" />
      <path d="M9 21H3v-6" />
      <path d="M14 10 3 21" />
    </svg>
  );
}

function DownloadMenu({ figure }: { figure: DashboardFigure }) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (!menuRef.current) {
        return;
      }

      if (!menuRef.current.contains(event.target as Node)) {
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

          <a
            href={figure.svgUrl}
            download
            onClick={() => setOpen(false)}
            className="block px-3 py-2 hover:bg-gray-50"
          >
            SVG
          </a>
        </div>
      )}
    </div>
  );
}

export default function FigureGallery({ figures }: { figures: DashboardFigure[] }) {
  const [selectedFigure, setSelectedFigure] = useState<DashboardFigure | null>(null);

  useEffect(() => {
    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setSelectedFigure(null);
      }
    }

    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, []);

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
        {figures.map((figure) => (
          <section
            key={figure.slug}
            className="flex flex-col rounded-2xl border border-gray-200 bg-white p-5 shadow-sm"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <h3 className="text-lg font-semibold text-gray-900">{figure.title}</h3>
                <p className="mt-1 text-sm leading-relaxed text-gray-500">
                  {figure.description}
                </p>
              </div>

              <div className="flex shrink-0 items-center gap-2">
                <button
                  type="button"
                  onClick={() => setSelectedFigure(figure)}
                  className="flex h-8 w-8 items-center justify-center rounded-md border bg-white text-gray-600 hover:bg-gray-50"
                  title="Open fullscreen"
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
        ))}
      </div>

      {selectedFigure && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
          onClick={() => setSelectedFigure(null)}
        >
          <div
            className="flex max-h-[95vh] w-full max-w-7xl flex-col rounded-2xl bg-white p-4 shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="mb-3 flex items-start justify-between gap-4">
              <div className="min-w-0">
                <h2 className="text-xl font-semibold text-gray-900">
                  {selectedFigure.title}
                </h2>
                <p className="mt-1 text-sm text-gray-500">
                  {selectedFigure.description}
                </p>
              </div>

              <div className="flex shrink-0 items-center gap-2">
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

import type { DashboardFigure } from "@/lib/research-dashboard/figures";

export default function FigureGallery({ figures }: { figures: DashboardFigure[] }) {
  if (figures.length === 0) {
    return (
      <div className="rounded-xl border border-yellow-200 bg-yellow-50 p-4 text-sm text-yellow-900">
        No generated figures found. Run <code className="font-mono">make generate-figures</code> or{" "}
        <code className="font-mono">make process-data</code>.
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {figures.map((figure) => (
        <section
          key={figure.slug}
          className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm"
        >
          <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">{figure.title}</h2>
              <p className="mt-1 text-sm text-gray-500">{figure.description}</p>
            </div>

            <div className="flex flex-wrap gap-2 text-sm">
              <a
                href={figure.pngUrl}
                download
                className="rounded-lg border px-3 py-2 hover:bg-gray-50"
              >
                Download PNG
              </a>
              <a
                href={figure.pdfUrl}
                download
                className="rounded-lg border px-3 py-2 hover:bg-gray-50"
              >
                Download PDF
              </a>
              <a
                href={figure.svgUrl}
                download
                className="rounded-lg border px-3 py-2 hover:bg-gray-50"
              >
                Download SVG
              </a>
            </div>
          </div>

          <div className="mt-5 overflow-x-auto rounded-xl border bg-gray-50 p-4">
            <img src={figure.pngUrl} alt={figure.title} className="mx-auto h-auto max-w-full" />
          </div>
        </section>
      ))}
    </div>
  );
}

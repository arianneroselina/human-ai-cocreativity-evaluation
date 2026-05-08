import type { ReactNode } from "react";
import { getDashboardFigures } from "@/lib/research-dashboard/figures";
import { getResearchDashboardData } from "@/lib/research-dashboard/data";
import DataTable from "./components/DataTable";
import FigureGallery from "./components/FigureGallery";
import StatCard from "./components/StatCard";

export const dynamic = "force-dynamic";

function formatPercent(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }

  return `${value.toFixed(digits)}%`;
}

function FoldableSection({
  id,
  title,
  description,
  badge,
  defaultOpen = false,
  children,
}: {
  id: string;
  title: string;
  description?: string;
  badge?: string | number;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  return (
    <details
      id={id}
      open={defaultOpen}
      className="group rounded-2xl border border-gray-200 bg-white shadow-sm"
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 p-5">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-bold text-gray-900">{title}</h2>

            {badge !== undefined && (
              <span className="rounded-full bg-gray-100 px-3 py-1 text-sm text-gray-700">
                {badge}
              </span>
            )}
          </div>

          {description && <p className="mt-1 text-sm text-gray-600">{description}</p>}
        </div>

        <span className="text-xl text-gray-400 transition group-open:rotate-90">›</span>
      </summary>

      <div className="border-t border-gray-100 p-5">{children}</div>
    </details>
  );
}

export default async function ResearchDashboardPage() {
  const data = await getResearchDashboardData();
  const figures = getDashboardFigures();

  const overviewStats = [
    { title: "Total Poems", value: data.totalPoems },
    { title: "Non-empty Poems", value: data.nonEmptyPoems },
    { title: "Empty Poems", value: data.emptyPoems },
    { title: "Evaluators", value: data.evaluatorProgress.length },
  ];

  const ratingStats = [
    {
      title: "Total Ratings",
      value: data.totalRatings,
      subtitle: `${data.expectedRatings} expected ratings`,
    },
    { title: "Completion", value: formatPercent(data.completionPercent) },
    {
      title: "Fully Rated Poems",
      value: `${data.fullyRatedPoems}/${data.nonEmptyPoems}`,
    },
    { title: "Incomplete Poems", value: data.incompletePoems.length },
  ];

  return (
    <main className="mx-auto max-w-7xl space-y-6 p-6">
      <section className="space-y-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Research Dashboard</h1>

          <p className="mt-2 max-w-3xl text-gray-600">
            Overview of evaluator progress, rating completeness, and thesis-ready figures.
          </p>
        </div>

        <nav className="flex flex-wrap gap-2 text-sm">
          <a href="#overview" className="rounded-full border px-4 py-2 hover:bg-gray-50">
            Overview
          </a>
          <a href="#progress" className="rounded-full border px-4 py-2 hover:bg-gray-50">
            Evaluator Progress
          </a>
          <a href="#incomplete" className="rounded-full border px-4 py-2 hover:bg-gray-50">
            Incomplete Poems
          </a>
          <a href="#figures" className="rounded-full border px-4 py-2 hover:bg-gray-50">
            Data Visualization
          </a>
        </nav>

        {!data.hasMasterDataset && (
          <div className="rounded-xl border border-yellow-200 bg-yellow-50 p-4 text-sm text-yellow-900">
            <strong>Missing master dataset.</strong> Run{" "}
            <code className="font-mono">make process-data</code> to generate the processed dataset
            and research figures.
          </div>
        )}
      </section>

      <section id="overview" className="space-y-4">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {overviewStats.map((stat) => (
            <StatCard key={stat.title} {...stat} />
          ))}
        </div>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {ratingStats.map((stat) => (
            <StatCard key={stat.title} {...stat} />
          ))}
        </div>
      </section>

      <FoldableSection
        id="progress"
        title="Evaluator Progress"
        badge={`${data.evaluatorProgress.length} evaluators`}
        description="Each evaluator is expected to rate all non-empty poems."
      >
        <DataTable
          rows={data.evaluatorProgress}
          columns={[
            {
              header: "Evaluator",
              render: (row) => row.evaluatorId,
            },
            {
              header: "Ratings",
              render: (row) => `${row.ratingCount}/${data.nonEmptyPoems}`,
            },
            {
              header: "Progress",
              render: (row) => formatPercent(row.progressPercent),
            },
            {
              header: "Completed",
              render: (row) => (row.completed ? "Yes" : "No"),
            },
          ]}
        />
      </FoldableSection>

      <FoldableSection
        id="incomplete"
        title="Incomplete Poems"
        badge={data.incompletePoems.length}
        description="Poems that do not yet have exactly three evaluator ratings."
      >
        <DataTable
          rows={data.incompletePoems}
          emptyText="All poems are fully rated."
          columns={[
            {
              header: "Poem ID",
              render: (row) => <span className="font-mono text-xs">{row.poemId}</span>,
            },
            {
              header: "Participant",
              render: (row) => row.participantId ?? "-",
            },
            {
              header: "Round",
              render: (row) => row.roundIndex ?? "-",
            },
            {
              header: "Task",
              render: (row) => row.taskId,
            },
            {
              header: "Workflow",
              render: (row) => row.workflow,
            },
            {
              header: "Ratings",
              render: (row) => `${row.ratingCount}/3`,
            },
          ]}
        />
      </FoldableSection>

      <FoldableSection
        id="figures"
        title="Data Visualization"
        badge={`${figures.length} figures`}
        description="Generated with Python/Matplotlib. Downloadable as PNG, PDF, or SVG."
        defaultOpen
      >
        <FigureGallery figures={figures} />
      </FoldableSection>
    </main>
  );
}

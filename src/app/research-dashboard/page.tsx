import type { ReactNode } from "react";
import { getDashboardFigures } from "@/lib/research-dashboard/figures";
import { getResearchDashboardData } from "@/lib/research-dashboard/data";
import { getWorkflowFeedbackSummaries } from "@/lib/research-dashboard/feedbackSummary";
import { getAnalysisFiles } from "@/lib/research-dashboard/analysisFiles";
import AnalysisFileList from "./components/AnalysisFileList";
import DataTable from "./components/DataTable";
import FigureGallery from "./components/FigureGallery";
import StatCard from "./components/StatCard";
import FeedbackSummaryCards from "./components/FeedbackSummaryCards";

export const dynamic = "force-dynamic";

function formatPercent(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }

  return `${value.toFixed(digits)}%`;
}

function NavLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a
      href={href}
      className="rounded-full border border-gray-200 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm transition hover:border-gray-300 hover:bg-gray-50 dark:border-slate-700 dark:bg-black dark:text-slate-300 dark:hover:border-slate-600 dark:hover:bg-slate-700"
    >
      {children}
    </a>
  );
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
      className="group scroll-mt-24 overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm dark:border-slate-700 dark:bg-black"
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 p-5 transition hover:bg-gray-50 dark:hover:bg-slate-800 [&::-webkit-details-marker]:hidden">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">{title}</h2>

            {badge !== undefined && (
              <span className="rounded-full bg-gray-100 px-3 py-1 text-sm font-medium text-gray-700">
                {badge}
              </span>
            )}
          </div>

          {description && (
            <p className="mt-1 max-w-3xl text-sm leading-6 text-gray-600 dark:text-slate-300">{description}</p>
          )}
        </div>

        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-gray-200 bg-white text-xl text-gray-500 transition group-open:rotate-90">
          ›
        </span>
      </summary>

      <div className="border-t border-gray-100 bg-white p-5 dark:border-slate-800 dark:bg-black">
        {children}
      </div>
    </details>
  );
}

function groupFigures<T extends { slug: string }>(figures: T[]) {
  return [
    {
      id: "workflow-figures",
      title: "Workflow Behavior",
      description: "Workflow choices, transitions, and final preferences.",
      figures: figures.filter((figure) =>
        ["01_", "02_", "03_", "04_", "05_", "05b_", "06_", "07_", "08_", "08b_", "09_"].some(
          (prefix) => figure.slug.startsWith(prefix)
        )
      ),
    },
    {
      id: "quality-figures",
      title: "Output Quality",
      description:
        "Quality by round and workflow, dimensions, efficiency, error-exposure interaction, and learning/fatigue effects.",
      figures: figures.filter((figure) =>
        ["11_", "12_", "13_", "14_", "15_", "16_", "17_", "17b_"].some((prefix) =>
          figure.slug.startsWith(prefix)
        )
      ),
    },
    {
      id: "evaluator-figures",
      title: "Evaluator Agreement",
      description:
        "Inter-rater agreement (ICC, Cohen's Kappa) and individual evaluator rating patterns.",
      figures: figures.filter((figure) =>
        ["61_", "62_", "63_", "64_"].some((prefix) => figure.slug.startsWith(prefix))
      ),
    },
    {
      id: "constraints-figures",
      title: "Constraint Fulfillment",
      description: "Whether submitted poems fulfilled the task constraints.",
      figures: figures.filter((figure) =>
        ["21_", "22_", "23_", "24_", "25_", "26_"].some((prefix) => figure.slug.startsWith(prefix))
      ),
    },
    {
      id: "experience-figures",
      title: "Participant Experience",
      description: "Satisfaction, frustration, AI performance ratings, and perceived task load.",
      figures: figures.filter((figure) =>
        ["31_", "32_", "33_", "34_"].some((prefix) => figure.slug.startsWith(prefix))
      ),
    },
    {
      id: "error-exposure-figures",
      title: "AI Error Exposure",
      description:
        "Round-5 error exposure, line-count error, post-error behavior, and subjective reactions.",
      figures: figures.filter((figure) =>
        ["41_", "42_", "43_", "44_"].some((prefix) => figure.slug.startsWith(prefix))
      ),
    },
    {
      id: "participant-figures",
      title: "Participant Information",
      description: "Age, gender, education, language, and AI attitude distributions.",
      figures: figures.filter((figure) =>
        ["51_", "52_", "53_", "54_", "55_", "56_"].some((prefix) => figure.slug.startsWith(prefix))
      ),
    },
  ].filter((group) => group.figures.length > 0);
}

export default async function ResearchDashboardPage() {
  const data = await getResearchDashboardData();
  const figures = getDashboardFigures();
  const figureGroups = groupFigures(figures);
  const feedbackSummaries = getWorkflowFeedbackSummaries();
  const analysisFiles = getAnalysisFiles();

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
  ];

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900 dark:bg-slate-950 dark:text-slate-100">
      <div className="mx-auto max-w-7xl space-y-6 px-6 py-6">
        <section className="rounded-3xl border border-gray-200 bg-white p-6 shadow-sm dark:border-slate-700 dark:bg-black">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500">
                Master Thesis Evaluation
              </p>

              <h1 className="text-4xl font-bold tracking-tight text-gray-950 dark:text-white">
                Research Dashboard
              </h1>

              <p className="mt-3 max-w-3xl text-base leading-7 text-gray-600 dark:text-slate-300">
                Overview of evaluator progress, rating completeness, generated figures, participant
                information, and qualitative feedback summaries.
              </p>
            </div>

            <div className="rounded-2xl bg-gray-900 px-5 py-4 text-white shadow-sm">
              <p className="text-sm text-gray-300">Completion</p>
              <p className="text-3xl font-bold">{formatPercent(data.completionPercent)}</p>
            </div>
          </div>

          {!data.hasDashboardDataset && (
            <div className="mt-5 rounded-xl border border-yellow-200 bg-yellow-50 p-4 text-sm text-yellow-900">
              <strong>Missing dataset.</strong> Run{" "}
              <code className="font-mono">make process-data</code> to generate the processed dataset
              and research figures.
            </div>
          )}
        </section>

        <nav className="sticky top-4 z-20 flex flex-wrap gap-2 rounded-2xl border border-gray-200 bg-white/90 p-3 shadow-sm backdrop-blur dark:border-slate-700 dark:bg-black/90">
          <NavLink href="#overview">Overview</NavLink>
          <NavLink href="#progress">Evaluator Progress</NavLink>
          <NavLink href="#figures">Data Visualization</NavLink>
          <NavLink href="#statistical-analysis">Statistical Analysis</NavLink>
          <NavLink href="#feedback-summaries">Feedback Summaries</NavLink>
        </nav>

        <section id="overview" className="scroll-mt-24 space-y-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Overview</h2>
            <p className="mt-1 text-sm text-gray-600 dark:text-slate-300">
              General dataset size and rating completion status.
            </p>
          </div>

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
          defaultOpen
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

        <section id="figures" className="scroll-mt-24 space-y-4">
          <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm dark:border-slate-700 dark:bg-black">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Data Visualization</h2>

            <p className="mt-2 max-w-3xl text-sm leading-6 text-gray-600 dark:text-slate-300">
              Generated with Python/Matplotlib. Figures are grouped by analysis topic and can be
              viewed fullscreen or downloaded as PNG, PDF, or SVG.
            </p>
          </div>

          {figureGroups.map((group, index) => (
            <FoldableSection
              key={group.id}
              id={group.id}
              title={group.title}
              badge={`${group.figures.length} figures`}
              description={group.description}
              defaultOpen={index < 2}
            >
              <FigureGallery figures={group.figures} />
            </FoldableSection>
          ))}
        </section>

        <FoldableSection
          id="statistical-analysis"
          title="Statistical Analysis"
          badge={`${analysisFiles.length} files`}
          description="Detailed model outputs and supplementary diagnostics, including phase-specific mixed-effects models, model cell counts, fixed effects, and poem-level evaluator disagreements."
          defaultOpen
        >
          <AnalysisFileList files={analysisFiles} />
        </FoldableSection>

        <FoldableSection
          id="feedback-summaries"
          title="Feedback Summaries"
          badge={`${feedbackSummaries.length} groups`}
          description="OpenAI-generated summaries of participant comments."
          defaultOpen
        >
          <FeedbackSummaryCards summaries={feedbackSummaries} />
        </FoldableSection>
      </div>
    </main>
  );
}

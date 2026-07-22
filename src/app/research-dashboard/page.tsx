import type { ReactNode } from "react";
import { getDashboardFigures } from "@/lib/research-dashboard/figures";
import { getResearchDashboardData } from "@/lib/research-dashboard/data";
import { getWorkflowFeedbackSummaries } from "@/lib/research-dashboard/feedbackSummary";
import { getAnalysisFiles } from "@/lib/research-dashboard/analysisFiles";
import AnalysisFileList from "./components/AnalysisFileList";
import DataTable from "./components/DataTable";
import StatCard from "./components/StatCard";
import FeedbackSummaryCards from "./components/FeedbackSummaryCards";
import FigureDashboard from "@/app/research-dashboard/components/FigureDashboard";
import FoldableSection from "@/app/research-dashboard/components/FoldableSection";

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

function groupFigures<T extends { slug: string }>(figures: T[]) {
  return [
    {
      id: "participant-figures",
      title: "Participant Information",
      description: "Age, gender, education, language, and AI attitude distributions.",
      figures: figures.filter((figure) =>
        ["51_", "52_", "53_", "54_", "55_", "56_"].some((prefix) => figure.slug.startsWith(prefix))
      ),
    },
    {
      id: "workflow-figures",
      title: "Workflow Behaviour",
      description: "Workflow choices, transitions, and final preferences.",
      figures: figures.filter((figure) =>
        ["01_", "02_", "03_", "04_", "05_", "05b_", "06_", "07_", "08_", "09_"].some((prefix) =>
          figure.slug.startsWith(prefix)
        )
      ),
    },
    {
      id: "evaluator-figures",
      title: "Evaluator Agreement",
      description:
        "Inter-rater agreement (ICC, Ordinal Krippendorff's Alpha, Cohen's Kappa) and individual evaluator rating patterns.",
      figures: figures.filter((figure) =>
        ["61_", "62_", "63_", "64_", "65_", "66_", "67_"].some((prefix) =>
          figure.slug.startsWith(prefix)
        )
      ),
    },
    {
      id: "quality-figures",
      title: "Output Quality",
      description: "Quality dimensions by round and workflow in practice rounds.",
      figures: figures.filter((figure) =>
        ["11_", "12_", "13_", "14_", "15_", "15b_"].some((prefix) => figure.slug.startsWith(prefix))
      ),
    },
    {
      id: "efficiency-figures",
      title: "Efficiency",
      description: "Efficiency metrics by round and workflow in practice rounds.",
      figures: figures.filter((figure) =>
        ["16_", "17_"].some((prefix) => figure.slug.startsWith(prefix))
      ),
    },
    {
      id: "constraints-figures",
      title: "Constraint Fulfillment",
      description: "Whether submitted poems fulfilled the task constraints in practice rounds.",
      figures: figures.filter((figure) =>
        ["21_", "22_", "23_"].some((prefix) => figure.slug.startsWith(prefix))
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
        "Round-5 error exposure, line-count error, post-error behaviour, and subjective reactions.",
      figures: figures.filter((figure) =>
        ["41_", "42_", "43_", "44_", "45_", "46_", "47_", "48_", "49_"].some((prefix) =>
          figure.slug.startsWith(prefix)
        )
      ),
    },
  ].filter((group) => group.figures.length > 0);
}

const FAVORITE_FIGURE_NUMBERS = [
  "02_",
  "03_",
  "07_",
  "08_", // workflow behaviour
  "62_",
  "64_",
  "65_",
  "66_", // evaluator agreement
  "11_", // output quality
  "16_",
  "17_", // efficiency
  "21_",
  "23_", // constraints fulfillment
  "32_",
  "33_", // participant experience
  "42_",
  "43_",
  "44_",
  "46_",
  "47_",
  "48_", // ai error exposure
];

const INTERESTING_FIGURE_NUMBERS = [
  "09_", // workflow behavior
  "63_", // evaluator agreement
  "12_", // output quality
  // constraints fulfillment
  "31_", // participant experience
  "45_",
  "49_", // ai error exposure
];

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
          <div className="rounded-2xl shadow-sm dark:bg-black">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Data Visualization</h2>

            <p className="mt-2 max-w-3xl text-sm leading-6 text-gray-600 dark:text-slate-300">
              Generated with Python/Matplotlib. Figures are grouped by analysis topic and can be
              viewed fullscreen or downloaded as PNG, PDF, or SVG.
            </p>
          </div>

          <FigureDashboard
            figureGroups={figureGroups}
            favoriteNumbers={FAVORITE_FIGURE_NUMBERS}
            interestingNumbers={INTERESTING_FIGURE_NUMBERS}
          />
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

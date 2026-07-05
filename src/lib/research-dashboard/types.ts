export type Workflow = "human" | "ai" | "human_ai" | "ai_human";

export type StatCardData = {
  title: string;
  value: string | number;
  subtitle?: string;
};

export type CountRow = {
  label: string;
  count: number;
  percent?: number;
};

export type WorkflowChoiceRow = {
  roundIndex: number;
  workflow: string;
  count: number;
  percent: number;
};

export type TransitionRow = {
  fromRound: number;
  toRound: number;
  fromWorkflow: string;
  toWorkflow: string;
  count: number;
};

export type QualityByWorkflowRow = {
  workflow: string;
  poems: number;
  meanFluency: number | null;
  meanThemeAlignment: number | null;
  meanMeaningfulness: number | null;
  meanPoeticness: number | null;
  meanOverallQuality: number | null;
};

export type SubjectiveByWorkflowRow = {
  workflow: string;
  rounds: number;
  meanSatisfaction: number | null;
  meanFrustration: number | null;
  meanEffort: number | null;
  meanPerformance: number | null;
};

export type AiTrustRoundRow = {
  condition: string;
  roundIndex: number;
  meanAiPerformance: number | null;
  meanAiUnderstanding: number | null;
  meanAiCollaboration: number | null;
  meanAiCreativitySupport: number | null;
  count: number;
};

export type ConstraintByWorkflowRow = {
  workflow: string;
  rounds: number;
  passedRate: number | null;
  meanConstraintScore: number | null;
};

export type QualityTimePoint = {
  poemId: string;
  workflow: string;
  participantId: number | null;
  roundIndex: number | null;
  timeMinutes: number;
  meanOverallQuality: number;
};

export type FinalRankingRow = {
  workflow: string;
  firstChoiceCount: number;
  averageRank: number | null;
};

export type CommentThemeRow = {
  theme: string;
  count: number;
};

export type EvaluatorProgressRow = {
  evaluatorId: string;
  ratingCount: number;
  progressPercent: number;
  completed: boolean;
};

export type IncompletePoemRow = {
  poemId: string;
  participantId: number | null;
  roundIndex: number | null;
  taskId: string;
  workflow: string;
  ratingCount: number;
};

export type ResearchDashboardData = {
  totalPoems: number;
  nonEmptyPoems: number;
  emptyPoems: number;
  totalRatings: number;
  expectedRatings: number;
  completionPercent: number;
  fullyRatedPoems: number;

  evaluatorProgress: EvaluatorProgressRow[];
  incompletePoems: IncompletePoemRow[];

  workflowChoiceRows: WorkflowChoiceRow[];
  transitionRows: TransitionRow[];
  qualityByWorkflow: QualityByWorkflowRow[];
  subjectiveByWorkflow: SubjectiveByWorkflowRow[];
  aiTrustByRound: AiTrustRoundRow[];
  constraintByWorkflow: ConstraintByWorkflowRow[];
  qualityTimePoints: QualityTimePoint[];
  finalRanking: FinalRankingRow[];
  commentThemes: CommentThemeRow[];

  hasDashboardDataset: boolean;
};

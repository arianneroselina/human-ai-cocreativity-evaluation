import "server-only";

import fs from "fs";
import path from "path";
import Papa from "papaparse";
import { prisma } from "@/lib/prisma";
import {
  getAiTrustByRound,
  getCommentThemes,
  getConstraintByWorkflow,
  getFinalRankingRows,
  getQualityByWorkflow,
  getQualityTimePoints,
  getSubjectiveByWorkflow,
  getTransitionRows,
  getWorkflowChoiceRows,
} from "./stats";
import type { ResearchDashboardData } from "./types";

const EXPECTED_EVALUATORS = 3;
const MASTER_DATASET_PATH = path.join(
  process.cwd(),
  "data",
  "processed",
  "master_round_dataset.csv",
);

const INPUTS_DIR = path.join(process.cwd(), "inputs");

function readCsvFile(filePath: string): Record<string, unknown>[] {
  if (!fs.existsSync(filePath)) return [];

  const content = fs.readFileSync(filePath, "utf-8");

  const parsed = Papa.parse<Record<string, unknown>>(content, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: true,
  });

  return parsed.data;
}

function readAllFinalFeedbackRows() {
  if (!fs.existsSync(INPUTS_DIR)) return [];

  const folders = fs
    .readdirSync(INPUTS_DIR, { withFileTypes: true })
    .filter((item) => item.isDirectory())
    .map((item) => item.name);

  return folders.flatMap((folder) => {
    const feedbackPath = path.join(INPUTS_DIR, folder, "Feedback.csv");
    return readCsvFile(feedbackPath);
  });
}

export async function getResearchDashboardData(): Promise<ResearchDashboardData> {
  const [
    totalPoems,
    nonEmptyPoems,
    emptyPoems,
    totalRatings,
    evaluatorSessions,
    poems,
  ] = await Promise.all([
    prisma.poem.count(),
    prisma.poem.count({
      where: {
        isEmpty: false,
      },
    }),
    prisma.poem.count({
      where: {
        isEmpty: true,
      },
    }),
    prisma.rating.count(),
    prisma.evaluationSession.findMany({
      select: {
        evaluatorId: true,
        completedAt: true,
        _count: {
          select: {
            ratings: true,
          },
        },
      },
      orderBy: {
        evaluatorId: "asc",
      },
    }),
    prisma.poem.findMany({
      where: {
        isEmpty: false,
      },
      select: {
        id: true,
        participantId: true,
        roundIndex: true,
        taskId: true,
        workflow: true,
        _count: {
          select: {
            ratings: true,
          },
        },
      },
      orderBy: [
        {
          participantId: "asc",
        },
        {
          roundIndex: "asc",
        },
      ],
    }),
  ]);

  const expectedRatings = nonEmptyPoems * EXPECTED_EVALUATORS;
  const completionPercent =
    expectedRatings > 0 ? (totalRatings / expectedRatings) * 100 : 0;

  const incompletePoems = poems
    .filter((poem) => poem._count.ratings !== EXPECTED_EVALUATORS)
    .map((poem) => ({
      poemId: poem.id,
      participantId: poem.participantId,
      roundIndex: poem.roundIndex,
      taskId: poem.taskId,
      workflow: poem.workflow,
      ratingCount: poem._count.ratings,
    }));

  const evaluatorProgress = evaluatorSessions.map((session) => ({
    evaluatorId: session.evaluatorId,
    ratingCount: session._count.ratings,
    progressPercent:
      nonEmptyPoems > 0 ? (session._count.ratings / nonEmptyPoems) * 100 : 0,
    completed: Boolean(session.completedAt),
  }));

  const hasMasterDataset = fs.existsSync(MASTER_DATASET_PATH);
  const masterRows = readCsvFile(MASTER_DATASET_PATH);
  const finalFeedbackRows = readAllFinalFeedbackRows();

  return {
    totalPoems,
    nonEmptyPoems,
    emptyPoems,
    totalRatings,
    expectedRatings,
    completionPercent,
    fullyRatedPoems: nonEmptyPoems - incompletePoems.length,

    evaluatorProgress,
    incompletePoems,

    workflowChoiceRows: getWorkflowChoiceRows(masterRows),
    transitionRows: getTransitionRows(masterRows),
    qualityByWorkflow: getQualityByWorkflow(masterRows),
    subjectiveByWorkflow: getSubjectiveByWorkflow(masterRows),
    aiTrustByRound: getAiTrustByRound(masterRows),
    constraintByWorkflow: getConstraintByWorkflow(masterRows),
    qualityTimePoints: getQualityTimePoints(masterRows),
    finalRanking: getFinalRankingRows(finalFeedbackRows),
    commentThemes: getCommentThemes(masterRows, finalFeedbackRows),

    hasMasterDataset,
  };
}

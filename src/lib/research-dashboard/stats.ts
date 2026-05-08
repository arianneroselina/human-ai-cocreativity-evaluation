import type {
  AiTrustRoundRow,
  CommentThemeRow,
  ConstraintByWorkflowRow,
  FinalRankingRow,
  QualityByWorkflowRow,
  QualityTimePoint,
  SubjectiveByWorkflowRow,
  TransitionRow,
  WorkflowChoiceRow,
} from "./types";

type AnyRow = Record<string, unknown>;

export function toNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;

  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : null;
}

export function mean(values: Array<number | null | undefined>) {
  const clean = values.filter(
    (value): value is number => typeof value === "number" && Number.isFinite(value)
  );

  if (clean.length === 0) return null;

  return clean.reduce((sum, value) => sum + value, 0) / clean.length;
}

export function percent(count: number, total: number) {
  if (total === 0) return 0;
  return (count / total) * 100;
}

function groupBy<T>(items: T[], getKey: (item: T) => string) {
  return items.reduce(
    (acc, item) => {
      const key = getKey(item);
      acc[key] ??= [];
      acc[key].push(item);
      return acc;
    },
    {} as Record<string, T[]>
  );
}

export function getWorkflowChoiceRows(rows: AnyRow[]): WorkflowChoiceRow[] {
  const choiceRows = rows.filter((row) => {
    const roundIndex = toNumber(row.roundIndex);
    return roundIndex !== null && roundIndex >= 5;
  });

  const byRound = groupBy(choiceRows, (row) => String(row.roundIndex));

  return Object.entries(byRound)
    .flatMap(([roundIndex, roundRows]) => {
      const byWorkflow = groupBy(roundRows, (row) => String(row.workflow));

      return Object.entries(byWorkflow).map(([workflow, workflowRows]) => ({
        roundIndex: Number(roundIndex),
        workflow,
        count: workflowRows.length,
        percent: percent(workflowRows.length, roundRows.length),
      }));
    })
    .sort((a, b) => a.roundIndex - b.roundIndex || a.workflow.localeCompare(b.workflow));
}

export function getTransitionRows(rows: AnyRow[]): TransitionRow[] {
  const byParticipant = groupBy(rows, (row) => String(row.participantId ?? "unknown"));
  const transitionCounts = new Map<string, TransitionRow>();

  Object.values(byParticipant).forEach((participantRows) => {
    const sortedRows = participantRows
      .map((row) => ({
        roundIndex: toNumber(row.roundIndex),
        workflow: String(row.workflow ?? ""),
      }))
      .filter((row) => row.roundIndex !== null && row.workflow)
      .sort((a, b) => Number(a.roundIndex) - Number(b.roundIndex));

    for (let index = 0; index < sortedRows.length - 1; index += 1) {
      const current = sortedRows[index];
      const next = sortedRows[index + 1];

      if (current.roundIndex === null || next.roundIndex === null) continue;

      // Focus on transition into and within choice phase: 4→5, 5→6, 6→7
      if (current.roundIndex < 4 || next.roundIndex < 5) continue;

      const key = [current.roundIndex, next.roundIndex, current.workflow, next.workflow].join("|");

      const existing = transitionCounts.get(key);

      if (existing) {
        existing.count += 1;
      } else {
        transitionCounts.set(key, {
          fromRound: current.roundIndex,
          toRound: next.roundIndex,
          fromWorkflow: current.workflow,
          toWorkflow: next.workflow,
          count: 1,
        });
      }
    }
  });

  return Array.from(transitionCounts.values()).sort(
    (a, b) => a.fromRound - b.fromRound || a.toRound - b.toRound || b.count - a.count
  );
}

export function getQualityByWorkflow(rows: AnyRow[]): QualityByWorkflowRow[] {
  const rowsWithWorkflow = rows.filter((row) => row.workflow);
  const byWorkflow = groupBy(rowsWithWorkflow, (row) => String(row.workflow));

  return Object.entries(byWorkflow)
    .map(([workflow, workflowRows]) => ({
      workflow,
      poems: workflowRows.length,
      meanFluency: mean(workflowRows.map((row) => toNumber(row.meanFluency))),
      meanThemeAlignment: mean(workflowRows.map((row) => toNumber(row.meanThemeAlignment))),
      meanMeaningfulness: mean(workflowRows.map((row) => toNumber(row.meanMeaningfulness))),
      meanPoeticness: mean(workflowRows.map((row) => toNumber(row.meanPoeticness))),
      meanOverallQuality: mean(workflowRows.map((row) => toNumber(row.meanOverallQuality))),
      qualityComposite: mean(workflowRows.map((row) => toNumber(row.qualityComposite))),
    }))
    .sort((a, b) => a.workflow.localeCompare(b.workflow));
}

export function getSubjectiveByWorkflow(rows: AnyRow[]): SubjectiveByWorkflowRow[] {
  const byWorkflow = groupBy(
    rows.filter((row) => row.workflow),
    (row) => String(row.workflow)
  );

  return Object.entries(byWorkflow)
    .map(([workflow, workflowRows]) => ({
      workflow,
      rounds: workflowRows.length,
      meanSatisfaction: mean(workflowRows.map((row) => toNumber(row.satisfactionResult))),
      meanFrustration: mean(workflowRows.map((row) => toNumber(row.frustration))),
      meanEffort: mean(workflowRows.map((row) => toNumber(row.effort))),
      meanPerformance: mean(workflowRows.map((row) => toNumber(row.performance))),
    }))
    .sort((a, b) => a.workflow.localeCompare(b.workflow));
}

export function getAiTrustByRound(rows: AnyRow[]): AiTrustRoundRow[] {
  const aiRows = rows.filter((row) => {
    const workflow = String(row.workflow ?? "");
    return workflow !== "human";
  });

  const byConditionRound = groupBy(aiRows, (row) => {
    const condition = String(row.condition || "unknown");
    const roundIndex = String(row.roundIndex ?? "unknown");
    return `${condition}|${roundIndex}`;
  });

  return Object.entries(byConditionRound)
    .map(([key, groupRows]) => {
      const [condition, roundIndex] = key.split("|");

      return {
        condition,
        roundIndex: Number(roundIndex),
        meanAiPerformance: mean(groupRows.map((row) => toNumber(row.aiPerformanceOverall))),
        meanAiUnderstanding: mean(groupRows.map((row) => toNumber(row.aiUnderstanding))),
        meanAiCollaboration: mean(groupRows.map((row) => toNumber(row.aiCollaboration))),
        meanAiCreativitySupport: mean(groupRows.map((row) => toNumber(row.aiCreativitySupport))),
        count: groupRows.length,
      };
    })
    .filter((row) => Number.isFinite(row.roundIndex))
    .sort((a, b) => a.condition.localeCompare(b.condition) || a.roundIndex - b.roundIndex);
}

export function getConstraintByWorkflow(rows: AnyRow[]): ConstraintByWorkflowRow[] {
  const byWorkflow = groupBy(
    rows.filter((row) => row.workflow),
    (row) => String(row.workflow)
  );

  return Object.entries(byWorkflow)
    .map(([workflow, workflowRows]) => {
      const passedValues = workflowRows
        .map((row): 0 | 1 | null => {
          if (row.passed === true || row.passed === "true" || row.passed === "t") {
            return 1;
          }

          if (row.passed === false || row.passed === "false" || row.passed === "f") {
            return 0;
          }

          return null;
        })
        .filter((value): value is 0 | 1 => value !== null);

      const passedTotal = passedValues.reduce(
        (sum: number, value) => sum + value,
        0,
      );

      return {
        workflow,
        rounds: workflowRows.length,
        passedRate:
          passedValues.length > 0
            ? (passedTotal / passedValues.length) * 100
            : null,
        meanConstraintScore: mean(workflowRows.map((row) => toNumber(row.constraintScore))),
      };
    })
    .sort((a, b) => a.workflow.localeCompare(b.workflow));
}

export function getQualityTimePoints(rows: AnyRow[]): QualityTimePoint[] {
  return rows
    .map((row) => {
      const timeMinutes = toNumber(row.effectiveTimeMinutes) ?? (toNumber(row.timeMs) ?? 0) / 60000;
      const qualityComposite = toNumber(row.qualityComposite);

      if (!timeMinutes || qualityComposite === null) return null;

      return {
        poemId: String(row.roundId ?? row.poemId ?? ""),
        workflow: String(row.workflow ?? "unknown"),
        participantId: toNumber(row.participantId),
        roundIndex: toNumber(row.roundIndex),
        timeMinutes,
        qualityComposite,
      };
    })
    .filter((row): row is QualityTimePoint => row !== null);
}

function parseRanking(value: unknown): string[] {
  if (!value || typeof value !== "string") return [];

  try {
    const parsed = JSON.parse(value);

    if (Array.isArray(parsed)) {
      return parsed
        .map((item) => {
          if (typeof item === "string") return item;
          if (item && typeof item === "object" && "workflow" in item) {
            return String(item.workflow);
          }
          return "";
        })
        .filter(Boolean);
    }
  } catch {
    // Fall back to separator-based parsing below.
  }

  return value
    .split(/[>,;\n|]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export function getFinalRankingRows(feedbackRows: AnyRow[]): FinalRankingRow[] {
  const workflows = ["human", "ai", "human_ai", "ai_human"];
  const rankSums = new Map<string, number>();
  const rankCounts = new Map<string, number>();
  const firstChoiceCounts = new Map<string, number>();

  workflows.forEach((workflow) => {
    rankSums.set(workflow, 0);
    rankCounts.set(workflow, 0);
    firstChoiceCounts.set(workflow, 0);
  });

  feedbackRows.forEach((row) => {
    const ranking = parseRanking(row.workflowRanking);

    ranking.forEach((workflow, index) => {
      if (!workflows.includes(workflow)) return;

      rankSums.set(workflow, (rankSums.get(workflow) ?? 0) + index + 1);
      rankCounts.set(workflow, (rankCounts.get(workflow) ?? 0) + 1);

      if (index === 0) {
        firstChoiceCounts.set(workflow, (firstChoiceCounts.get(workflow) ?? 0) + 1);
      }
    });
  });

  return workflows.map((workflow) => {
    const rankCount = rankCounts.get(workflow) ?? 0;

    return {
      workflow,
      firstChoiceCount: firstChoiceCounts.get(workflow) ?? 0,
      averageRank: rankCount > 0 ? (rankSums.get(workflow) ?? 0) / rankCount : null,
    };
  });
}

const themeKeywords: Record<string, string[]> = {
  "AI error / misunderstanding": [
    "error",
    "mistake",
    "wrong",
    "incorrect",
    "misunderstood",
    "not understand",
    "didn't understand",
  ],
  "Control / ownership": ["control", "ownership", "own", "my text", "edit"],
  "Speed / time": ["time", "fast", "quick", "slow", "deadline"],
  Creativity: ["creative", "creativity", "idea", "inspiration"],
  Quality: ["quality", "better", "good", "bad", "improve"],
  "Rules / constraints": ["rule", "constraint", "requirement", "forbidden", "required"],
  Frustration: ["frustrated", "frustrating", "annoying", "stress", "difficult"],
  Trust: ["trust", "reliable", "confidence", "depend"],
  Helpfulness: ["helpful", "support", "assist", "useful"],
};

export function getCommentThemes(rows: AnyRow[], feedbackRows: AnyRow[]): CommentThemeRow[] {
  const comments = [
    ...rows.map((row) => String(row.roundComment ?? "")),
    ...feedbackRows.map((row) => String(row.comments ?? "")),
    ...feedbackRows.map((row) => String(row.rankingReason ?? "")),
  ].filter((comment) => comment.trim().length > 0);

  return Object.entries(themeKeywords)
    .map(([theme, keywords]) => {
      const count = comments.filter((comment) => {
        const lower = comment.toLowerCase();
        return keywords.some((keyword) => lower.includes(keyword));
      }).length;

      return {
        theme,
        count,
      };
    })
    .filter((row) => row.count > 0)
    .sort((a, b) => b.count - a.count);
}

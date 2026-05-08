export type Poem = {
  id: string;

  sessionId: string | null;
  participantId: number | null;
  roundIndex: number | null;

  taskId: string;
  topic: string;
  text: string;
  workflow: "human" | "ai" | "human_ai" | "ai_human";

  isEmpty: boolean;
  timeMs: number | null;
  wordCount: number | null;
  charCount: number | null;
  passed: boolean;

  startedAt: Date | null;
  submittedAt: Date | null;
};

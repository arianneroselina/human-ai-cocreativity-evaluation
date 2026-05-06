export type Poem = {
  id: string;
  taskId: string;
  topic: string;
  text: string;
  workflow: string;
  timeMs: number | null;
  wordCount: number | null;
  charCount: number | null;
  passed: boolean;
};

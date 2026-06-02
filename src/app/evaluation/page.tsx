import EvaluationWorkbench from "./components/EvaluationWorkbench";
import { prisma } from "@/lib/prisma";

export const dynamic = "force-dynamic";

export default async function Page() {
  const poems = await prisma.poem.findMany({
    where: {
      isEmpty: false,
    },
    select: {
      id: true,
      sessionId: true,
      participantId: true,
      roundIndex: true,
      taskId: true,
      topic: true,
      text: true,
      workflow: true,
      isEmpty: true,
      timeMs: true,
      wordCount: true,
      charCount: true,
      passed: true,
      startedAt: true,
      submittedAt: true,
    },
    orderBy: [
      {
        participantId: "asc",
      },
      {
        roundIndex: "asc",
      },
    ],
  });

  return (
    <main className="mx-auto max-w-7xl p-6">
      <EvaluationWorkbench poems={poems} />
    </main>
  );
}

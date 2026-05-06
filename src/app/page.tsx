import EvaluationWorkbench from "./evaluation/evaluationWorkbench";
import { prisma } from "@/lib/prisma";

export const dynamic = "force-dynamic";

export default async function Page() {
  const poems = await prisma.poem.findMany({
    where: {
      isEmpty: false,
    },
    select: {
      id: true,
      taskId: true,
      topic: true,
      text: true,
      workflow: true,
      timeMs: true,
      wordCount: true,
      charCount: true,
      passed: true,
    },
    orderBy: {
      id: "asc",
    },
  });

  return (
    <main className="mx-auto max-w-7xl p-6">
      <EvaluationWorkbench poems={poems} />
    </main>
  );
}

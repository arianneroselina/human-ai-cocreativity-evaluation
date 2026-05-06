import "dotenv/config";
import { PrismaClient, Workflow } from "@prisma/client";
import { PrismaPg } from "@prisma/adapter-pg";
import { loadPoems } from "@/data/loadPoems";

const adapter = new PrismaPg({
  connectionString: process.env.PRISMA_DATABASE_URL!,
});

const prisma = new PrismaClient({ adapter });

function mapWorkflow(workflow: string): Workflow {
  const normalized = workflow.trim().toLowerCase();

  switch (normalized) {
    case "human":
      return Workflow.human;
    case "ai":
      return Workflow.ai;
    case "human_ai":
      return Workflow.human_ai;
    case "ai_human":
      return Workflow.ai_human;
    default:
      throw new Error(`Unknown workflow value: ${workflow}`);
  }
}

async function seedPoems() {
  console.log("Seeding poems...");
  const poems = loadPoems();

  try {
    for (const poem of poems) {
      await prisma.poem.upsert({
        where: {
          id: poem.id,
        },
        update: {
          taskId: poem.taskId,
          topic: poem.topic,
          text: poem.text,
          workflow: mapWorkflow(poem.workflow),

          isEmpty: poem.text.trim() === "",
          timeMs: poem.timeMs,
          wordCount: poem.wordCount,
          charCount: poem.charCount,
          passed: poem.passed,
        },
        create: {
          id: poem.id,
          taskId: poem.taskId,
          topic: poem.topic,
          text: poem.text,
          workflow: mapWorkflow(poem.workflow),

          isEmpty: poem.text.trim() === "",
          timeMs: poem.timeMs,
          wordCount: poem.wordCount,
          charCount: poem.charCount,
          passed: poem.passed,
        },
      });
    }

    console.log(`Successfully seeded ${poems.length} poems.`);
  } catch (error) {
    console.error("Error seeding poems:", error);
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
}

seedPoems();

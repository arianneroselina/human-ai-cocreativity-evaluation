import "dotenv/config";
import { PrismaClient } from "@prisma/client";
import { PrismaPg } from "@prisma/adapter-pg";
import { loadPoems } from "@/data/loadPoems";

const adapter = new PrismaPg({
  connectionString: process.env.PRISMA_DATABASE_URL!,
});

const prisma = new PrismaClient({ adapter });

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
          workflow: poem.workflow,

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
          workflow: poem.workflow,

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

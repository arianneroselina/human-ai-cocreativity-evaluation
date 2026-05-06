import "dotenv/config";
import { PrismaClient } from "@prisma/client";
import { PrismaPg } from "@prisma/adapter-pg";
import { loadedPoems } from "@/data/loadPoems";

const adapter = new PrismaPg({
  connectionString: process.env.DATABASE_URL!,
});

const prisma = new PrismaClient({ adapter });

async function seedPoems() {
  console.log("Seeding poems...");

  try {
    for (const poem of loadedPoems) {
      await prisma.poem.create({
        data: {
          id: poem.id,
          taskId: poem.taskId,
          topic: poem.topic,
          text: poem.text,
          workflow: poem.workflow,
        },
      });
    }

    console.log(`Successfully seeded ${loadedPoems.length} poems.`);
  } catch (error) {
    console.error("Error seeding poems:", error);
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
}

seedPoems();

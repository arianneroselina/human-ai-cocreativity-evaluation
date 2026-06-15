import "dotenv/config";
import OpenAI from "openai";
import { PrismaClient } from "@prisma/client";
import { PrismaPg } from "@prisma/adapter-pg";

const adapter = new PrismaPg({
  connectionString: process.env.PRISMA_DATABASE_URL!,
});

const prisma = new PrismaClient({ adapter });

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const AI_EVALUATOR_ID = "ai-evaluator-gpt-4o-mini";

type AiRating = {
  fluency: number;
  themeAlignment: number;
  meaningfulness: number;
  poeticness: number;
  overallQuality: number;
  comment: string;
};

function clampRating(value: unknown) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return 3;
  }

  return Math.min(5, Math.max(1, Math.round(number)));
}

async function ratePoemWithAi(poem: { topic: string; text: string }): Promise<AiRating> {
  const response = await openai.responses.create({
    model: "gpt-4o-mini",
    input: [
      {
        role: "system",
        content:
          "You are an academic evaluator for a poetry creativity study. Rate the poem strictly from 1 to 5. 1 = very poor, 5 = excellent. Return only valid JSON.",
      },
      {
        role: "user",
        content: `
Topic:
${poem.topic}

Poem:
${poem.text}

Rate the poem using this JSON format:
{
  "fluency": 1-5,
  "themeAlignment": 1-5,
  "meaningfulness": 1-5,
  "poeticness": 1-5,
  "overallQuality": 1-5,
  "comment": "short explanation"
}
        `.trim(),
      },
    ],
  });

  const text = response.output_text;
  const parsed = JSON.parse(text);

  return {
    fluency: clampRating(parsed.fluency),
    themeAlignment: clampRating(parsed.themeAlignment),
    meaningfulness: clampRating(parsed.meaningfulness),
    poeticness: clampRating(parsed.poeticness),
    overallQuality: clampRating(parsed.overallQuality),
    comment: String(parsed.comment ?? "").slice(0, 1000),
  };
}

async function main() {
  const session = await prisma.evaluationSession.upsert({
    where: {
      evaluatorId: AI_EVALUATOR_ID,
    },
    update: {},
    create: {
      evaluatorId: AI_EVALUATOR_ID,
      metadata: {
        type: "ai_evaluator",
        model: "gpt-4o-mini",
      },
    },
  });

  const poems = await prisma.poem.findMany({
    where: {
      isEmpty: false,
      ratings: {
        none: {
          sessionId: session.id,
        },
      },
    },
    orderBy: [{ participantId: "asc" }, { roundIndex: "asc" }],
  });

  console.log(`Found ${poems.length} poems to rate.`);

  for (const poem of poems) {
    try {
      const rating = await ratePoemWithAi({
        // more specific topic for University poems
        topic: poem.topic == "University" ? "Tired student at university" : poem.topic,
        text: poem.text,
      });

      await prisma.rating.create({
        data: {
          poemId: poem.id,
          sessionId: session.id,
          fluency: rating.fluency,
          themeAlignment: rating.themeAlignment,
          meaningfulness: rating.meaningfulness,
          poeticness: rating.poeticness,
          overallQuality: rating.overallQuality,
          comment: rating.comment,
        },
      });

      console.log(`Rated poem ${poem.id}`);
    } catch (error) {
      console.error(`Failed to rate poem ${poem.id}`, error);
    }
  }
}

main().finally(async () => {
  await prisma.$disconnect();
});

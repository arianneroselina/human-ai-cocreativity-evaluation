import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function POST(request: Request) {
  const body = await request.json();
  const evaluatorId = String(body.evaluatorId ?? "").trim();

  if (!evaluatorId) {
    return NextResponse.json(
      { error: "Evaluator ID is required." },
      { status: 400 },
    );
  }

  const session = await prisma.evaluationSession.upsert({
    where: {
      evaluatorId,
    },
    update: {},
    create: {
      evaluatorId,
      userAgent: request.headers.get("user-agent"),
    },
    include: {
      ratings: {
        select: {
          poemId: true,
        },
      },
    },
  });

  return NextResponse.json({
    sessionId: session.id,
    evaluatorId: session.evaluatorId,
    ratedPoemIds: session.ratings.map((rating) => rating.poemId),
  });
}

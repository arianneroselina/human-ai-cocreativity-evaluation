import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

function isValidScore(value: unknown) {
  return typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= 5;
}

export async function POST(request: Request) {
  try {
    const body = await request.json();

    const {
      sessionId,
      poemId,
      fluency,
      themeAlignment,
      meaningfulness,
      poeticness,
      overallQuality,
      comment,
    } = body;

    if (!sessionId || typeof sessionId !== "string") {
      return NextResponse.json({ error: "Missing sessionId." }, { status: 400 });
    }

    if (!poemId || typeof poemId !== "string") {
      return NextResponse.json({ error: "Missing poemId." }, { status: 400 });
    }

    if (
      !isValidScore(fluency) ||
      !isValidScore(themeAlignment) ||
      !isValidScore(meaningfulness) ||
      !isValidScore(poeticness) ||
      !isValidScore(overallQuality)
    ) {
      return NextResponse.json(
        { error: "All ratings must be integers from 1 to 5." },
        { status: 400 }
      );
    }

    const session = await prisma.evaluationSession.findUnique({
      where: {
        id: sessionId,
      },
      select: {
        id: true,
      },
    });

    if (!session) {
      return NextResponse.json({ error: "Evaluation session not found." }, { status: 404 });
    }

    const cleanComment =
      typeof comment === "string" && comment.trim() !== "" ? comment.trim() : null;

    const rating = await prisma.rating.upsert({
      where: {
        poemId_sessionId: {
          poemId,
          sessionId,
        },
      },
      update: {
        fluency,
        themeAlignment,
        meaningfulness,
        poeticness,
        overallQuality,
        comment: cleanComment,
      },
      create: {
        poemId,
        sessionId,
        fluency,
        themeAlignment,
        meaningfulness,
        poeticness,
        overallQuality,
        comment: cleanComment,
      },
    });

    return NextResponse.json({
      success: true,
      ratingId: rating.id,
      sessionId,
    });
  } catch (error) {
    console.error("Failed to save rating:", error);

    return NextResponse.json({ error: "Failed to save rating." }, { status: 500 });
  }
}

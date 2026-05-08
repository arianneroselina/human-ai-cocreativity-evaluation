import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

function isValidScore(value: unknown) {
  return (
    typeof value === "number" &&
    Number.isInteger(value) &&
    value >= 1 &&
    value <= 10
  );
}

export async function POST(request: Request) {
  try {
    const body = await request.json();

    const {
      sessionId,
      poemId,
      clarity,
      creativity,
      relevance,
      quality,
      comment,
    } = body;

    if (!sessionId || typeof sessionId !== "string") {
      return NextResponse.json(
        { error: "Missing sessionId." },
        { status: 400 },
      );
    }

    if (!poemId || typeof poemId !== "string") {
      return NextResponse.json(
        { error: "Missing poemId." },
        { status: 400 },
      );
    }

    if (
      !isValidScore(clarity) ||
      !isValidScore(creativity) ||
      !isValidScore(relevance) ||
      !isValidScore(quality)
    ) {
      return NextResponse.json(
        { error: "All ratings must be integers from 1 to 10." },
        { status: 400 },
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
      return NextResponse.json(
        { error: "Evaluation session not found." },
        { status: 404 },
      );
    }

    const cleanComment =
      typeof comment === "string" && comment.trim() !== ""
        ? comment.trim()
        : null;

    const rating = await prisma.rating.upsert({
      where: {
        poemId_sessionId: {
          poemId,
          sessionId,
        },
      },
      update: {
        clarity,
        creativity,
        relevance,
        quality,
        comment: cleanComment,
      },
      create: {
        poemId,
        sessionId,
        clarity,
        creativity,
        relevance,
        quality,
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

    return NextResponse.json(
      { error: "Failed to save rating." },
      { status: 500 },
    );
  }
}

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
      poemId,
      clarity,
      creativity,
      relevance,
      quality,
      comment,
    } = body;

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

    const session = await prisma.evaluationSession.create({
      data: {},
    });

    const rating = await prisma.rating.create({
      data: {
        poemId,
        sessionId: session.id,
        clarity,
        creativity,
        relevance,
        quality,
        comment: comment?.trim() || null,
      },
    });

    return NextResponse.json({
      success: true,
      ratingId: rating.id,
      sessionId: session.id,
    });
  } catch (error) {
    console.error("Failed to save rating:", error);

    return NextResponse.json(
      { error: "Failed to save rating." },
      { status: 500 },
    );
  }
}

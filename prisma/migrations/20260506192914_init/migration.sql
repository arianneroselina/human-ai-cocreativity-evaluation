-- CreateEnum
CREATE TYPE "Workflow" AS ENUM ('human', 'ai', 'human_ai', 'ai_human');

-- CreateTable
CREATE TABLE "Poem" (
    "id" TEXT NOT NULL,
    "taskId" TEXT NOT NULL,
    "topic" TEXT NOT NULL,
    "text" TEXT NOT NULL,
    "workflow" "Workflow" NOT NULL,
    "isEmpty" BOOLEAN NOT NULL DEFAULT false,
    "timeMs" INTEGER,
    "wordCount" INTEGER,
    "charCount" INTEGER,
    "passed" BOOLEAN NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Poem_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "EvaluationSession" (
    "id" TEXT NOT NULL,
    "evaluatorCode" TEXT,
    "startedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completedAt" TIMESTAMP(3),
    "userAgent" TEXT,
    "metadata" JSONB,

    CONSTRAINT "EvaluationSession_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Rating" (
    "id" TEXT NOT NULL,
    "poemId" TEXT NOT NULL,
    "sessionId" TEXT NOT NULL,
    "clarity" INTEGER NOT NULL,
    "creativity" INTEGER NOT NULL,
    "relevance" INTEGER NOT NULL,
    "quality" INTEGER NOT NULL,
    "comment" TEXT,
    "timeSpentMs" INTEGER,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Rating_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Rating_poemId_sessionId_key" ON "Rating"("poemId", "sessionId");

-- AddForeignKey
ALTER TABLE "Rating" ADD CONSTRAINT "Rating_poemId_fkey" FOREIGN KEY ("poemId") REFERENCES "Poem"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Rating" ADD CONSTRAINT "Rating_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "EvaluationSession"("id") ON DELETE CASCADE ON UPDATE CASCADE;

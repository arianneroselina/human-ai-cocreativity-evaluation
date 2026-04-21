export type CreationMode = "Human" | "AI" | "Co-created";

export type PoemEntry = {
  id: string;
  participantId: string;
  topicId: string;
  topicLabel: string;
  title: string;
  text: string;
  creationMode: CreationMode;
};

export type EvaluationPair = {
  id: string;
  topicId: string;
  topicLabel: string;
  left: PoemEntry;
  right: PoemEntry;
};

const topics = [
  "Rainy City",
  "Childhood Memory",
  "Loneliness",
  "Ocean at Night",
  "Hope",
  "Machine Dreams",
  "Farewell",
] as const;

const topicIds = ["topic-1", "topic-2", "topic-3", "topic-4", "topic-5", "topic-6", "topic-7"] as const;

const participantIds = Array.from({ length: 24 }, (_, index) => `P${String(index + 1).padStart(2, "0")}`);
const creationModes: CreationMode[] = ["Human", "AI", "Co-created"];

function createPoemText(topic: string, participantNumber: number, variant: number) {
  const stanzas = [
    `In ${topic.toLowerCase()}, the windows keep a soft account of light,`,
    `and somewhere a small memory folds itself into the evening.`,
    `I carry its echo like a note pressed flat inside a book,`,
    `waiting for the page that lets it breathe again.`,
    `The street, the room, the thought — each shifts a little closer,`,
    `until silence sounds less empty than before.`,
  ];

  return stanzas
    .map((line, idx) => {
      if ((idx + participantNumber + variant) % 3 === 0) {
        return `${line} `;
      }
      if ((idx + participantNumber + variant) % 4 === 0) {
        return `${line.replace("soft", "faint").replace("small", "quiet")}`;
      }
      return line;
    })
    .join("\n");
}

export const allPoems: PoemEntry[] = participantIds.flatMap((participantId, participantIndex) =>
  topics.map((topicLabel, topicIndex) => {
    const creationMode = creationModes[(participantIndex + topicIndex) % creationModes.length];
    return {
      id: `${participantId}-${topicIds[topicIndex]}`,
      participantId,
      topicId: topicIds[topicIndex],
      topicLabel,
      title: `${topicLabel} #${participantIndex + 1}`,
      creationMode,
      text: createPoemText(topicLabel, participantIndex + 1, topicIndex + 1),
    };
  })
);

export const evaluationPairs: EvaluationPair[] = topicIds.flatMap((topicId, topicIndex) => {
  const poemsForTopic = allPoems.filter((poem) => poem.topicId === topicId);
  return Array.from({ length: 4 }, (_, pairIndex) => {
    const left = poemsForTopic[pairIndex * 2];
    const right = poemsForTopic[pairIndex * 2 + 1];

    return {
      id: `${topicId}-pair-${pairIndex + 1}`,
      topicId,
      topicLabel: topics[topicIndex],
      left,
      right,
    };
  });
});

export const evaluationSummary = {
  participants: participantIds.length,
  topics: topics.length,
  totalPoems: allPoems.length,
  visiblePairs: evaluationPairs.length,
};

export const topicOverview = topicIds.map((topicId, index) => ({
  id: topicId,
  label: topics[index],
  poemCount: allPoems.filter((poem) => poem.topicId === topicId).length,
  pairCount: evaluationPairs.filter((pair) => pair.topicId === topicId).length,
}));

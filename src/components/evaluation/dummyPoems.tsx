export type DummyPoem = {
  id: string;
  taskId: string;
  topic: string;
  text: string;
  // hidden in UI, only useful later for analysis
  workflow: "human" | "ai" | "human_ai" | "ai_human";
};

export const dummyPoems: DummyPoem[] = [
  {
    id: "poem-001",
    taskId: "task-01",
    topic: "Rainy evening",
    workflow: "human",
    text: `The rain taps softly on the glass
A silver song in evening light
The street forgets its hurried past
And windows bloom with amber bright
I watch the clouds dissolve the day
While quiet thoughts drift far away`,
  },
  {
    id: "poem-002",
    taskId: "task-02",
    topic: "A robot learning kindness",
    workflow: "ai",
    text: `I learned to hold a flower gently
Not as data, but as dawn
My metal hands became a garden
Where careful little hopes were drawn
A kindness grew inside my code
And lit the path I walked upon`,
  },
  {
    id: "poem-003",
    taskId: "task-03",
    topic: "The ocean at night",
    workflow: "human_ai",
    text: `The ocean breathes beneath the moon
Its salted whisper cool and wide
The stars fall softly into waves
Like lanterns carried by the tide
I taste the dark on moving air
And hear the deep world turn inside`,
  },
];

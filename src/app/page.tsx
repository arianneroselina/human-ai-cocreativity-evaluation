import { loadPoems } from "@/data/loadPoems";
import EvaluationWorkbench from "@/components/evaluation/evaluationWorkbench";

export default function Page() {
  const poems = loadPoems();

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-6 sm:px-6 lg:px-8">
      <EvaluationWorkbench poems={poems}/>
    </div>
  );
}

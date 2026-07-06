import type { AnalysisFile } from "@/lib/research-dashboard/analysisFiles";

export default function AnalysisFileList({ files }: { files: AnalysisFile[] }) {
  if (files.length === 0) {
    return (
      <div className="rounded-xl border border-yellow-200 bg-yellow-50 p-4 text-sm text-yellow-900">
        No analysis files found. Run <code className="font-mono">make process-data</code> to
        generate statistical analysis outputs.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="border-b bg-gray-50 text-gray-600 dark:border-slate-700 dark:bg-black dark:text-white">
          <tr>
            <th className="px-4 py-3">File</th>
            <th className="px-4 py-3">Size</th>
            <th className="px-4 py-3">Download</th>
          </tr>
        </thead>
        <tbody>
          {files.map((file) => (
            <tr key={file.name} className="border-b">
              <td className="px-4 py-3 font-mono text-xs">{file.name}</td>
              <td className="px-4 py-3 text-gray-500">{file.sizeKb} KB</td>
              <td className="px-4 py-3">
                <a
                  href={file.url}
                  download
                  className="inline-flex items-center gap-1.5 rounded-md border border-gray-200 bg-white px-3 py-1.5 text-xs font-medium text-gray-700 shadow-sm transition hover:border-gray-300 hover:bg-gray-50"
                >
                  ↓ Download
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

import type { ReactNode } from "react";

type Column<T> = {
  header: string;
  render: (row: T) => ReactNode;
};

export default function DataTable<T>({
  rows,
  columns,
  emptyText = "No data available.",
}: {
  rows: T[];
  columns: Column<T>[];
  emptyText?: string;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="border-b bg-gray-50 text-gray-600">
          <tr>
            {columns.map((column) => (
              <th key={column.header} className="px-4 py-3">
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td className="px-4 py-4 text-gray-500" colSpan={columns.length}>
                {emptyText}
              </td>
            </tr>
          ) : (
            rows.map((row, index) => (
              <tr key={index} className="border-b">
                {columns.map((column) => (
                  <td key={column.header} className="px-4 py-3">
                    {column.render(row)}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

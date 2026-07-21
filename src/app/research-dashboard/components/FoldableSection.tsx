import type { ReactNode } from "react";

export default function FoldableSection({
  id,
  title,
  description,
  badge,
  defaultOpen = false,
  children,
}: {
  id: string;
  title: string;
  description?: string;
  badge?: string | number;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  return (
    <details
      id={id}
      open={defaultOpen}
      className="group scroll-mt-24 overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm dark:border-slate-700 dark:bg-black"
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 p-5 transition hover:bg-gray-50 dark:hover:bg-slate-800 [&::-webkit-details-marker]:hidden">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">{title}</h2>

            {badge !== undefined && (
              <span className="rounded-full bg-gray-100 px-3 py-1 text-sm font-medium text-gray-700">
                {badge}
              </span>
            )}
          </div>

          {description && (
            <p className="mt-1 max-w-3xl text-sm leading-6 text-gray-600 dark:text-slate-300">
              {description}
            </p>
          )}
        </div>

        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-gray-200 bg-white text-xl text-gray-500 transition group-open:rotate-90">
          ›
        </span>
      </summary>

      <div className="border-t border-gray-100 bg-white p-5 dark:border-slate-800 dark:bg-black">
        {children}
      </div>
    </details>
  );
}

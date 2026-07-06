export default function StatCard({
  title,
  value,
  subtitle,
}: {
  title: string;
  value: string | number;
  subtitle?: string;
}) {
  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm dark:border-slate-700 dark:bg-black">
      <p className="text-sm text-gray-500 dark:text-slate-200">{title}</p>
      <p className="mt-2 text-3xl font-semibold text-gray-900 dark:text-white">{value}</p>
      {subtitle && <p className="mt-1 text-sm text-gray-500 dark:text-slate-200">{subtitle}</p>}
    </div>
  );
}

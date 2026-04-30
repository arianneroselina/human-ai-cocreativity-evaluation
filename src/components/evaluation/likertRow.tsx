"use client";

export type Likert = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10;

type LikertRowProps = {
  label: string;
  value: Likert | null;
  onChange: (v: Likert) => void;
  left?: string;
  right?: string;
};

const values: Likert[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

export default function LikertRow({
                                    label,
                                    value,
                                    onChange,
                                    left = "Low",
                                    right = "High",
                                  }: LikertRowProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">{label}</span>
        <span className="text-xs text-muted-foreground">{value ?? "—"}</span>
      </div>

      <div className="space-y-1">
        <div className="grid grid-cols-10 gap-1.5" role="radiogroup" aria-label={label}>
          {values.map((v) => {
            const active = value === v;

            return (
              <button
                key={v}
                type="button"
                role="radio"
                aria-checked={active}
                onClick={() => onChange(v)}
                className={[
                  "h-8 rounded-md border text-xs transition",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  active
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border bg-background hover:bg-accent",
                ].join(" ")}
              >
                {v}
              </button>
            );
          })}
        </div>

        <div className="flex justify-between text-[11px] text-muted-foreground">
          <span>{left}</span>
          <span>{right}</span>
        </div>
      </div>
    </div>
  );
}

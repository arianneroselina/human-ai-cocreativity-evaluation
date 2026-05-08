import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function parseNumber(value: unknown): number | null {
  if (typeof value !== "string" || value.trim() === "") return null;

  const parsed = Number(value);

  return Number.isFinite(parsed) ? parsed : null;
}

export function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

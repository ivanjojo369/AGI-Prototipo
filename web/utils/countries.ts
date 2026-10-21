// web/utils/countries.ts
export const COUNTRIES = [
  { code: "JP", name: "Japan" },
  { code: "KR", name: "Korea" },
  { code: "US", name: "United States" },
  { code: "MX", name: "Mexico" },
  { code: "DE", name: "Germany" } 
] as const;

export type CountryCode = (typeof COUNTRIES)[number]["code"];

export function isCountryCode(x: string | undefined | null): x is CountryCode {
  if (!x) return false;
  return COUNTRIES.some((c) => c.code === x.toUpperCase());
}

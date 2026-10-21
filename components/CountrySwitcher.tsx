'use client';

import { useState, type ChangeEvent } from 'react';

const COUNTRIES = [
  { code: 'us', label: 'Estados Unidos' },
  { code: 'mx', label: 'México' },
  { code: 'es', label: 'España' },
] as const;

type CountryCode = typeof COUNTRIES[number]['code'] | '';

export default function CountrySwitcher() {
  const [value, setValue] = useState<CountryCode>('');

  async function onChange(e: ChangeEvent<HTMLSelectElement>) {
    const code = (e.target.value || '') as CountryCode;
    setValue(code);

    // Si tienes un endpoint para guardar preferencia:
    // (No falla si aún no lo creas)
    try {
      await fetch('/api/prefs/country', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ country: code || null }),
      });
    } catch {
      // silenciar por ahora
    }
  }

  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-zinc-500">Salida por país:</span>
      <select
        className="rounded-md border px-2 py-1"
        value={value}
        onChange={onChange}
      >
        <option value="">Auto</option>
        {COUNTRIES.map((c) => (
          <option key={c.code} value={c.code}>
            {c.label}
          </option>
        ))}
      </select>
    </div>
  );
}

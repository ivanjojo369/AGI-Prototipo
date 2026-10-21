// web/app/green/page.tsx
"use client";

import { useState } from "react";

type Leaderboard = {
  summary: {
    num_tasks: number;
    avg_score: number;
    p50_lat_ms: number;
    p95_lat_ms: number;
    avg_tokens_per_s: number;
    no_answer_rate: number;
    elapsed_s: number;
  };
  results: Array<{
    task: string;
    question: string;
    answer: string;
    score: number;
    citations_ok: boolean;
    lat_ms: number;
    tokens_per_s: number;
  }>;
};

export default function GreenBenchPage() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<Leaderboard | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function runBench() {
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const res = await fetch("/api/green/run", { method: "POST" });
      const json = await res.json();
      if (!res.ok || !json.ok) throw new Error(json?.error || "Fallo al ejecutar el bench");
      setData(json.report as Leaderboard);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="p-6 max-w-5xl mx-auto">
      <h1 className="text-2xl font-bold mb-2">GREEN Agent — Research Bench</h1>
      <p className="text-sm text-gray-600 mb-6">
        Ejecuta el benchmark del agente de investigación y muestra el <code>leaderboard.json</code>.
      </p>

      <button
        onClick={runBench}
        disabled={loading}
        className="px-4 py-2 rounded-md bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50"
      >
        {loading ? "Ejecutando…" : "Ejecutar bench"}
      </button>

      {error && (
        <div className="mt-4 p-3 rounded-md bg-red-100 text-red-800 border border-red-200">
          Error: {error}
        </div>
      )}

      {data && (
        <section className="mt-6 space-y-4">
          <h2 className="text-xl font-semibold">Resumen</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
            <Stat k="Tareas" v={data.summary.num_tasks} />
            <Stat k="Avg score" v={round(data.summary.avg_score)} />
            <Stat k="p50 lat (ms)" v={round(data.summary.p50_lat_ms)} />
            <Stat k="p95 lat (ms)" v={round(data.summary.p95_lat_ms)} />
            <Stat k="tokens/s" v={round(data.summary.avg_tokens_per_s)} />
            <Stat k="no-answer rate" v={round(data.summary.no_answer_rate)} />
            <Stat k="elapsed (s)" v={round(data.summary.elapsed_s)} />
          </div>

          <h3 className="text-lg font-medium mt-6">Resultados por tarea</h3>
          <div className="space-y-3">
            {data.results.map((r) => (
              <div key={r.task} className="p-3 rounded-lg border">
                <div className="text-xs text-gray-500 mb-1">Tarea {r.task}</div>
                <div className="font-medium mb-1">{r.question}</div>
                <div className="text-sm">
                  <span className="font-semibold">Respuesta:</span> {r.answer}
                </div>
                <div className="text-xs mt-1 text-gray-600">
                  score={round(r.score)} · lat={r.lat_ms}ms · t/s={round(r.tokens_per_s)} · citas_ok=
                  {String(r.citations_ok)}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}

function Stat({ k, v }: { k: string; v: number }) {
  return (
    <div className="p-3 rounded-md border">
      <div className="text-gray-500 text-xs">{k}</div>
      <div className="text-base font-semibold">{v}</div>
    </div>
  );
}

function round(n: number) {
  return Math.round((n + Number.EPSILON) * 1000) / 1000;
}

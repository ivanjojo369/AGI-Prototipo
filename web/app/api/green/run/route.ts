// web/app/api/green/run/route.ts
import { NextResponse } from "next/server";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import fs from "node:fs/promises";
import path from "node:path";

export const runtime = "nodejs";

const pexecFile = promisify(execFile);

// Root del repo (sube 1 nivel desde web/)
function repoRoot(...p: string[]) {
  return path.join(process.cwd(), "..", ...p);
}

export async function POST() {
  const PYTHON = process.env.PYTHON_EXE || "python";
  const HOST = process.env.AB_HOST || "127.0.0.1";
  const PORT = process.env.AB_PORT || "8020";
  const API_KEY = process.env.AB_API_KEY || "";

  // Forzamos que el runner deje el archivo dentro de green_agent/
  const outPath = repoRoot("green_agent", "leaderboard.json");

  const args = ["-m", "green_agent.runner", "--host", HOST, "--port", PORT, "--out", outPath];
  if (API_KEY) args.push("--api-key", API_KEY);

  try {
    const { stdout, stderr } = await pexecFile(PYTHON, args, {
      cwd: repoRoot(),        // ejecutar desde raíz del repo
      windowsHide: true,
      timeout: 5 * 60 * 1000,
      maxBuffer: 10 * 1024 * 1024,
    });

    // Leer el leaderboard forzado
    const raw = await fs.readFile(outPath, "utf-8");
    const report = JSON.parse(raw);

    return NextResponse.json({
      ok: true,
      message: "Bench ejecutado",
      runner_stdout: stdout,
      runner_stderr: stderr,
      leaderboard_path: outPath,
      report,
    });
  } catch (err: any) {
    return NextResponse.json(
      { ok: false, error: err?.message || String(err), stack: err?.stack },
      { status: 500 }
    );
  }
}

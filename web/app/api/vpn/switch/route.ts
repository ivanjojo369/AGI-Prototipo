import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));
  const country = String(body?.country ?? 'auto').toLowerCase();

  const py = process.env.PYTHON_BIN || 'python';
  const script = process.env.VPN_MANAGER_PATH || 'tools/vpn_manager.py';

  const out: string[] = [];
  const err: string[] = [];

  const code = await new Promise<number>((resolve) => {
    const p = spawn(py, [script, '--country', country], { env: { ...process.env } });
    p.stdout.on('data', (d) => out.push(d.toString()));
    p.stderr.on('data', (d) => err.push(d.toString()));
    p.on('close', (code) => resolve(code ?? 0));
  });

  return NextResponse.json({
    ok: code === 0,
    code,
    out: out.join(''),
    err: err.join(''),
  });
}

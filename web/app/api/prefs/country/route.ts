import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));
  const c = String(body?.country ?? 'auto').toLowerCase();

  const res = NextResponse.json({ ok: true, country: c });
  res.cookies.set('country', c, {
    path: '/',
    httpOnly: false,
    sameSite: 'lax',
    maxAge: 60 * 60 * 24 * 365, // 1 año
  });
  return res;
}

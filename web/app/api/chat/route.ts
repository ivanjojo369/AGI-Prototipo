// web/app/api/chat/route.ts
import { NextResponse } from "next/server";

export const runtime = "nodejs";

type Msg = {
  role: "user" | "assistant" | "system";
  content: string;
};

function quipuBase(): string {
  return (
    process.env.QUIPU_LOOP_URL ||
    "http://127.0.0.1:8130"
  ).replace(/\/+$/, "");
}

async function parseResponse(response: Response) {
  const raw = await response.text();

  try {
    return JSON.parse(raw);
  } catch {
    return { reply: raw };
  }
}

export async function GET(req: Request) {
  const base = quipuBase();
  const url = new URL(req.url);
  const action = url.searchParams.get("action") || "list";
  const id = url.searchParams.get("id") || "";

  try {
    let target = `${base}/conversations`;

    if (action === "load") {
      if (!id) {
        return NextResponse.json(
          { error: true, detail: "Falta conversation id." },
          { status: 400 }
        );
      }

      target = `${base}/conversations/${encodeURIComponent(id)}`;
    }

    const response = await fetch(target, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });

    const data = await parseResponse(response);

    if (!response.ok) {
      return NextResponse.json(
        {
          error: true,
          status: response.status,
          detail: data?.detail ?? "Error en Quipu Loop",
        },
        { status: response.status }
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch (err: any) {
    return NextResponse.json(
      {
        error: true,
        detail:
          err?.message ??
          "No se pudo conectar con Quipu Loop",
      },
      { status: 500 }
    );
  }
}

export async function POST(req: Request) {
  const base = quipuBase();

  let body: any;

  try {
    body = await req.json();
  } catch {
    body = {};
  }

  const action = String(body?.action || "send_message");

  try {
    if (action === "create_conversation") {
      const response = await fetch(`${base}/conversations`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          title: body?.title ?? "Nuevo chat",
        }),
        cache: "no-store",
      });

      const data = await parseResponse(response);

      if (!response.ok) {
        return NextResponse.json(
          {
            error: true,
            status: response.status,
            detail: data?.detail ?? "No se pudo crear el chat",
          },
          { status: response.status }
        );
      }

      return NextResponse.json(data, { status: 200 });
    }

    const messages: Msg[] =
      body?.messages ??
      (body?.message
        ? [{
            role: "user",
            content: String(body.message),
          }]
        : []);

    const response = await fetch(`${base}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        conversation_id: body?.conversation_id ?? "",
        message: body?.message ?? "",
        messages,
      }),
      cache: "no-store",
    });

    const data = await parseResponse(response);

    if (!response.ok) {
      return NextResponse.json(
        {
          error: true,
          status: response.status,
          detail: data?.detail ?? "Error en Quipu Loop",
        },
        { status: response.status }
      );
    }

    return NextResponse.json(
      {
        reply: data?.reply ?? "",
        engine: data?.engine ?? "QuipuLoop",
        conversation_id: data?.conversation_id ?? "",
        conversation: data?.conversation ?? null,
        scientific_engine: Boolean(data?.scientific_engine),
        scientific_intent: data?.scientific_intent ?? null,
      },
      { status: 200 }
    );
  } catch (err: any) {
    return NextResponse.json(
      {
        error: true,
        detail:
          err?.message ??
          "No se pudo conectar con Quipu Loop",
      },
      { status: 500 }
    );
  }
}

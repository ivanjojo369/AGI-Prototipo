"use client";

import { useEffect, useState } from "react";
import { COUNTRIES, isCountryCode } from "@/utils/countries";

type Status = {
  connected: boolean;
  country: string | null;
  pid?: number | null;
  message?: string;
};

export default function VpnSwitcher() {
  const [country, setCountry] = useState<string>("JP");
  const [pref, setPref] = useState<string | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [msg, setMsg] = useState<string | null>(null);

  async function fetchPref() {
    try {
      const r = await fetch("/api/prefs/country", { cache: "no-store" });
      if (!r.ok) return;
      const j = await r.json();
      if (j?.country && isCountryCode(j.country)) {
        setPref(j.country);
        setCountry(j.country);
      }
    } catch {}
  }

  async function savePref(next: string) {
    try {
      await fetch("/api/prefs/country", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ country: next }),
      });
      setPref(next);
    } catch {}
  }

  async function doAction(action: "connect" | "disconnect" | "status") {
    setLoading(true);
    setMsg(null);
    try {
      const r = await fetch("/api/vpn/switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body:
          action === "connect"
            ? JSON.stringify({ action, country })
            : JSON.stringify({ action }),
      });

      const j = (await r.json().catch(() => ({}))) as Status & {
        error?: string;
        ok?: boolean;
      };

      if (!r.ok) {
        setMsg(j?.message || j?.error || "Error");
      } else {
        setStatus(j);
        if (action === "connect") setMsg(`Conectando a ${country}…`);
        if (action === "disconnect") setMsg("Desconectado.");
      }
    } catch (e: any) {
      setMsg(e?.message ?? "Error de red");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchPref().then(() => doAction("status"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      style={{
        maxWidth: 560,
        margin: "0 auto",
        display: "grid",
        gap: 12,
        padding: 12,
        border: "1px solid #e5e7eb",
        borderRadius: 16,
      }}
    >
      <label style={{ fontSize: 14 }}>País preferido</label>

      <select
        value={country}
        disabled={loading}
        onChange={(e) => {
          const next = e.target.value.toUpperCase();
          setCountry(next);
          if (isCountryCode(next)) savePref(next);
        }}
        style={{
          padding: 8,
          borderRadius: 8,
          border: "1px solid #d1d5db",
        }}
      >
        {COUNTRIES.map((c) => (
          <option key={c.code} value={c.code}>
            {c.name} ({c.code})
          </option>
        ))}
      </select>

      <div style={{ fontSize: 12, color: "#6b7280" }}>
        Guardado: {pref ?? "—"}
      </div>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button
          onClick={() => doAction("connect")}
          disabled={loading}
          style={btn}
        >
          Conectar
        </button>
        <button
          onClick={() => doAction("disconnect")}
          disabled={loading}
          style={btn}
        >
          Desconectar
        </button>
        <button
          onClick={() => doAction("status")}
          disabled={loading}
          style={btn}
        >
          Estado
        </button>
      </div>

      {msg && <div style={{ fontSize: 13 }}>{msg}</div>}

      <div
        style={{
          padding: 12,
          border: "1px solid #e5e7eb",
          borderRadius: 12,
          background: "#fafafa",
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 8 }}>Estado</div>
        <pre style={{ fontSize: 12, whiteSpace: "pre-wrap", margin: 0 }}>
          {status ? JSON.stringify(status, null, 2) : "—"}
        </pre>
      </div>
    </div>
  );
}

const btn: React.CSSProperties = {
  padding: "8px 12px",
  borderRadius: 10,
  border: "1px solid #d1d5db",
  background: "white",
  cursor: "pointer",
};

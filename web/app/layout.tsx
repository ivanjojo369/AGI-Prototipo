// web/app/layout.tsx
export const metadata = {
  title: "Synchronexis Chat",
  description: "Interfaz mínima para tu AGI doméstica textual",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es" suppressHydrationWarning>
      <body
        suppressHydrationWarning
        style={{
          minHeight: "100vh",
          margin: 0,
          background: "#f7f7f7",
          color: "#111",
          fontFamily:
            "-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,Apple Color Emoji,Segoe UI Emoji",
        }}
      >
        {children}
      </body>
    </html>
  );
}

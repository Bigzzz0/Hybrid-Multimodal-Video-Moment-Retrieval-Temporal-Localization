import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VideoMoment AI Studio - Pure Visual Spatiotemporal Grounding",
  description: "Pure-visual natural language video moment retrieval using SigLIP 2, CapRL Q6, calibrated temporal proposals, and LanceDB",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#060911] text-gray-100 antialiased selection:bg-cyan-500/30 selection:text-cyan-200 relative">
        {/* Ambient Cyber Lighting Mesh */}
        <div className="ambient-glow" />

        <header className="sticky top-0 z-50 border-b border-surfaceBorder/80 glass-panel px-6 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-3.5">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 via-cyan-400 to-emerald-400 p-[1.5px] shadow-lg shadow-cyan-500/20">
              <div className="w-full h-full bg-[#080c16] rounded-[10px] flex items-center justify-center font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-indigo-300 text-lg">
                V
              </div>
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-2">
                VideoMoment <span className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-full bg-cyan-950/80 text-cyan-300 border border-cyan-800">Pure Visual</span>
              </h1>
                <p className="text-[11px] text-gray-400">SigLIP 2 NaFlex • CapRL Q6 • Calibrated Temporal Proposals • LanceDB</p>
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs font-mono">
            <span className="hidden sm:flex items-center gap-1.5 text-cyan-300 font-medium px-2.5 py-1 rounded-full bg-cyan-950/40 border border-cyan-800/40">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
              Pure Visual Focus
            </span>
            <span className="flex items-center gap-1.5 text-emerald-400 font-medium px-2.5 py-1 rounded-full bg-emerald-950/40 border border-emerald-800/40">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              NVIDIA GPU Acceleration Active
            </span>
          </div>
        </header>

        <main className="max-w-7xl mx-auto p-4 sm:p-6 space-y-6 relative z-10">
          {children}
        </main>
      </body>
    </html>
  );
}

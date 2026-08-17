import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VideoMoment AI - SOTA Multimodal Video Retrieval",
  description: "Natural Language Video Moment Retrieval & Temporal Localization using SigLIP 2, Qwen2.5-VL 7B, Whisper-Turbo, and LanceDB",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-background text-gray-100 antialiased selection:bg-cyan-500/30 selection:text-cyan-200">
        <header className="sticky top-0 z-50 border-b border-surfaceBorder/80 glass-panel px-6 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-cyan-500 to-emerald-400 p-[1px] shadow-lg shadow-cyan-500/20">
              <div className="w-full h-full bg-background rounded-[11px] flex items-center justify-center font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-indigo-400 text-lg">
                V
              </div>
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-2">
                VideoMoment <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">SOTA AI</span>
              </h1>
              <p className="text-[11px] text-gray-400">SigLIP 2 • Qwen2.5-VL 7B • Whisper-Turbo • LanceDB</p>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs">
            <span className="flex items-center gap-1.5 text-emerald-400 font-medium px-2.5 py-1 rounded-full bg-emerald-950/40 border border-emerald-800/40">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              On-Premise GPU Engine Active
            </span>
          </div>
        </header>

        <main className="max-w-7xl mx-auto p-6 space-y-6">
          {children}
        </main>
      </body>
    </html>
  );
}

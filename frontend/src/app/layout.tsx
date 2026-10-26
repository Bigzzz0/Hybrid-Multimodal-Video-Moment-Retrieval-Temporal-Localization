import type { Metadata } from "next";
import "./globals.css";
import { LanguageProvider } from "@/lib/language";

export const metadata: Metadata = {
  title: "Moment Search - ค้นหาเหตุการณ์ในวิดีโอ CCTV ด้วยภาษาธรรมชาติ",
  description: "Pure-Visual CCTV Moment Retrieval with SigLIP 2, CapRL Q6 captions, and calibrated temporal localization",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="th" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              try {
                const savedTheme = localStorage.getItem('theme');
                if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                  document.documentElement.classList.add('dark');
                } else {
                  document.documentElement.classList.remove('dark');
                }
              } catch (_) {}
            `,
          }}
        />
      </head>
      <body className="min-h-screen text-[var(--text)] bg-[var(--bg)] transition-colors duration-150">
        <LanguageProvider>{children}</LanguageProvider>
      </body>
    </html>
  );
}

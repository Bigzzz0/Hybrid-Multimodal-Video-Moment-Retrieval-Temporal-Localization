/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        themeBg: "var(--bg)",
        themePanel: "var(--panel)",
        themePanelElevated: "var(--panel-elevated)",
        themeLine: "var(--line)",
        themeLineStrong: "var(--line-strong)",
        themeText: "var(--text)",
        themeMuted: "var(--text-muted)",
        themeAccent: "var(--accent)",
        themeAccentHover: "var(--accent-hover)",
        themeAccentSubtle: "var(--accent-subtle)",
        themeAccentText: "var(--accent-text)",
        themeCard: "var(--card-bg)",
        themeCardHover: "var(--card-hover)",
        themeTag: "var(--tag-bg)",
        // Keep legacy aliases safely mapped to CSS variables
        background: "var(--bg)",
        surface: "var(--panel)",
        surfaceBorder: "var(--line)",
      },
      fontFamily: {
        sans: ["'IBM Plex Sans Thai'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
      },
    },
  },
  plugins: [],
};

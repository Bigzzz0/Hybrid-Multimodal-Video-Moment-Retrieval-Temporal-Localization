const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const rootDir = path.resolve(__dirname, '..');
const mdPath = path.join(rootDir, 'Assignment4_Academic_Report.md');
const outHtmlPath = path.join(rootDir, 'Assignment4_Academic_Report.html');

try {
  const htmlBody = execSync(`npx -y marked -i "${mdPath}"`, { encoding: 'utf-8' });

  const fullHtml = `<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<title>รายงานทางวิชาการ (Assignment 4) - Hybrid Multimodal Video Moment Retrieval and Temporal Localization System</title>
<link href="https://fonts.googleapis.com/css2?family=Sarabun:ital,wght@0,300;0,400;0,600;0,700;1,400&display=swap" rel="stylesheet">
<style>
  @page {
    size: A4;
    margin: 2.54cm 2.54cm 2.54cm 2.54cm;
  }
  body {
    font-family: 'Sarabun', 'TH Sarabun New', sans-serif;
    font-size: 16pt;
    line-height: 1.65;
    color: #111827;
    background-color: #ffffff;
    max-width: 900px;
    margin: 0 auto;
    padding: 2cm 1.5cm;
  }
  h1 { font-size: 22pt; font-weight: 700; text-align: center; margin-top: 1.5em; margin-bottom: 0.5em; color: #0f172a; }
  h2 { font-size: 19pt; font-weight: 700; margin-top: 1.2em; margin-bottom: 0.4em; color: #1e293b; }
  h3 { font-size: 17pt; font-weight: 600; margin-top: 1.0em; margin-bottom: 0.3em; color: #334155; }
  h4 { font-size: 16pt; font-weight: 600; margin-top: 0.8em; margin-bottom: 0.2em; color: #475569; }
  p { text-align: justify; text-justify: inter-cluster; margin-bottom: 0.8em; text-indent: 1.25cm; }
  div[align="center"] p, .no-indent, h1 + p { text-indent: 0 !important; }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 1.2em 0;
    font-size: 13.5pt;
  }
  th, td {
    border: 1px solid #94a3b8;
    padding: 8px 10px;
    text-align: left;
    vertical-align: top;
  }
  th {
    background-color: #f1f5f9;
    font-weight: 700;
    text-align: center;
  }
  pre, code {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12pt;
    background-color: #f8fafc;
    border-radius: 4px;
  }
  pre {
    padding: 12px;
    border: 1px solid #e2e8f0;
    overflow-x: auto;
  }
  hr {
    border: none;
    border-top: 1px solid #cbd5e1;
    margin: 2em 0;
  }
  @media print {
    body { padding: 0; max-width: 100%; }
    .no-print { display: none; }
  }
</style>
</head>
<body>
${htmlBody}
</body>
</html>`;

  fs.writeFileSync(outHtmlPath, fullHtml, 'utf-8');
  console.log('Successfully generated Assignment4_Academic_Report.html');
} catch (err) {
  console.error('Error generating HTML:', err.message);
}

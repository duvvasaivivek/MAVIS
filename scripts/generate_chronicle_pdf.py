"""
Generate Clickable PDF from MAVIS_COMPREHENSIVE_CHRONICLE.md
=============================================================
Converts the Markdown chronicle into an academic PDF with:
1. Working Table of Contents internal navigation links (Named Destinations).
2. Rendered LaTeX mathematical equations via MathJax 3.
3. Rendered Mermaid.js architecture and timeline diagrams.
4. Clean GitHub-style print typography with page-break protection.
5. Automated Chrome/Edge headless PDF compilation.
"""

import os
import re
import subprocess
from markdown_it import MarkdownIt

def generate_pdf():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    md_path = os.path.join(project_root, "MAVIS_COMPREHENSIVE_CHRONICLE.md")
    html_path = os.path.join(project_root, "MAVIS_COMPREHENSIVE_CHRONICLE.html")
    pdf_path = os.path.join(project_root, "MAVIS_COMPREHENSIVE_CHRONICLE.pdf")

    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    def slugify(text):
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'[*_`$]', '', text)
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text.strip('-')

    toc_matches = re.findall(r'\[(.*?)\]\(#(.*?)\)', md_text[:3000])
    known_slugs = {title.strip(): anchor.strip() for title, anchor in toc_matches}

    lines = md_text.split('\n')
    new_lines = []
    for line in lines:
        h_match = re.match(r'^(#{1,4})\s+(.+)$', line)
        if h_match:
            hashes, title = h_match.groups()
            auto_slug = slugify(title)
            clean_title = re.sub(r'[*_`$]', '', title).strip()
            anchor_id = auto_slug
            for t_title, t_slug in known_slugs.items():
                if t_slug in auto_slug or auto_slug in t_slug or clean_title in t_title or t_title in clean_title:
                    anchor_id = t_slug
                    break
            new_lines.append(f'<span id="{anchor_id}" class="section-anchor"></span>')
            new_lines.append(f'{hashes} {title}')
        else:
            new_lines.append(line)

    processed_md = '\n'.join(new_lines)

    md_parser = MarkdownIt('commonmark').enable('table')
    body_html = md_parser.render(processed_md)
    body_html = re.sub(r'<pre><code class="language-mermaid">([\s\S]*?)</code></pre>', r'<div class="mermaid">\1</div>', body_html)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MAVIS Comprehensive Technical Chronicle</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  }},
  svg: {{
    fontCache: 'global'
  }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
  document.addEventListener("DOMContentLoaded", function() {{
    mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
  }});
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

body {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.65;
  color: #24292f;
  background-color: #ffffff;
  padding: 40px 60px;
  max-width: 960px;
  margin: 0 auto;
}}

h1, h2, h3, h4, h5, h6 {{
  color: #0f172a;
  font-weight: 600;
  margin-top: 28px;
  margin-bottom: 14px;
  line-height: 1.3;
}}

h1 {{ font-size: 26px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
h2 {{ font-size: 20px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-top: 36px; }}
h3 {{ font-size: 16px; margin-top: 24px; }}
h4 {{ font-size: 14px; }}

a {{
  color: #0969da;
  text-decoration: none;
}}
a:hover {{
  text-decoration: underline;
}}

table {{
  border-collapse: collapse;
  width: 100%;
  margin: 18px 0;
  font-size: 12.5px;
}}
table th, table td {{
  border: 1px solid #d0d7de;
  padding: 8px 12px;
  text-align: left;
}}
table th {{
  background-color: #f6f8fa;
  font-weight: 600;
  color: #24292f;
}}
table tr:nth-child(even) {{
  background-color: #fcfcfd;
}}

pre, code {{
  font-family: 'JetBrains Mono', Consolas, Monaco, monospace;
}}
code {{
  font-size: 12px;
  background-color: #f1f5f9;
  padding: 2px 5px;
  border-radius: 4px;
  color: #0f172a;
}}
pre {{
  background-color: #0f172a;
  color: #f8fafc;
  padding: 14px 18px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 12px;
  line-height: 1.5;
}}
pre code {{
  background-color: transparent;
  color: inherit;
  padding: 0;
}}

blockquote {{
  margin: 16px 0;
  padding: 8px 16px;
  color: #475569;
  background-color: #f8fafc;
  border-left: 4px solid #3b82f6;
  border-radius: 0 4px 4px 0;
}}

hr {{
  border: none;
  border-top: 1px solid #e2e8f0;
  margin: 32px 0;
}}

.mermaid {{
  display: flex;
  justify-content: center;
  margin: 20px 0;
}}

.section-anchor {{
  display: block;
  position: relative;
  top: -10px;
  visibility: hidden;
}}

@media print {{
  @page {{
    size: A4;
    margin: 16mm 18mm 18mm 18mm;
  }}
  body {{
    padding: 0;
    max-width: 100%;
    font-size: 11pt;
    line-height: 1.5;
  }}
  h1, h2, h3, h4 {{
    page-break-after: avoid;
  }}
  table, pre, blockquote, .mermaid {{
    page-break-inside: avoid;
  }}
  a {{
    color: #0969da !important;
  }}
}}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    browser = chrome_path if os.path.exists(chrome_path) else edge_path

    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        f"file:///{html_path}"
    ]

    subprocess.run(cmd, check=True)
    if os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"[PDF Generator] Successfully generated: {pdf_path} ({size_kb:.1f} KB)")
    else:
        print("[PDF Generator] Error: Failed to generate PDF.")

if __name__ == "__main__":
    generate_pdf()

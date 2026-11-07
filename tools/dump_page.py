import sys
import pymupdf
from ingest import PDFExtractor, ExtractorConfig

path, pg = sys.argv[1], int(sys.argv[2])
c = ExtractorConfig()
ex = PDFExtractor(c)
page = pymupdf.open(path)[pg - 1]
blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
sep_y = ex._find_separator_y(blocks)
print(f"=== pdf page {pg} | footnote separator y = {sep_y} ===")


def render(ln):
    out = []
    for s in ln["spans"]:
        t = s["text"]
        if s["size"] <= c.marker_font_max and t.strip():
            out.append(f"⟦{t.strip()}⟧")
        else:
            out.append(t)
    return "".join(out)


print("\n--- BODY lines that contain a marker (y, text) ---")
for b in blocks:
    if b["bbox"][1] < c.header_max_y:
        continue
    for ln in b["lines"]:
        if sep_y is not None and ln["bbox"][1] >= sep_y:
            continue
        txt = render(ln)
        if "⟦" in txt:
            print(f"y={ln['bbox'][1]:6.1f}  {txt.strip()[:200]}")

print("\n--- FOOTNOTE area ---")
for b in blocks:
    for ln in b["lines"]:
        if sep_y is not None and ln["bbox"][1] >= sep_y:
            print(f"y={ln['bbox'][1]:6.1f}  {render(ln).strip()[:200]}")
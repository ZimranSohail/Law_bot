import sys
import pymupdf

path, pg = sys.argv[1], int(sys.argv[2])
page = pymupdf.open(path)[pg - 1]
print(f"page height {page.rect.height:.0f}")
for b in page.get_text("dict")["blocks"]:
    if b["type"] != 0:
        continue
    for ln in b["lines"]:
        if ln["bbox"][1] < 430:
            continue
        sizes = sorted({round(s["size"], 1) for s in ln["spans"]})
        txt = "".join(s["text"] for s in ln["spans"])
        print(f"y={ln['bbox'][1]:6.1f} sizes={sizes} len={len(txt.strip()):3} {txt.strip()[:90]!r}")
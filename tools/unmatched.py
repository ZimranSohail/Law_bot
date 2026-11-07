import sys
from collections import defaultdict
from ingest import PDFExtractor

pages = PDFExtractor().extract(sys.argv[1])
by_page = defaultdict(list)
for p in pages:
    fn = {n for n, _ in p.footnotes}
    for m in p.markers:
        if m.number not in fn:
            by_page[p.pdf_page].append(m.number)

print("pages with unmatched markers:", len(by_page))
for pg, nums in sorted(by_page.items()):
    p = next(x for x in pages if x.pdf_page == pg)
    print(f"  page {pg}: markers {sorted(set(nums), key=int)} | footnotes extracted: {len(p.footnotes)}")
import sys
from ingest import PDFExtractor

pages = PDFExtractor().extract(sys.argv[1])

tot_m = tot_f = 0
marker_no_note, note_no_marker = [], []
for p in pages:
    fn = {n for n, _ in p.footnotes}
    mk = {m.number for m in p.markers}
    tot_m += len(p.markers)
    tot_f += len(p.footnotes)
    marker_no_note += [(p.pdf_page, n) for n in sorted(mk - fn, key=str)]
    note_no_marker += [(p.pdf_page, n) for n in sorted(fn - mk, key=str)]

print(f"{len(pages)} pages | {tot_m} markers | {tot_f} footnotes")
print(f"markers with no footnote on their page: {len(marker_no_note)}  e.g. {marker_no_note[:12]}")
print(f"footnotes with no marker on their page: {len(note_no_marker)}  e.g. {note_no_marker[:12]}")

for pg in (113, 35):
    p = next(x for x in pages if x.pdf_page == pg)
    print(f"\n--- page {pg}: {len(p.markers)} markers ---")
    for m in p.markers:
        print(f"  {m.number:>2} | ...{m.before[-22:]!r} | {m.after[:42]!r}")
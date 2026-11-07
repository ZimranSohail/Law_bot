import io, contextlib, sys
from ingest import PDFExtractor, TextCleaner, ArticleChunker

pdf = sys.argv[1] if len(sys.argv) > 1 else "constitution_of_Pakistan.pdf"
pages = PDFExtractor().extract(pdf)
with contextlib.redirect_stdout(io.StringIO()):
    doc = TextCleaner().clean(pages)

ch = ArticleChunker(anchor=True)
new = ch.chunk(doc)
old = ArticleChunker(anchor=False).chunk(doc)

print(f"chunks: {len(new)} | identical ids+text vs old: "
      f"{[c.id for c in new] == [c.id for c in old] and all(a.text == b.text for a, b in zip(new, old))}")
s = ch.anchor_stats
print(f"markers resolved to a chunk: {s['resolved']}/{s['markers']} ({s['resolved']/s['markers']:.0%})")
print(f"chunks with notes: old {sum('amendment_notes' in c.metadata for c in old)} -> "
      f"new {sum('amendment_notes' in c.metadata for c in new)} | with article_notes: "
      f"{sum('article_notes' in c.metadata for c in new)}")

unres = []
for pg, ms in doc.markers_by_page.items():
    texts = doc.footnote_text_by_page.get(pg, {})
    for idx, m in enumerate(ms):
        if (pg, idx) not in ch.resolved:
            unres.append((pg, m.number, m.before[-22:], m.after[:22], texts.get(m.number, "")[:45]))
unres.sort()
print(f"\nunresolved markers: {len(unres)} (first 15)")
for u in unres[:15]:
    print("  p%d #%s | ...%r | %r | note: %r" % u)

for art in ("175", "175A", "19", "19A", "10A", "11", "24", "25A", "2A", "140A"):
    print(f"\n=== Article {art} ===")
    for c in new:
        if c.metadata.get("article_number") == art and (c.metadata.get("amendment_notes") or c.metadata.get("article_notes")):
            print(f"  {c.id}")
            if "article_notes" in c.metadata:
                print("     ARTICLE:", c.metadata["article_notes"][:160])
            print("     notes  :", c.metadata.get("amendment_notes", "")[:230])
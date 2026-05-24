import stanza

nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", verbose=False)
doc = nlp("The horse raced past the barn fell.")

print("Stanza parse (garden-path / wrong)")
for sent in doc.sentences:
    for w in sent.words:
        hw = "ROOT" if w.head == 0 else sent.words[w.head-1].text
        print(f"  {w.id:<3} {w.text:<8} {w.upos:<6} head={w.head} ({hw})  {w.deprel}")

print("Correct parse (manual UD annotation)")
correct = [
    (1, "The",   "DET",   2, "horse", "det"),
    (2, "horse", "NOUN",  7, "fell",  "nsubj"),
    (3, "raced", "VERB",  2, "horse", "acl"),
    (4, "past",  "ADP",   6, "barn",  "case"),
    (5, "the",   "DET",   6, "barn",  "det"),
    (6, "barn",  "NOUN",  3, "raced", "obl"),
    (7, "fell",  "VERB",  0, "ROOT",  "root"),
    (8, ".",     "PUNCT", 7, "fell",  "punct"),
]
for wid, word, upos, head, hw, rel in correct:
    print(f"  {wid:<3} {word:<8} {upos:<6} head={head} ({hw})  {rel}")

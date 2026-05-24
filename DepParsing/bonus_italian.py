
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stanza

OUT = "results/bonus"
os.makedirs(OUT, exist_ok=True)

# Gold trees (UD annotation)

# 1a: "Lo voglio leggere."  (I want to read it)
# Lo = obj of leggere (clitic has climbed past voglio)
GOLD_P1A = [(1,"Lo","PRON",3,"leggere","obj"),
            (2,"voglio","AUX",3,"leggere","aux"),
            (3,"leggere","VERB",0,"ROOT","root"),
            (4,".","PUNCT",3,"leggere","punct")]

# 1b: "Glielo voglio mandare."  (I want to send it to him)
# Glielo = obj of mandare :  Stanza wrongly labels it nsubj
GOLD_P1B = [(1,"Glielo","PRON",3,"mandare","obj"),
            (2,"voglio","AUX",3,"mandare","aux"),
            (3,"mandare","VERB",0,"ROOT","root"),
            (4,".","PUNCT",3,"mandare","punct")]

# 2a: "Ha telefonato Gianni."  (Gianni has called, VS order)
GOLD_P2A = [(1,"Ha","AUX",2,"telefonato","aux"),
            (2,"telefonato","VERB",0,"ROOT","root"),
            (3,"Gianni","PROPN",2,"telefonato","nsubj"),
            (4,".","PUNCT",2,"telefonato","punct")]

# 2b: "In questo periodo arrivano molti turisti."
# turisti = nsubj of arrivano, appears after the verb
GOLD_P2B = [(1,"In","ADP",3,"periodo","case"),
            (2,"questo","DET",3,"periodo","det"),
            (3,"periodo","NOUN",4,"arrivano","obl"),
            (4,"arrivano","VERB",0,"ROOT","root"),
            (5,"molti","DET",6,"turisti","det"),
            (6,"turisti","NOUN",4,"arrivano","nsubj"),
            (7,".","PUNCT",4,"arrivano","punct")]

def parse(nlp, sentence):
    doc = nlp(sentence)
    rows = []
    for sent in doc.sentences:
        for w in sent.words:
            hw = "ROOT" if w.head == 0 else sent.words[w.head-1].text
            rows.append((w.id, w.text, w.upos, w.head, hw, w.deprel))
    return rows


def print_table(rows, title):
    print(f"\n{title}")
    print(f"  {'ID':<4} {'WORD':<14} {'UPOS':<7} {'HEAD':<6} {'HEAD_WORD':<14} DEPREL")
    print("  " + "-" * 56)
    for wid, word, upos, head, hw, rel in rows:
        print(f"  {wid:<4} {word:<14} {upos:<7} {head:<6} {hw:<14} {rel}")


print("Loading Stanza Italian pipeline...")
nlp = stanza.Pipeline("it", processors="tokenize,pos,lemma,depparse", verbose=False)

pred_p1a = parse(nlp, "Lo voglio leggere.")
pred_p1b = parse(nlp, "Glielo voglio mandare.")
pred_p2a = parse(nlp, "Ha telefonato Gianni.")
pred_p2b = parse(nlp, "In questo periodo arrivano molti turisti.")

print_table(pred_p1a, "1a 'Lo voglio leggere.'  (gold: Lo -> obj -> leggere)")
print_table(pred_p1b, "1b 'Glielo voglio mandare.'  (gold: Glielo -> obj -> mandare)")
print_table(pred_p2a, "2a 'Ha telefonato Gianni.'  (gold: Gianni -> nsubj -> telefonato)")
print_table(pred_p2b, "2b  'In questo periodo arrivano molti turisti.'")
#diagram for latex:
def draw_pair(gold, pred, gold_title, pred_title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 3))
    for ax, rows, title in [(axes[0], gold, gold_title), (axes[1], pred, pred_title)]:
        words = [r[1] for r in rows]
        heads = [r[3] for r in rows]
        labels = [r[5] for r in rows]
        n = len(words)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.3, 2.5)
        ax.axis("off")
        ax.set_title(title, fontsize=9, fontweight="bold")
        for i, word in enumerate(words):
            ax.text(i, 0, word, ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", ec="gray", lw=0.8))
        for dep_i, (head_i, label) in enumerate(zip(heads, labels)):
            if head_i == 0:
                ax.annotate("", xy=(dep_i, 0.4), xytext=(dep_i, 1.8),
                            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2))
                ax.text(dep_i, 1.95, "ROOT", ha="center", fontsize=7,
                        color="#1f77b4", fontweight="bold")
                ax.text(dep_i+0.08, 1.1, label, ha="left", fontsize=6, color="#1f77b4")
            else:
                h, d = head_i-1, dep_i
                rad = -0.35 if d > h else 0.35
                ax.annotate("", xy=(d, 0.4), xytext=(h, 0.4),
                            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1,
                                            connectionstyle=f"arc3,rad={rad}"))
                mid = (h+d)/2.0
                height = min(0.4 + abs(h-d)*0.28, 2.1)
                ax.text(mid, height, label, ha="center", fontsize=6, color="#d62728",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    plt.tight_layout()
    path = os.path.join(OUT, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


draw_pair(GOLD_P1A, pred_p1a, "Gold: Lo voglio leggere", "Stanza output", "arc_p1a.png")
draw_pair(GOLD_P1B, pred_p1b, "Gold: Glielo voglio mandare", "Stanza output (ERROR)", "arc_p1b.png")
draw_pair(GOLD_P2A, pred_p2a, "Gold: Ha telefonato Gianni", "Stanza output", "arc_p2a.png")
draw_pair(GOLD_P2B, pred_p2b, "Gold: In questo periodo...", "Stanza output", "arc_p2b.png")

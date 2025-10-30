import argparse, math, os, re, sys
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Iterable

ALPHABET = [chr(c) for c in range(ord('a'), ord('z') + 1)]
START, END = "<S>", "</S>"
ASPELL_FILE = "aspell.txt"

#penalties for insertions and deletions
LOG_INS_PEN = math.log(1e-2)
LOG_DEL_PEN = math.log(1e-2)

# helper functions
def only_letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def parse_line(line: str) -> Tuple[str, List[str]]:
    line = line.strip()
    if not line: return None, []
    if ":" in line:
        left, right = line.split(":", 1)
        correct = left.strip()
        typos = [t for t in re.split(r"[,\s]+", right) if t]
        return (correct, typos) if correct and typos else (None, [])
    if "," in line:
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            correct, rest = parts[0], parts[1:]
            typos = []
            for r in rest:
                typos.extend([t for t in r.split() if t])
            return (correct, typos) if correct and typos else (None, [])
    return None, []

def load_aspell() -> Tuple[List[str], List[Tuple[str,str]]]:
    path = os.path.join(os.getcwd(), ASPELL_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Couldn't find '{ASPELL_FILE}' in {os.getcwd()}")
    correct_words, pairs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            corr, typos = parse_line(raw)
            if not corr or not typos: continue
            correct_words.append(corr)
            for t in typos:
                pairs.append((corr, t))
    if not correct_words or not pairs:
        raise ValueError("No training data parsed from aspell.txt.")
    return correct_words, pairs

# train emission and transition probabilities
def align_and_count_emissions(pairs: List[Tuple[str,str]]) -> Dict[str, Counter]:
    counts = {c: Counter() for c in ALPHABET}
    for correct, typo in pairs:
        c = only_letters(correct); t = only_letters(typo)
        if not c or not t: continue
        sm = SequenceMatcher(a=c, b=t, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("equal", "replace"):
                span = min(i2 - i1, j2 - j1)
                for k in range(span):
                    gt, obs = c[i1+k], t[j1+k]
                    if gt in counts and obs in ALPHABET:
                        counts[gt][obs] += 1
    return counts

def count_transitions(correct_words: List[str]) -> Dict[str, Counter]:
    trans = defaultdict(Counter)
    for w in correct_words:
        wl = only_letters(w)
        if not wl: continue
        trans[START][wl[0]] += 1
        for a, b in zip(wl, wl[1:]):
            trans[a][b] += 1
        trans[wl[-1]][END] += 1
    return trans

def normalize_laplace(row: Counter, support: List[str], alpha: float) -> Dict[str, float]:
    total = sum(row.values()) + alpha * len(support)
    return {sym: (row.get(sym, 0) + alpha) / total for sym in support}

def train(alpha_emiss: float = 1.0, alpha_trans: float = 1.0):
    correct_words, pairs = load_aspell()
    emiss_counts = align_and_count_emissions(pairs)
    trans_counts  = count_transitions(correct_words)

    emissions = {s: normalize_laplace(emiss_counts[s], ALPHABET, alpha_emiss) for s in ALPHABET}
    transitions = {}
    next_support = ALPHABET + [END]
    for st in [START] + ALPHABET:
        transitions[st] = normalize_laplace(trans_counts.get(st, Counter()), next_support, alpha_trans)

    # Build lexicon for fallback 
    lexicon = sorted({only_letters(w) for w in correct_words if only_letters(w)})
    return emissions, transitions, lexicon

# fast viterbi for same length 
def viterbi_subst_only(observed: str,
                       emissions: Dict[str, Dict[str, float]],
                       transitions: Dict[str, Dict[str, float]]) -> str:
    if not observed: return observed
    states = ALPHABET
    V = []
    first = observed[0] if observed[0] in ALPHABET else None
    layer0 = {}
    for s in states:
        lp = math.log(transitions[START].get(s, 1e-12)) + math.log(emissions[s].get(first, 1e-12))
        layer0[s] = (lp, None)
    V.append(layer0)
    for t in range(1, len(observed)):
        obs = observed[t] if observed[t] in ALPHABET else None
        layer = {}
        for s in states:
            emit_lp = math.log(emissions[s].get(obs, 1e-12))
            best_lp, best_prev = -1e100, None
            for p in states:
                lp = V[-1][p][0] + math.log(transitions[p].get(s, 1e-12)) + emit_lp
                if lp > best_lp: best_lp, best_prev = lp, p
            layer[s] = (best_lp, best_prev)
        V.append(layer)
    best_lp, best_s = -1e100, None
    for s in states:
        lp = V[-1][s][0] + math.log(transitions[s].get(END, 1e-12))
        if lp > best_lp: best_lp, best_s = lp, s
    path = [best_s]
    for t in range(len(observed)-1, 0, -1):
        path.append(V[t][path[-1]][1])
    path.reverse()
    return "".join(path)

# lexicon-based fallback with alignment
def edit_distance_cap(s: str, t: str, cap: int = 2) -> int:
    """Simple capped Levenshtein to prune candidates quickly."""
    m, n = len(s), len(t)
    if abs(m - n) > cap: return cap + 1
    prev = list(range(n+1))
    for i in range(1, m+1):
        cur = [i] + [0]*n
        row_min = cur[0]
        for j in range(1, n+1):
            cost = 0 if s[i-1] == t[j-1] else 1
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
            row_min = min(row_min, cur[j])
        if row_min > cap: return cap + 1
        prev = cur
    return prev[-1]

# alignment scoring with insertions and deletions
def score_alignment(correct: str, observed: str,
                    emissions: Dict[str, Dict[str, float]],
                    transitions: Dict[str, Dict[str, float]]) -> float:
    g = correct
    o = observed
    Lg, Lo = len(g), len(o)
    NEG = -1e100

    DP = [[NEG]*(Lo+1) for _ in range(Lg+1)]
    DP[0][0] = 0.0 

    # helper to get transition from start to curr
    def trans_lp(prev_c: str, cur_c: str, at_start: bool) -> float:
        if at_start:
            return math.log(transitions[START].get(cur_c, 1e-12))
        return math.log(transitions[prev_c].get(cur_c, 1e-12))

    for i in range(Lg+1):
        for j in range(Lo+1):
            base = DP[i][j]
            #unreachable
            if base <= NEG/2:
                continue
            if j < Lo:
                DP[i][j+1] = max(DP[i][j+1], base + LOG_INS_PEN)

            if i < Lg:
                cur_c = g[i]
                #skip observer
                add_trans = trans_lp(g[i-1] if i>0 else None, cur_c, at_start=(i==0))
                DP[i+1][j] = max(DP[i+1][j], base + LOG_DEL_PEN + add_trans)

                if j < Lo:
                    obs = o[j]
                    emit = math.log(emissions[cur_c].get(obs, 1e-12))
                    DP[i+1][j+1] = max(DP[i+1][j+1], base + add_trans + emit)

    # end transition
    best = -1e100
    for j in range(Lo+1):
        # pay remaining
        rem_ins = (Lo - j) * LOG_INS_PEN
        end_lp = math.log(transitions[g[-1]].get(END, 1e-12)) if Lg > 0 else math.log(transitions[START].get(END, 1e-12))
        cand = DP[Lg][j] + rem_ins + end_lp
        if cand > best: best = cand
    return best

# generate lexicon candidates within edit distance cap
def lexicon_candidates(observed: str, lexicon: List[str], cap: int = 2) -> Iterable[str]:
    obs = observed
    for w in lexicon:
        if w and obs and w[0] != obs[0]:
            continue
        if edit_distance_cap(obs, w, cap) <= cap:
            yield w

def decode_with_fallback(observed: str,
                         emissions: Dict[str, Dict[str, float]],
                         transitions: Dict[str, Dict[str, float]],
                         lexicon: List[str]) -> str:
    #first use same length
    v = viterbi_subst_only(observed, emissions, transitions)
    if v in lexicon:
        return v

    #otherwise use fallback
    best_w, best_lp = None, -1e100
    for cand in lexicon_candidates(observed, lexicon, cap=2):
        lp = score_alignment(cand, observed, emissions, transitions)
        if lp > best_lp:
            best_lp, best_w = lp, cand
    return best_w if best_w else v  # fall back to viterbi result if nothing better

# user facing functions
def fix_word_token(token: str,
                   emissions: Dict[str, Dict[str, float]],
                   transitions: Dict[str, Dict[str, float]],
                   lexicon: List[str]) -> str:
    m = re.match(r"^([A-Za-z]+)(.*)$", token)
    if not m: return token
    letters, tail = m.group(1), m.group(2)
    low = letters.lower()

    # If already a known correct word, return as-is
    if low in lexicon:
        decoded = low
    else:
        decoded = decode_with_fallback(low, emissions, transitions, lexicon)

    if letters[0].isupper():
        decoded = decoded.capitalize()
    return decoded + tail

def fix_text(text: str,
             emissions: Dict[str, Dict[str, float]],
             transitions: Dict[str, Dict[str, float]],
             lexicon: List[str]) -> str:
    return " ".join(fix_word_token(tok, emissions, transitions, lexicon) for tok in text.split())

# main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="Correct a single line and exit.")
    args = parser.parse_args()

    try:
        emissions, transitions, lexicon = train()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    if args.text is not None:
        print(fix_text(args.text, emissions, transitions, lexicon))
        return

    print("Training complete. Enter text to correct or ctrl + c to exit:")
    try:
        while True:
            line = input("> ")
            print(fix_text(line, emissions, transitions, lexicon))
    except KeyboardInterrupt:
        print("\nExiting.")

if __name__ == "__main__":
    main()

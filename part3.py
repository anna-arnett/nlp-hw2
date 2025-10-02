
"""
Goal: Highest-probability parse using a PCFG, with POS tags (from your BiLSTM)
      as the terminal layer.

Inputs:
  - cfg / counts / probs: representations of your PCFG
  - true POS tags or POS tags as the output of a trained BiLSTM POS tagger

Output:
  - For each sentence line from stdin or a file, print one bracketed tree to stdout
    (or an empty line if no parse) with its probability.

Keep these three phases conceptually separate:
  (1) POS tagging + diagonal initialization
  (2) CKY dynamic program over spans (big nested loop)
  (3) Root selection + backpointer reconstruction + printing

You are free to choose your exact data structures, as long as you can:
  - store best scores for labels over spans
  - remember how each best item was built (backpointers)
  - reconstruct a bracketed tree string at the end

-----------------------------------------------------------------------------
0) GRAMMAR + PROBABILITIES + <unk>
-----------------------------------------------------------------------------
* When reading a sequence of POS tags, map each tag to itself if in vocab, 
  or else to a special token like "<unk>" (but keep original for printing).
* After reconstruction, print the original word as the leaf instead.
* Use log probabilities to avoid underflow:
  score = log P(A -> B C) + score(left) + score(right)
"""

import math
import sys
import collections 

import trees
import utils
import part2

def build_pcfg(train_trees_path: str):
    part2.counts.clear()
    with open(train_trees_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            t = trees.Tree.from_str(s)
            if t.root is None:
                continue
            part2.collect_rules_from_tree(t)
    
    probs = part2.compute_probs(part2.counts)

    # convert to log
    probs_log = {
        lhs: {rhs: (math.log(p) if p > 0 else -math.inf) for rhs, p in rhs_map.items()}
        for lhs, rhs_map in probs.items()
    }

    # index binary rules by RHS for CKY
    cfg_by_rhs = collections.defaultdict(dict)
    for lhs, rhs_map in probs.items():
        for rhs, p in rhs_map.items():
            if len(rhs) == 2 and p > 0:
                cfg_by_rhs[rhs][lhs] = math.log(p)
    
    return part2.counts, probs_log, cfg_by_rhs

class CKYParser:
  def __init__(self, probs_log, cfg_by_rhs, start_symbol="TOP"):
      self.probs_log = probs_log
      self.cfg_by_rhs = cfg_by_rhs
      self.start_symbol = start_symbol 

        
  def parse(self, words, pos_tags):
    n = len(words)
    chart = collections.defaultdict(dict)
    backptr = collections.defaultdict(dict)

    # diagonal init
    for i in range(n):
        tag = pos_tags[i]
        rhs = (tag,)
        lp = self.probs_log.get(tag, {}).get(rhs, -math.inf)
        if lp > -math.inf:
            chart[(i, i+1)][tag] = lp
            backptr[(i, i+1)][tag] = ("TERM", words[i])
    
    for span_len in range(2, n+1):
        for i in range(0, n-span_len+1):
            k = i + span_len
            _ = chart[(i, k)]
            _ = backptr[(i, k)]

            for j in range(i+1, k):
                left_cell = chart[(i, j)]
                right_cell = chart[(j, k)]
                if not left_cell or not right_cell:
                    continue 
                
                for L in left_cell.keys():
                    for R in right_cell.keys():
                        rhs = (L, R)
                        if rhs not in self.cfg_by_rhs:
                            continue
                        for A, logp in self.cfg_by_rhs[rhs].items():
                            cand = left_cell[L] + right_cell[R] + logp
                            if A not in chart[(i, k)] or cand > chart[(i, k)][A]:
                                chart[(i, k)][A] = cand
                                backptr[(i, k)][A] = ("BIN", L, j, R)

    def reconstruct(label, i, k):
        bp = backptr[(i, k)][label]
        if bp[0] == "TERM":
            word = bp[1]
            return f"({label} {word})"
        else:
            _, left_label, j, right_label = bp
            left_tree = reconstruct(left_label, i, j)
            right_tree = reconstruct(right_label, j, k)
            return f"({label} {left_tree} {right_tree})"
        
    full_cell = chart.get((0, n), {})
    if self.start_symbol not in full_cell:
        return "", None # no parse
    
    best_logprob = full_cell[self.start_symbol]
    tree_str = reconstruct(self.start_symbol, 0, n)
    return tree_str, best_logprob 
  

if __name__ == "__main__":
    counts, probs_log, cfg_by_rhs = build_pcfg("data/train.trees")
    parser = CKYParser(probs_log, cfg_by_rhs, start_symbol="TOP")

    test_sents = utils.read_pos_file("data/test.pos")
    print("CKY with GOLD POS tags (first 10)")
    for i, sent in enumerate(test_sents[:10], 1):
        words = [w for (w, t) in sent]
        tags =  [t for (w, t) in sent]

        if not tags or tags[-1] != "PUNC":
            words.append(".")
            tags.append("PUNC")

        tree_str, logp = parser.parse(words, tags)
        if tree_str:
            print(tree_str)
            print(f"# logprob: {logp:.4f}")
        else:
            print("")

    # BiLSTM
    import torch
    from part1 import BiLSTMTagger

    ckpt = torch.load("bilstm_pos.pth", map_location="cpu")

    tagger = BiLSTMTagger(data=[], embedding_dim=ckpt["embedding_dim"], hidden_dim=ckpt["hidden_dim"])
    tagger.words = ckpt["words"]
    tagger.tags = ckpt["tags"]

    tagger.emb = torch.nn.Embedding(len(tagger.words), ckpt["embedding_dim"])
    tagger.lstm = torch.nn.LSTM(input_size=ckpt["embedding_dim"], hidden_size=ckpt["hidden_dim"],
                                num_layers=1, bidirectional=True, batch_first=False, dropout=0.0)
    tagger.dropout = torch.nn.Dropout(p=0.2)
    tagger.W_out = torch.nn.Linear(2*ckpt["hidden_dim"], len(tagger.tags))

    tagger.load_state_dict(ckpt["model_state_dict"], strict=True)
    tagger.eval()

    def tag_sentence(model, words):
        idxs = [model.words.numberize(w.lower()) for w in words]
        x = torch.tensor(idxs, dtype=torch.long)
        with torch.no_grad():
            scores = model(x)
            pred_idx = model.predict(scores).tolist()
        return [model.tags.denumberize(j) for j in pred_idx]
    
    print("\nCKY with PREDICTED POS tags (first 10)")
    for i, sent in enumerate(test_sents[:10], 1):
        words = [w for (w, _) in sent]
        pred_tags = tag_sentence(tagger, words)

        if not pred_tags or pred_tags[-1] != "PUNC":
            words.append(".")
            pred_tags.append("PUNC")
        
        tree_str, logp = parser.parse(words, pred_tags)
        if tree_str:
            print(tree_str)
            print(f"# logprob: {logp:.4f}")
        else:
            print("")
  

"""
-----------------------------------------------------------------------------
2) CKY DYNAMIC PROGRAM (THE BIG NESTED LOOP)
-----------------------------------------------------------------------------
The standard CKY fill uses three nested loops over span length, start index,
and split point. Conceptually:

  for span_length in 2..n:
    for i in 0..(n - span_length):
      k = i + span_length
      initialize chart[(i, k)] and backptr[(i, k)] (empty)

      for j in (i+1)..(k-1):   # split index
        # Consider all ways to combine a left piece (i, j) with a right piece (j, k)
        for each left_label in chart[(i, j)]:
          for each right_label in chart[(j, k)]:
            # Check if any rule A -> left_label right_label exists in your PCFG
            for each A with P(A -> left_label right_label):
              candidate_score = chart[(i, j)][left_label] + \
                                chart[(j, k)][right_label] + \
                                log P(A -> left_label right_label)
              if candidate_score is better than current chart[(i, k)][A]:
                  update chart[(i, k)][A] = candidate_score
                  set backptr[(i, k)][A] = (left_label, j, right_label)

Notes:
  - Only binary rules are considered here (CNF).
  - Keep everything in log-space abd use addition rather than multiplication.
"""
"""

-----------------------------------------------------------------------------
3) ROOT SELECTION, RECONSTRUCTION, PRINTING
-----------------------------------------------------------------------------
- After the table is filled, focus on the full span (0, n).
  * Prefer the designated start symbol (e.g., 'TOP') if present at (0, n).
  * If 'TOP' is not present, produce an empty parse.

- Reconstruct the tree via backpointers:
  * Define a recursive function:
      reconstruct(label, i, k):
        bp = backptr[(i, k)][label]
        if bp is a terminal word (or "<unk>"):
            return "(label word_or_original)"
        else:
            (left_label, j, right_label) = bp
            left_subtree  = reconstruct(left_label,  i, j)
            right_subtree = reconstruct(right_label, j, k)
            return f"(label {left_subtree} {right_subtree})"

  * Ensure terminals print the original word here rather than POS tags.

- Output:
  * Print the bracketed tree string for each input sentence (or an empty line
    if no parse), one sentence per line.
  * Print the final score for TOP at (0, n) as a log-prob.

-----------------------------------------------------------------------------
4) PRACTICAL TIPS / DECISIONS (YOU CHOOSE)
-----------------------------------------------------------------------------
- Data structures:
    * dict-of-dicts is fine; you can also use defaultdicts.
    * You can index chart/backptr by tuples (i, k) or use a 2D list.

- Efficiency:
    * Iterate only over labels that actually occur in the subspans.
    * If your PCFG is stored by RHS (B, C) -> {A: prob}, you can quickly find
      candidate parents A for a given pair (B, C).

- Scores:
    * Prefer log-space: add logs instead of multiplying probabilities.

- Debugging:
    * Print the diagonal cells after initialization to verify POS entries.
    * For a tiny sentence (2–3 words), print intermediate chart cells per length.
    * If reconstruction fails, check that backptr entries are actually written
      whenever you write a score.

-----------------------------------------------------------------------------
5) MINIMUM I/O LOOP
-----------------------------------------------------------------------------
for each line from stdin:
  tokens = line.split()
  tags = run_pos_tagger(orig_tokens)

  initialize empty chart/backptr

  # diagonal init
  for i in range(n):
    fill chart[(i, i+1)] and backptr[(i, i+1)] with POS entries

  # CKY nested loops (length, start i, split j) using binary PCFG rules
  fill chart/backptr for spans of length >= 2

  if TOP in chart[(0, n)]:
      tree_str = reconstruct('TOP', 0, n)   # bracketed
      print(tree_str)
      print(logprob_of_TOP_to_stderr)
  else:
      print("")  # empty line if no parse
"""
import trees
import fileinput
import collections
import re
"""You should not need any other imports, but you may import anything that helps."""

counts = collections.defaultdict(collections.Counter)
"""TODO: Collect all the tree branching rules used in the parses in train.trees, and count their frequencies.
	* Use trees.Tree.from_str(), bottomup(), and other helpful functions from trees.py.
	* Goal: end up with three dictionaries, counts, probs, and cfg. 
		* counts has entries count[LHS][RHS], like count[NP][(DT, NN)].
		* probs has entries prob[LHS][RHS] = count[LHS][RHS] / sum(count[LHS].values())
		* cfg simply has the rules of the grammar, stored using whichever structure is usable to you for your CKY implementation. For instance, indexing by RHS may be easier to look up for CKY (cfg[RHS][LHS]."""

def add_rule(lhs, rhs_tuple):
    counts[lhs][rhs_tuple] += 1
    
def pos_from_label(lbl: str) -> str:
    if '_' in lbl:
        return lbl.split('_')[-1]
    return lbl 

def collect_rules_from_tree(t: trees.Tree):
    def is_preterminal(node):
        return len(node.children)==1 and len(node.children[0].children) == 0
    
    for node in t.bottomup():
        # skip words
        if len(node.children) == 0:
            continue 
        
		# if all children are leaves 
        if all(len(ch.children) == 0 for ch in node.children):
            pos = pos_from_label(node.label)
            add_rule(pos, (pos,))
            continue 
        

        lhs = node.label
        rhs = []
        for ch in node.children:
            if is_preterminal(ch):
                rhs.append(pos_from_label(ch.label))
            else:
                rhs.append(ch.label)
        add_rule(lhs, tuple(rhs))

def compute_probs(counts_dict):
    probs = {}
    for lhs, ctr in counts_dict.items():
        total = sum(ctr.values())
        probs[lhs] = {rhs: (c / total if total else 0.0) for rhs, c in ctr.items()}
    return probs 

def build_cfg_index_by_rhs(probs_dict):
    cfg = collections.defaultdict(dict)
    for lhs, rhs_map in probs_dict.items():
        for rhs, p in rhs_map.items():
            cfg[rhs][lhs] = p
    return cfg 

def print_pcfg(probs_dict):
    for lhs in sorted(probs_dict):
        for rhs in sorted(probs_dict[lhs]):
            rhs_str = " ".join(rhs)
            print(f"{lhs} -> {rhs_str} # {probs_dict[lhs][rhs]:.4f}")
            
def summarize(counts_dict, probs_dict):
    # 1) number of unique rules 
    unique_rules = sum(len(ctr) for ctr in counts_dict.values())
    print(f"\nUnique rules: {unique_rules}")
    
	# 2) top 5 most frequent rules overall
    all_rules = []
    for lhs, ctr in counts_dict.items():
        for rhs, c in ctr.items():
            all_rules.append((c, lhs, rhs))
    all_rules.sort(key=lambda x: (-x[0], x[1], x[2]))
    print("\nTop 5 most frequent rules:")
    for c, lhs, rhs in all_rules[:5]:
        print(f"{lhs} -> {' '.join(rhs)} # {c}")
    
	# 3) top five highest-prob NP rules 
    if 'NP' in probs_dict:
        np_rules = sorted(probs_dict['NP'].items(), key=lambda kv: (-kv[1], kv[0]))
        print("\nTop 5 NP rules by probability:")
        for rhs, p in np_rules[:5]:
            print(f"NP -> {' '.join(rhs)} # {p:.4f}")
    else:
        print("\nTop 5 NP rules by probability: (no NP rules found)")
        
if __name__ == "__main__":
    for line in fileinput.input():
        s = line.strip()
        if not s:
            continue
        t = trees.Tree.from_str(s)
        if t.root is None:
            continue
        collect_rules_from_tree(t)
        
    probs = compute_probs(counts)
    cfg = build_cfg_index_by_rhs(probs)
    print_pcfg(probs)
    summarize(counts, probs)
    
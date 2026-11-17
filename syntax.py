"""Ordinary positive-weight CFG enumeration; lexical feature fields are metadata."""
from functools import lru_cache

def parse(grammar,candidates):
    rules=grammar["rules"]
    @lru_cache(None)
    def visit(cat,left,right,unary_seen=()):
        out=[]
        if right-left==1:
            for c in candidates[left]:
                if c["cat"]==cat:out.append((c["weight"],{"lexical_id":c["id"],"cat":cat}))
        for r in rules:
            if r["lhs"]["cat"]!=cat:continue
            rhs=r["rhs"]
            if len(rhs)==1 and rhs[0]["cat"] not in unary_seen:
                for w,t in visit(rhs[0]["cat"],left,right,unary_seen+(cat,)):
                    out.append((r["weight"]*w,{"rule":r["id"],"children":[t]}))
            elif len(rhs)==2:
                for split in range(left+1,right):
                    for w1,t1 in visit(rhs[0]["cat"],left,split,()):
                        for w2,t2 in visit(rhs[1]["cat"],split,right,()):
                            out.append((r["weight"]*w1*w2,{"rule":r["id"],"children":[t1,t2]}))
        return tuple(out)
    trees=visit(grammar["start"]["cat"],0,len(candidates))
    return dict(trees=[{"weight":w,"tree":t} for w,t in trees],total_weight=sum(w for w,t in trees))

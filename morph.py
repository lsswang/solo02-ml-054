"""A small ordinary lexicon with direct forms and simple plural suffixes."""
class Morphology:
    def __init__(self,entries):self.entries=list(entries)
    def analyze(self,word):
        out=[]
        for e in self.entries:
            if e["stem"]==word:
                out.append(dict(id=e["id"]+":base",entry_id=e["id"],surface=word,cat=e["cat"],features=dict(e["features"]),weight=e["weight"],path=[]))
            if e["cat"]=="N" and e["stem"]+"s"==word:
                f=dict(e["features"]);f["number"]="pl"
                out.append(dict(id=e["id"]+":plural-s",entry_id=e["id"],surface=word,cat=e["cat"],features=f,weight=e["weight"],path=["plural-s"]))
        return out
    def generate(self,entry_id,features):
        entry=next(e for e in self.entries if e["id"]==entry_id)
        text=entry["stem"]+("s" if entry["cat"]=="N" and features.get("number")=="pl" else "")
        return [text]

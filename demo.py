import json
from pathlib import Path
from morph import Morphology
from syntax import parse
if __name__=="__main__":
    x=json.loads((Path(__file__).parent/"fixtures/cases.json").read_text())
    model=Morphology(x["before_entries"])
    rows=[model.analyze(w) for w in x["ordinary_sentence"]]
    print(json.dumps({"generated":model.generate("cat",{"number":"pl"}),"analyses":rows,"parse":parse(x["ordinary_grammar"],rows)},indent=2))

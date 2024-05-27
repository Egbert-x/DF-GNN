import spacy
import scispacy
from scispacy.linking import EntityLinker
import json
from tqdm import tqdm

repo_root = '.'
medqa_root = f'{repo_root}/data/medmcqa'

def load_entity_linker(threshold=0.90):
    nlp = spacy.load("en_core_sci_md")
    # linker = EntityLinker(
    #     resolve_abbreviations=True,
    #     name="umls",
    #     threshold=threshold)
    # nlp.add_pipe(linker)
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    linker = nlp.get_pipe("scispacy_linker")
    return nlp, linker

nlp, linker = load_entity_linker()

def entity_linking_to_umls(sentence, nlp, linker):
    doc = nlp(sentence)
    entities = doc.ents
    all_entities_results = []
    for mm in range(len(entities)):
        entity_text = entities[mm].text
        entity_start = entities[mm].start
        entity_end = entities[mm].end
        all_linked_entities = entities[mm]._.kb_ents
        all_entity_results = []
        for ii in range(len(all_linked_entities)):
            curr_concept_id = all_linked_entities[ii][0]
            curr_score = all_linked_entities[ii][1]
            curr_scispacy_entity = linker.kb.cui_to_entity[all_linked_entities[ii][0]]
            curr_canonical_name = curr_scispacy_entity.canonical_name
            curr_TUIs = curr_scispacy_entity.types
            curr_entity_result = {"Canonical Name": curr_canonical_name, "Concept ID": curr_concept_id,
                                  "TUIs": curr_TUIs, "Score": curr_score}
            all_entity_results.append(curr_entity_result)
        curr_entities_result = {"text": entity_text, "start": entity_start, "end": entity_end,
                                "start_char": entities[mm].start_char, "end_char": entities[mm].end_char,
                                "linking_results": all_entity_results}
        all_entities_results.append(curr_entities_result)
    return all_entities_results
# sent = "Which of the following statements concerning positive predictive value (PPV) is correct?"
# ent_link_results = entity_linking_to_umls(sent, nlp, linker)
# print(ent_link_results)
def process(input):
    nlp, linker = load_entity_linker()
    stmts = input
    for stmt in tqdm(stmts):
        stem = stmt['question']['stem']
        stem = stem[:3500]
        stmt['question']['stem_ents'] = entity_linking_to_umls(stem, nlp, linker)
        for ii, choice in enumerate(stmt['question']['choices']):
            text = stmt['question']['choices'][ii]['text']
            stmt['question']['choices'][ii]['text_ents'] = entity_linking_to_umls(text, nlp, linker)
    return stmts

for fname in ["dev", "test", "train"]:
    with open(f"{medqa_root}/statement/{fname}.statement_new.jsonl") as fin:
        stmts = [json.loads(line) for line in fin]
        res = process(stmts)
    with open(f"{medqa_root}/statement/{fname}.statement_new.medmcqa_linked_md_051.jsonl", 'w') as fout:
        for dic in res:
            print (json.dumps(dic), file=fout)
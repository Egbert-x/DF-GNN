import csv
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# 构造ground数据集
repo_root = '.'
medqa_root = f'{repo_root}/data/medmcqa'
umls_to_ddb = {}
with open("../data/semmedVER43_2022_R_PREDICATION.csv", "r", encoding="gb18030", errors="ignore") as f:
    for row in csv.reader(f, skipinitialspace=True):
        #print(row)
        #print(row[4], row[5], row[8], row[9])
        # if row[5] == "Positive Predictive Value of Diagnostic Test" or row[9] =="Positive Predictive Value of Diagnostic Test":
        #     print("Positive Predictive Value of Diagnostic Test find!")
        umls_to_ddb[row[4]] = row[5]
        umls_to_ddb[row[8]] = row[9]
def map_to_ddb(ent_obj):
    res = []
    ignore = []
    for ent_cand in ent_obj['linking_results']:
        CUI  = ent_cand['Concept ID']
        name = ent_cand['Canonical Name']
        if CUI in umls_to_ddb and name == umls_to_ddb[CUI]:
            ddb_cid = CUI
            res.append((ddb_cid, name))
            # if CUI == "C1514243" and name == "Positive Predictive Value of Diagnostic Test":
            #     print("Positive Predictive Value of Diagnostic Test find!")
        else:
            ignore.append((CUI, name))
    return res

def map_to_ddb_ignore(i, cid, ent_obj):
    res = []
    ignore = []
    for ent_cand in ent_obj['linking_results']:
        CUI  = ent_cand['Concept ID']
        name = ent_cand['Canonical Name']
        if CUI in umls_to_ddb and name == umls_to_ddb[CUI]:
            ddb_cid = CUI
            res.append((ddb_cid, name))
            # if CUI == "C1514243" and name == "Positive Predictive Value of Diagnostic Test":
            #     print("Positive Predictive Value of Diagnostic Test find!")
        else:
            ignore.append((i, cid, CUI, name))
    with open(f"{repo_root}/data/ddb/sem_ignore_md_ac_051.txt", "a") as fout:
        for sem_name in ignore:
            print(sem_name, file=fout)

    return res

def process(fname):
    with open(f"{medqa_root}/statement/{fname}.statement_new.medmcqa_linked_md_051.jsonl") as fin:
        stmts = [json.loads(line) for line in fin]
    with open(f"{medqa_root}/grounded/{fname}.grounded_md_051_new.jsonl", 'w') as fout:
        i = 1
        for stmt in tqdm(stmts):
            sent = stmt['question']['stem']
            qc = []
            qc_names = []
            for ent_obj in stmt['question']['stem_ents']:
                res = map_to_ddb(ent_obj)
                for elm in res:
                    ddb_cid, name = elm
                    qc.append(ddb_cid)
                    qc_names.append(name)
            for cid, choice in enumerate(stmt['question']['choices']):
                ans = choice['text']
                ac = []
                ac_names = []
                for ent_obj in choice['text_ents']:
                    # res = map_to_ddb_ignore(i, cid, ent_obj)
                    res = map_to_ddb(ent_obj)
                    for elm in res:
                        ddb_cid, name = elm
                        ac.append(ddb_cid)
                        ac_names.append(name)
                out = {'sent': sent, 'ans': ans, 'qc': qc, 'qc_names': qc_names, 'ac': ac, 'ac_names': ac_names}
                print (json.dumps(out), file=fout)
            i+=1

os.system(f'mkdir -p {medqa_root}/grounded')
for fname in ["dev", "test", "train"]:
    process(fname)
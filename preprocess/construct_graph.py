import json
from tqdm import tqdm
import csv
import json
import configparser
import time
import networkx as nx

# 分离多个cui
def separate_semmed_cui(semmed_cui: str) -> list:
    """
    separate semmed cui with | by perserving the replace the numbers after |
    `param`:
        semmed_cui: single or multiple semmed_cui separated by |
    `return`:
        sep_cui_list: list of all separated semmed_cui
    """
    sep_cui_list = []
    sep = semmed_cui.split("|")
    first_cui = sep[0]
    sep_cui_list.append(first_cui)
    ncui = len(sep)
    for i in range(ncui - 1):
        last_digs = sep[i + 1]
        len_digs = len(last_digs)
        if len_digs < 8:
            sep_cui = first_cui[:8 - len(last_digs)] + last_digs
            sep_cui_list.append(sep_cui)
    return sep_cui_list

if __name__ == '__main__':
    repo_root = '.'
    merged_relations = ["process_of", "affects", "augments", "causes", "diagnoses", "interacts_with", "part_of", "precedes", "predisposes", "produces", "isa"]
    with open("./data/ddb/sem_cuis.txt", "r", encoding="gbk") as fin:
        idx2cui = [c.strip() for c in fin]
    cui2idx = {c: i for i, c in enumerate(idx2cui)}
    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}
    print(id2relation)
    # 创建有向多图
    print("generating graph of SemMed using newly extracted cui list...")
    graph = nx.MultiDiGraph()
    nrow = sum(1 for _ in open("../data/semmedVER43_2022_R_PREDICATION.csv", "r", encoding="gb18030", errors="ignore")) # 列数
    with open("../data/semmedVER43_2022_R_PREDICATION.csv", "r", encoding="gb18030", errors="ignore") as f:
        attrs = set()
        for row in csv.reader(f, skipinitialspace=True):
            # print(row[3].lower())
            # print(row[4])
            # print(row[8])
            # time.sleep(5)
            if row[3].lower() not in id2relation: # ls[3]为relationship
                #print("not in")
                continue
            # ls[4]、ls[8]是头实体和尾实体
            if row[4] == row[8]:
                #print("invalid")
                continue
            weight = 1.
            rel = relation2id[row[3].lower()]  # 转成id形式
            if row[4].startswith("C") and row[8].startswith("C"):
                if len(row[4]) == 8 and len(row[8]) == 8:
                    # 是否在词汇列表中
                    if row[4] in idx2cui and row[8] in idx2cui:
                        subj = cui2idx[row[4]]
                        obj = cui2idx[row[8]]
                        if (subj, obj, rel) not in attrs:
                            # print("type1 add")
                            graph.add_edge(subj, obj, rel=rel, weight=weight)
                            attrs.add((subj, obj, rel))
                            graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                            attrs.add((obj, subj, rel + len(relation2id)))
                # subj可能有多个cui，分隔开
                elif len(row[4]) != 8 and len(row[8]) == 8:
                    cui_list = separate_semmed_cui(row[4]) # 分离中间用"|"隔开的cui
                    subj_list = [cui2idx[s] for s in cui_list if s in idx2cui] # 同上，以id表示
                    if row[8] in idx2cui:
                        obj = cui2idx[row[8]]
                        for subj in subj_list:
                            if (subj, obj, rel) not in attrs:
                                # print("type2 add")
                                graph.add_edge(subj, obj, rel=rel, weight=weight)
                                attrs.add((subj, obj, rel))
                                graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                                attrs.add((obj, subj, rel + len(relation2id)))
                # 同上处理
                elif len(row[4]) == 8 and len(row[8]) != 8:
                    cui_list = separate_semmed_cui(row[8])
                    obj_list = [cui2idx[o] for o in cui_list if o in idx2cui]
                    if row[4] in idx2cui:
                        subj = cui2idx[row[4]]
                        for obj in obj_list:
                            if (subj, obj, rel) not in attrs:
                                # print("type3 add")
                                graph.add_edge(subj, obj, rel=rel, weight=weight)
                                attrs.add((subj, obj, rel))
                                graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                                attrs.add((obj, subj, rel + len(relation2id)))
                # 同上处理
                else:
                    cui_list1 = separate_semmed_cui(row[4])
                    subj_list = [cui2idx[s] for s in cui_list1 if s in idx2cui]
                    cui_list2 = separate_semmed_cui(row[8])
                    obj_list = [cui2idx[o] for o in cui_list2 if o in idx2cui]
                    for subj in subj_list:
                        for obj in obj_list:
                            if (subj, obj, rel) not in attrs:
                                # print("type4 add")
                                graph.add_edge(subj, obj, rel=rel, weight=weight)
                                attrs.add((subj, obj, rel))
                                graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                                attrs.add((obj, subj, rel + len(relation2id)))

    output_path = f"{repo_root}/data/ddb/sem.graph"
    nx.write_gpickle(graph, output_path)
    print(len(attrs))
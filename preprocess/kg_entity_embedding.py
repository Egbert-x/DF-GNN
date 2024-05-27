import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig

if __name__ == '__main__':
    repo_root = '.'
    # tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    # bert_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")
    bert_model = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-large")
    # tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")
    # bert_model = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-base")
    device = torch.device('cuda:1')
    bert_model.to(device)
    bert_model.eval()
    with open(f"{repo_root}/data/ddb/sem_vocab_800000.txt") as f:
        names = [line.strip() for line in f]
    embs = []
    tensors = tokenizer(names, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        for i, j in enumerate(tqdm(names)):
            outputs = bert_model(input_ids=tensors["input_ids"][i:i + 1].to(device),
                                 attention_mask=tensors['attention_mask'][i:i + 1].to(device))
            out = np.array(outputs[1].squeeze().tolist()).reshape((1, -1))
            embs.append(out)
    embs = np.concatenate(embs)
    np.save(f"{repo_root}/data/ddb/sem_ent_emb_biolink_large_800000.npy", embs)
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(
            "line-corporation/line-distilbert-base-japanese",
            trust_remote_code=True,
        )
model = AutoModel.from_pretrained("line-corporation/line-distilbert-base-japanese")

input_ids = torch.tensor(
        [
            tokenizer.encode(
                "This is an example sentence",
                add_special_tokens=True
            )
        ]
    )

with torch.no_grad():
    last_hidden_states = model(input_ids)[0]

embedding = last_hidden_states[0][0]

print(embedding)
print(embedding.size())
print(embedding.shape)
print(input_ids)
reshape_embedding = torch.reshape(embedding, (1, embedding.size()[0]))

import faiss
index = faiss.IndexFlatIP(768)
index.add(reshape_embedding)
D, I = index.search(reshape_embedding, 1)
print(D)
print(I)

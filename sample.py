import torch
from transformers import AutoTokenizer, AutoModel

# sample
# from transformers import pipeline
# unmasker = pipeline(
#         'fill-mask',
#         model='line-corporation/line-distilbert-base-japanese', 
#         trust_remote_code=True
#         )
# print(unmasker("LINE株式会社で[MASK]の研究・開発をしている。"))

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

embeddings = last_hidden_states[0]

print(embeddings)
print(embeddings.size())
print(input_ids)

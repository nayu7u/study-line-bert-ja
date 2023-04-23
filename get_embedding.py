import torch
from transformers import AutoTokenizer, AutoModel

class GetEmbedding:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
                    "line-corporation/line-distilbert-base-japanese",
                    trust_remote_code=True,
                )
        self.model = AutoModel.from_pretrained("line-corporation/line-distilbert-base-japanese")

    def exec(self, sentence):
        input_ids = torch.tensor(
                        [ self.tokenizer.encode( sentence, add_special_tokens=True) ]
                    )
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
        embedding = last_hidden_states[0][0]
        return torch.reshape(embedding, (1, embedding.size()[0]))

if __name__ == "__main__":

    # faiss test
    import faiss
    index = faiss.IndexFlatIP(768)

    get_embedding = GetEmbedding()
    sentence_1 = get_embedding.exec("これは例文です")
    sentence_2 = get_embedding.exec("これはサンプルの文章です")
    sentence_3 = get_embedding.exec("これは本番用の文章です")
    sentence_4 = get_embedding.exec("これは実際に使用された文章です")

    index.add(sentence_1)
    index.add(sentence_2)
    index.add(sentence_3)
    index.add(sentence_4)

    D, I = index.search(sentence_4, 4)
    print(D)
    print(I)

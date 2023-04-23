from transformers import pipeline
unmasker = pipeline(
        'fill-mask',
        model='line-corporation/line-distilbert-base-japanese', 
        trust_remote_code=True
        )
print(unmasker("LINE株式会社で[MASK]の研究・開発をしている。"))


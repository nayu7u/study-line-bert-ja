FROM pytorch/pytorch
RUN pip install transformers ipadic fugashi unidic_lite unidic rhoknp sudachipy sentencepiece PyGithub

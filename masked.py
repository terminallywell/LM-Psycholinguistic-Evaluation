from math import log2
from transformers import AutoTokenizer, pipeline, FillMaskPipeline


def prob_mask(fill_masker: FillMaskPipeline, context: str, word: str, surprisal: bool = False) -> float:
    '''Probability of a masked word/token given context'''
    score = fill_masker(context, targets=' ' + word)[0]['score']

    if surprisal:
        return -log2(score)
    return score

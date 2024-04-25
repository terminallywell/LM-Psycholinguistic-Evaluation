import torch
from math import exp2, log2
from transformers import PreTrainedTokenizerBase, PreTrainedModel


def prob_next(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, context: str, word: str, surprisal: bool = False) -> float:
    '''Probability of a word/token given preceding context.'''
    input_ids = tokenizer.encode(context, return_tensors='pt')
    word_id = tokenizer.encode(' ' + word, add_special_tokens=False)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    next_probs = logits[0, -1].softmax(dim=0)
    prob = next_probs[word_id].item()
    
    if surprisal:
        return -log2(prob)
    return prob


def prob_whole(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, sentence: str, surprisal: bool = False) -> float:
    '''Probability of a whole sentence as a product of probabilities of each token in it.'''
    input_ids = tokenizer.encode('\n' + sentence, return_tensors='pt') # arbitrarily chose \n as start token
    n = input_ids.size(dim=1) - 1
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits

    logp = 0
    for i in range(n):
        word_id = input_ids[0][i + 1]
        next_probs = logits[0, i].softmax(dim=0)
        prob = next_probs[word_id]
        logp += log2(prob)

    logp /= n # normalizer: keep or not?

    if surprisal:
        return -logp
    return exp2(logp)

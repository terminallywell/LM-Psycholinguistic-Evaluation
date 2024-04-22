# To delete cached models, run command: huggingface-cli delete-cache

import torch
from math import log2
from transformers import AutoTokenizer, AutoModelForCausalLM

# save your huggingface token as token.txt in your folder
with open('token.txt', encoding='utf8') as file:
    TOKEN = file.read().strip()


causal_models = [
    'openai-community/gpt2',
    'openai-community/gpt2-medium',
    'openai-community/gpt2-large',
    'facebook/opt-350m',
    'google/gemma-2b'
]

MODEL_NAME = causal_models[4]

tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)
model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=TOKEN)


def prob_next(context: str, word: str, surprisal: bool = False) -> float:
    input_ids = tokenizer.encode(context, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=1)[0]
    word_id = tokenizer.encode(' ' + word, add_special_tokens=False)
    prob = probs[word_id].item()

    # print(tokenizer.decode(torch.argmax(probs)), probs[torch.argmax(probs)].item())

    if surprisal:
        return -log2(prob)
    return prob


# tests
prob_next('Hello! My name', 'is')


principle_b = [
    'Before offering him a fancy pastry,', # constraint
    'Before offering her a fancy pastry,',
    'Before anyone offered him a fancy pastry,', # no constraint
    'Before anyone offered her a fancy pastry,',
]

sbj_obj = [
    'While he was taking orders,',
    'While she was taking orders,',
    'While he was taking orders, a couple of customers annoyed',
    'While she was taking orders, a couple of customers annoyed',
]

contexts = principle_b
word = 'Michael'

for context in contexts:
    surprisal = prob_next(context, word, True)
    print(f'Surprisal of "{word}" in "{context} {word} ...": {surprisal:.3f}')

abs(prob_next(contexts[0], word) - prob_next(contexts[1], word))
abs(prob_next(contexts[2], word) - prob_next(contexts[3], word))

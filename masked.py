# To delete cached models, run command: huggingface-cli delete-cache


from math import log2
from transformers import AutoTokenizer, pipeline

# save your huggingface token as token.txt in your folder
with open('token.txt', encoding='utf8') as file:
    TOKEN = file.read().strip()


masked_models = [
    'FacebookAI/roberta-large',
    'FacebookAI/roberta-base',
    'distilbert/distilbert-base-uncased',
    'google-bert/bert-base-uncased',
]

MODEL_NAME = masked_models[0]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)
mask = tokenizer.mask_token
fill_masker = pipeline(model=MODEL_NAME, token=TOKEN)


def prob_masked(context: str, word: str, surprisal: bool = False) -> float:
    score = fill_masker(context, targets=' ' + word)[0]['score']

    if surprisal:
        return -log2(score)
    return score


# tests
prob_masked(f'The key to the cabinet {mask} rusty from many years of disuse.', 'was')
prob_masked(f'The key to the cabinet {mask} rusty from many years of disuse.', 'were')
prob_masked(f'The key to the cabinets {mask} rusty from many years of disuse.', 'was')
prob_masked(f'The key to the cabinets {mask} rusty from many years of disuse.', 'were')
prob_masked(f'The keys to the cabinets {mask} rusty from many years of disuse.', 'was')
prob_masked(f'The keys to the cabinets {mask} rusty from many years of disuse.', 'were')


prob_masked(f'Give a man a fish, and you feed {mask} for a day.', 'him', False)
prob_masked(f'Give a man a fish, and you feed {mask} for a day.', 'her', False)
prob_masked(f'Give a woman a fish, and you feed {mask} for a day.', 'him', False)
prob_masked(f'Give a woman a fish, and you feed {mask} for a day.', 'her', False)

prob_masked(f'What do you {mask} of my new dress?.', 'think', False)

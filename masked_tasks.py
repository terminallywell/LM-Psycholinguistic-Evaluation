# To delete cached models, run command: huggingface-cli delete-cache

from transformers import AutoTokenizer, pipeline
from masked import prob_mask

# save your huggingface token as token.txt in your folder
with open('token.txt', encoding='utf8') as file:
    TOKEN = file.read().strip()

# list of hugging face models
masked_models = [
    'FacebookAI/roberta-large',
    'FacebookAI/roberta-base',
    'distilbert/distilbert-base-uncased',
    'google-bert/bert-base-uncased',
]

MODEL_NAME = masked_models[1]

# initializing tokenizer and pipeline
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)
MASK = tokenizer.mask_token
fill_masker = pipeline(model=MODEL_NAME, token=TOKEN)



# tests
prob_mask(fill_masker, f'Hi, my name is {MASK}. Nice to meet you.', 'Alex')


for inter in ['cabinet', 'cabinets']:
    for verb in ['was', 'were']:
        context = f'The key to the {inter} {MASK} rusty from many years of disuse.'
        surprisal = prob_mask(fill_masker, context, verb, True)
        print(f'Surprisal of "{verb}" in "{context}": {surprisal:.3f}')


prob_mask(fill_masker, f'Give a man a fish, and you feed {MASK} for a day.', 'him', False)
prob_mask(fill_masker, f'Give a man a fish, and you feed {MASK} for a day.', 'her', False)
prob_mask(fill_masker, f'Give a woman a fish, and you feed {MASK} for a day.', 'him', False)
prob_mask(fill_masker, f'Give a woman a fish, and you feed {MASK} for a day.', 'her', False)

prob_mask(fill_masker, f'What do you {MASK} of my new dress?.', 'think', False)

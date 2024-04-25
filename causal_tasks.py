# To delete cached models, run command: huggingface-cli delete-cache

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from causal import prob_next, prob_whole

# save your huggingface token as token.txt in your folder
with open('token.txt', encoding='utf8') as file:
    TOKEN = file.read().strip()

# list of hugging face models
causal_models = [
    'openai-community/gpt2',
    'openai-community/gpt2-medium',
    'openai-community/gpt2-large',
    'facebook/opt-350m',
    'google/gemma-2b',
]

MODEL_NAME = causal_models[3]

# initializing tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=TOKEN)
model.eval()


# next-word tests
def surp_next(context, word):
    return prob_next(tokenizer, model, context, word, True)


prob_next(tokenizer, model, 'Hi, my name', 'is')

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

for context1 in contexts:
    surprisal = surp_next(context1, word)
    print(f'Surprisal of "{word}" in "{context1} {word} ...": {surprisal:.3f}')


# file reading
task = pd.read_csv('Tasks/PrincipleB_sample.csv', index_col=0)
task['Surprisal'] = task.apply(lambda row: surp_next(row['Context'], row['Target']), axis=1)
task.to_csv(f'Tasks/PrincipleB_sample_{MODEL_NAME.split('/')[1]}.csv')

# whole-sentence tests
prob_whole(tokenizer, model, 'He can eat the bread.', True)
prob_whole(tokenizer, model, 'This is a longer sentence, but it makes more sense than the other one.', True)
prob_whole(tokenizer, model, "Sarah is a great person, but for some reason, a lot of people seem to really dislike her.", True)
prob_whole(tokenizer, model, "Sarah is a great person, but for some reason, a lot of people seem to really dislike him.", True)

sentences = [
    'Whose DNA did the report claim that matched the sample?',
    'Whose DNA did the report claim matched the sample?',
    'Whose DNA did the report claim that the sample matched?',
    'Whose DNA did the report claim the sample matched?',
]

for sen in sentences:
    prob_whole(tokenizer, model, sen, True)

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

MODEL_NAME = causal_models[4]

# initializing tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=TOKEN)
model.eval()


# next-word surprisal tasks
def surp_next(context, word):
    return prob_next(tokenizer, model, context, word, True)

task_list = [
    'PrincipleB',
    'Sbj-Obj'
]

TASK = task_list[1]


###############################
# execute task & record results
task = pd.read_csv(f'tasks/{TASK}.csv', index_col=0)
task['Surprisal'] = task.apply(lambda row: surp_next(row['Context'], row['Target']), axis=1)
task.to_csv(f'results/{TASK}_{MODEL_NAME.split('/')[1]}.csv')

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

task_list = [
    'principle-b',
    'sbj-obj'
]

TASK = task_list[0]
data = pd.read_csv(f'tasks/{TASK}/{TASK}.csv')


def surp_next(context, word):
    return prob_next(tokenizer, model, context, word, True)

###############################
if __name__ == '__main__':
    for modelname in causal_models:
        # initializing tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(modelname, token=TOKEN)
        model = AutoModelForCausalLM.from_pretrained(modelname, token=TOKEN)
        model.eval()

        # execute task & record results
        data['Surprisal'] = data.apply(lambda row: surp_next(row['Context'], row['Target']), axis=1)
        data.to_csv(f'tasks/{TASK}/results/{TASK}_{modelname.split('/')[1]}.csv', index=False)
        print('Finished:', modelname)

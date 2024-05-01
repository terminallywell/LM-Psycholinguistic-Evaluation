# To delete cached models, run command: huggingface-cli delete-cache

import pandas as pd
from transformers import AutoTokenizer, pipeline, DebertaTokenizer
from masked import prob_mask

# save your huggingface token as token.txt in your folder
with open('token.txt', encoding='utf8') as file:
    TOKEN = file.read().strip()

# list of hugging face models
masked_models = [
    'distilbert/distilbert-base-uncased',
    'google-bert/bert-base-uncased',
    'FacebookAI/roberta-base',
    'FacebookAI/roberta-large',
]


task_list = [
    'Attraction',
]


TASK = task_list[0]
data = pd.read_csv(f'tasks/{TASK}.csv')


def surp_mask(context, word):
    return prob_mask(fill_masker, context, word, True)

###############################
if __name__ == '__main__':
    for modelname in masked_models:
        # initializing tokenizer and pipeline
        tokenizer = AutoTokenizer.from_pretrained(modelname, token=TOKEN)
        MASK = tokenizer.mask_token
        fill_masker = pipeline(model=modelname, token=TOKEN)

        # execute task & record results
        data['Surprisal'] = data.apply(lambda row: surp_mask(row['Context'].format(MASK), row['Target']), axis=1)
        data.to_csv(f'results/{TASK}_{modelname.split('/')[1]}.csv', index=False)
        print('Finished:', modelname)

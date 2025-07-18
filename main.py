import os
import torch
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from config import bnbConfig, LoRAConfig, trainerConfig
from preprocessing_FT import loadnSampling, preprocessing


###################################################################################################

#path info
MODEL_NAME = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
OUTPUT_DIR = "./ouput"                                          
LORA_DIR = "./LoRA"                   
DATASET_PATH = "./data/AI_hub_conversation_data(session).xlsx"
FILE_VER = 'session'

#prompt
PROMPT = '''
You are a competent translator. Translate the following english dialogue into Korean.
You should output the translation result of the input data. Do not include any other contents.'''   #zero-shot


####################################################################################################


#Model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = bnbConfig,
    device_map="auto",
    torch_dtype=torch.float16,
)

#tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

#LoRA
model = prepare_model_for_kbit_training(base_model)
model = get_peft_model(model, LoRAConfig)  



#FT_data loading & sampling
num_samples = 3000  
dataset = loadnSampling(DATASET_PATH, num_samples=num_samples)

#FT_preprocessing
dataset = Dataset.from_pandas(dataset)

tokenized_dataset = dataset.map(
    lambda x: preprocessing(x, prompt = PROMPT, tokenizer=tokenizer),
    batched=True,
    remove_columns=dataset.column_names
)

#FT_trainer initialization
trainer = SFTTrainer(
    model=model,
    args=trainerConfig,
    train_dataset=tokenized_dataset,
)

#FT
trainer.train()
trainer.model.save_pretrained(LORA_DIR)


#######################################################################################################

model = PeftModel.from_pretrained(
    base_model,
    LORA_DIR,
    device_map='auto'
)

##########################################Implementation###############################################


def translation(context):
    with torch.no_grad():
        prompt = f"Translate the following English text to Korean. \n\nEnglish: {context}\n\nKorean: "

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
                **inputs,
                max_new_tokens=512,
                top_p=0.9,
                top_k=50,
                temperature=0.3,
                #repitition_penalty=1.2,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
                )

        full_output = tokenizer.decode(output[0], skip_special_tokens=True)

        # "Korean:" 이후의 텍스트만 추출
        korean_prefix = "Korean:"
        korean_start = full_output.find(korean_prefix)
        if korean_start != -1:
            return full_output[korean_start + len(korean_prefix):].strip()

        return full_output[len(prompt):].strip()
    


#test_samples
test_inp1 = '''
A : How much are they?
B : They are 3$ each.
A : Too expansive... Any discount?
B : No, they are already the cheapest in this town.
A : Okay, then I'll take it.
'''

test_inp2 ='''
B : Excuse me. Where is the history Museum?
G : Its on the green street.
B : How do I get there?
G : Go straight three blocks and turn right.
B : Thank you so much.
'''

test_inp3 ='''
I: Professor, I'm so sorry, but I completely forgot about the exam date. Is there anything I can do?
J: Well, you forgot? How did that happen?
I: I was sick last week, and I just haven't been myself for the last couple of weeks.
J: Hmm... OK, I'll give you another chance this time. You can either take a make-up exam or write a research paper.
I: Thank you very much, professor. I'll take the make-up exam.
'''


input_li = [test_inp1, test_inp2, test_inp3]
gen_li = []


#print
print("\n파인튜닝 완료 후 테스트 :")

for index, test_input in enumerate(input_li):
  
  gen = translation(test_input)
  gen_li.append(gen)

  print(f"#{index}")
  print(f"input : {test_input}")
  print(f"translation result : {gen}")
  print('\n'*2)


#result in excel format
res_dict = {}
res_dict['en'] = input_li
res_dict['ko'] = gen_li

result = pd.DataFrame(res_dict)
result.to_excel(f'DRE_translation_result({FILE_VER}).xlsx')

import os
import torch
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

###
from config import bnbConfig, LoRAConfig, trainerConfig
from preprocessing_FT import loadnSampling, preprocessing


#basic info
MODEL_NAME = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
OUTPUT_DIR = "./ouput"
LORA_DIR = "./LoRA"                    #./ 현재 폴더를 나타낸다, .는 현재 위치를 의미함
DATASET_PATH = "./data/AI_hub_preprocessed.xlsx"

PROMPT = '''
You are a competent translator. When conversation data is input, translate from English to Korean in a natural, colloquial way.
Only output the translation of the input data and do not include any other contents.'''   #zero-shot



#Model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = bnbConfig,
    device_map="auto",
    torch_dtype=torch.float16,
)

#tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#LoRA
model = prepare_model_for_kbit_training(base_model)
model = get_peft_model(model, LoRAConfig)  # LoRA 적용   #model.print_trainable_parameters()




#FT_data loading & sampling
num_samples = 5000  
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



##########################################Implementation###############################################


def en2ko_simple(context):
    with torch.no_grad():
        prompt = f"Translate the following English text to Korean:\n\nEnglish: {context}\n\nKorean:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        terminators = [
                tokenizer.eos_token_id,  # 기본 EOS 토큰
                #tokenizer.convert_tokens_to_ids("."),  # 마침표
                #tokenizer.convert_tokens_to_ids("。"),  # 동아시아 마침표
                tokenizer.convert_tokens_to_ids("\n\n")  # 빈 줄
            ]

        terminators = [t for t in terminators if t is not None and t >= 0]

        output = model.generate(
                **inputs,
                max_new_tokens=512,
                top_p=0.9,
                top_k=50,
                temperature=0.1,
                do_sample=True,
                eos_token_id=terminators if terminators else None,  # 종료 토큰 추가
                #early_stopping=True  # 종료 토큰을 만나면 일찍 중단
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

sample_li = [test_inp1, test_inp2, test_inp3]


#print
for index, test_input in enumerate(sample_li):
  print(f"#{index}")
  print("\n파인튜닝 완료 후 테스트:")
  print(f"입력: {test_input}")
  print(f"번역 결과: {en2ko_simple(test_input)}")
  print('\n'*2)
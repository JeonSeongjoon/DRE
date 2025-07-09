from transformers import BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
import torch


bnbConfig = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_use_double_quant=True,
)


LoRAConfig = LoraConfig(
    r=8,                     # LoRA의 랭크
    lora_alpha=32,           # LoRA의 알파 파라미터
    target_modules = ["q_proj", "k_proj"],  #, "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,       # LoRA의 드롭아웃 비율
    bias="none",             # 바이어스를 학습하지 않음
    task_type="CAUSAL_LM"    # 작업 유형: 인과적 언어 모델링
)


trainerConfig = TrainingArguments(
    output_dir="./ouput",
    num_train_epochs=3,                  # 학습 에폭 수
    per_device_train_batch_size=1,       # 배치 크기 (코랩 메모리에 맞게 조정)
    gradient_accumulation_steps=8,       # 그래디언트 누적 (배치 크기를 효과적으로 늘림)
    save_steps=50,                       # 체크포인트 저장 간격
    logging_steps=10,                    # 로깅 간격
    learning_rate=2e-4,                  # 학습률
    weight_decay=0.001,                  # 가중치 감쇠
    fp16=True,                           # 16비트 부동소수점 사용
    bf16=False,                          # bfloat16 사용하지 않음
    max_grad_norm=0.3,                   # 그래디언트 클리핑
    max_steps=-1,                        # 최대 학습 스텝 (-1은 모든 에폭 완료)
    warmup_ratio=0.03,                   # 워밍업 비율
    group_by_length=True,                # 길이별 그룹화
    lr_scheduler_type="cosine",          # 학습률 스케줄러 유형
    save_total_limit=3,                  # 최대 체크포인트 저장 수
    push_to_hub=False,                   # 허깅페이스 허브에 모델 업로드 (False)
    report_to="none",                    # 로그 보고 대상
)





'''
#8bits quantization config
class bnbConfig(BitsAndBytesConfig):
    def __init__(self,
                 load_in_8bit=True,
                 bnb_8bit_compute_dtype=torch.float16,
                 bnb_8bit_quant_type="nf4",              #nf4, fp4, int8
                 bnb_8bit_use_double_quant=True):

        super().__init__(load_in_8bit = load_in_8bit,
                         bnb_8bit_compute_dtype = bnb_8bit_compute_dtype,
                         bnb_8bit_quant_type = bnb_8bit_quant_type,
                         bnb_8bit_use_double_quant = bnb_8bit_use_double_quant)


#LoRA config
class LoRAConfig(LoraConfig):
    def __init__(self,
                 r=8,                                    # LoRA의 랭크
                 lora_alpha=32,                          # LoRA의 알파 파라미터
                 target_modules = ["q_proj", "k_proj"],  #"v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                 lora_dropout=0.05,                      # LoRA의 드롭아웃 비율
                 bias="none",                            # 바이어스를 학습하지 않음
                 task_type="CAUSAL_LM"):                 # 작업 유형: 인과적 언어 모델링
    
        super().__init__(r=r,
                         lora_alpha=lora_alpha,
                         target_modules=target_modules,
                         lora_dropout=lora_dropout,
                         bias=bias,
                         task_type=task_type)


#Trainer config
class trainerConfig(TrainingArguments):
    def __init__(self,
                 output_dir=None,                     #You should input the output directory
                 num_train_epochs=3,                  # 학습 에폭 수
                 per_device_train_batch_size=1,       # 배치 크기 (코랩 메모리에 맞게 조정)
                 gradient_accumulation_steps=8,       # 그래디언트 누적 (배치 크기를 효과적으로 늘림)
                 save_steps=50,                       # 체크포인트 저장 간격
                 logging_steps=10,                    # 로깅 간격
                 learning_rate=2e-4,                  # 학습률
                 weight_decay=0.001,                  # 가중치 감쇠
                 fp16=True,                           # 16비트 부동소수점 사용
                 bf16=False,                          # bfloat16 사용하지 않음
                 max_grad_norm=0.3,                   # 그래디언트 클리핑
                 max_steps=-1,                        # 최대 학습 스텝 (-1은 모든 에폭 완료)
                 warmup_ratio=0.03,                   # 워밍업 비율
                 group_by_length=True,                # 길이별 그룹화
                 lr_scheduler_type="cosine",          # 학습률 스케줄러 유형
                 save_total_limit=3,                  # 최대 체크포인트 저장 수
                 push_to_hub=False,                   # 허깅페이스 허브에 모델 업로드 (False)
                 report_to="none"):                   # 로그 보고 대상

        super().__init__(output_dir = output_dir,
                         num_train_epochs=num_train_epochs,
                         per_device_train_batch_size=per_device_train_batch_size,
                         gradient_accumulation_steps=gradient_accumulation_steps,
                         save_steps=save_steps,
                         logging_steps=logging_steps,
                         learning_rate=learning_rate,
                         weight_decay=weight_decay,
                         fp16=fp16,
                         bf16=bf16,
                         max_grad_norm=max_grad_norm,
                         max_steps=max_steps,
                         warmup_ratio=warmup_ratio,
                         group_by_length=group_by_length,
                         lr_scheduler_type=lr_scheduler_type,
                         save_total_limit=save_total_limit,
                         push_to_hub=push_to_hub,
                         report_to=report_to)     
'''         

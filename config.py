from transformers import BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
import torch


#8bits quantization config
bnbConfig = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_quant_type="fp4",
    bnb_8bit_use_double_quant=True,
)


#LoRA config

# changed the rank of LoRA
# changed the target_modules to include qkvo of attention and ffn layers to fine-tuning
# It is for the better performance

LoRAConfig = LoraConfig(
    r=16,                     # LoRA의 랭크
    lora_alpha=32,           # LoRA의 알파 파라미터
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,       # LoRA의 드롭아웃 비율
    bias="none",             # 바이어스를 학습하지 않음
    task_type="CAUSAL_LM"    # 작업 유형: 인과적 언어 모델링
)


#Trainer config

# For the stable fine-tuning 
# changed the batch_size and gradient acc steps
# add the evaluation process
# lower the learning rate considering the transfer learning(fine-tuning)

trainerConfig = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,                  # 학습 에폭 수
    per_device_train_batch_size=8,       # 배치 크기 (코랩 메모리에 맞게 조정)
    #per_device_eval_batch_size=2,        # 검증 배치 크기
    gradient_accumulation_steps=2,       # 그래디언트 누적 (배치 크기를 효과적으로 늘림)
    #eval_strategy="steps",               # 검증 전략
    #eval_steps=100,                      # 검증 간격
    save_steps=50,                       # 체크포인트 저장 간격
    logging_steps=10,                    # 로깅 간격
    learning_rate=5e-5,                  # 학습률
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

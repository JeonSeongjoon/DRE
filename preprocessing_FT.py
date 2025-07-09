import pandas as pd
from datasets import Dataset, load_dataset


def loadnSampling(file_path, num_samples=None):

    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        dataset = load_dataset('json', data_files=file_path)
        df = dataset['train'].to_pandas()
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. Excel 또는 JSON 파일을 사용하세요.")

    if num_samples and num_samples < len(df):
        df = df.sample(num_samples, random_state=42)

    return df



def preprocessing(examples, prompt, tokenizer):
    inputs = []
    targets = []

    for en_text, ko_text in zip(examples["English"], examples["Korean"]):

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": en_text}
        ]

        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False              ### False로 처음 돌려보기
        )

        inputs.append(model_inputs)
        targets.append(ko_text)


    # Tokenize
    model_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )

    # Set labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )["input_ids"]


    model_inputs["labels"] = labels

    return model_inputs




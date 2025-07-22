import pandas as pd
from datasets import Dataset, load_dataset


def loadnSampling(file_path, num_samples=3000):

    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        dataset = load_dataset('json', data_files=file_path)
        df = dataset['train'].to_pandas()
    else:
        raise ValueError("Unvalid data format, please check the file format agian.")

    if num_samples and num_samples < len(df):
        df = df.sample(num_samples, random_state=42)

    return df



def preprocessing(examples, prompt, tokenizer):
    input_texts = []
    full_texts = []

    for en_text, ko_text in zip(examples["English"], examples["Korean"]):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"English: \n{en_text}"}
        ]

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        full_text = input_text + "\n\nKorean:\n" + ko_text

        input_texts.append(input_text)
        full_texts.append(full_text)

    # model input = full text
    model_inputs = tokenizer(
        full_texts,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # model label = texts maksing the input text part from full_text
    input_tokenized = tokenizer(
        input_texts,
        max_length=512,
        truncation=True,
        add_special_tokens=False
    )

    labels = model_inputs["input_ids"].clone()

    for i, input_ids in enumerate(input_tokenized["input_ids"]):
        input_len = len(input_ids)
        labels[i, :input_len] = -100

    model_inputs["labels"] = labels


    return model_inputs




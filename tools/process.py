# 针对部分数据的预处理方法
import json
from transformers import AutoTokenizer

NUM = 100
REPEAT = 3


def process_train_dataset():
    # setting
    all_data = []
    token_length = []

    tokenizer = AutoTokenizer.from_pretrained('path', trust_remote_code=True)

    for i in range(1, NUM + 1):
        input_file = f'context1/{i:03d}-context1-x1.md'
        output_file = f'context2/{i:03d}-context2-x1.md'

        context1 = open(input_file, 'r', encoding='utf-8').read()
        context2 = open(output_file, 'r', encoding='utf-8').read()

        data_i = {
            'messages': [
                {'role': 'system', 'content': '你是 AI 信息内容安全专家'},
                {'role': 'user', 'content': f'请你分析以下内容:\n```\n{context1}\n```\n输出分析结果'},
                {'role': 'assistant', 'content': f'分析结果如下:\n```\n{context2}\n```\n'}
            ]
        }
        tokens = tokenizer.apply_chat_template(data_i['messages'], add_generation_prompt=False)
        print(len(tokens))
        all_data.append(data_i)
        token_length.append(len(tokens))

    print(max(token_length))

    final_data = []
    for _ in range(REPEAT):
        final_data.extend(all_data)

    with open('path', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    pass

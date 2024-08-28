# 对于部分数据的切分处理
from pathlib import Path
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import json


def list_files_in_directory(directory):
    file_paths = []
    for path in Path(directory).rglob('*'):
        if path.is_file():
            file_paths.append(str(path))
    return file_paths


def save_contxt_to_file(contxt, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(contxt)


def split_text(text, max_length=500):
    parts = []
    while len(text) > max_length:
        # 找到最近的换行符
        split_index = text.rfind('\n', 0, max_length)
        if split_index == -1:
            # 如果没有找到换行符，则在 max_length 处切分
            split_index = max_length
        parts.append(text[:split_index])
        text = text[split_index:]
    parts.append(text)
    return parts


# 数据源处理 / 收集非结构化数据
def create_markdown_dataset():
    # 指定目录路径
    directory_path = 'path'

    # 获取文件路径列表
    file_paths = list_files_in_directory(directory_path)

    i = 0
    # 打印文件路径列表
    for file_path in file_paths:
        print(file_path)
        if i == 6000:
            break

        num = str(i).zfill(4)
        try:
            with open(file_path, 'r', encoding='GBK', errors='ignore') as f:
                contxt = f.read()
        except Exception as e:
            print(f"An error occurred: {e}")
            pass  # 发生异常时忽略当前文件并继续处理下一个文件

        if len(contxt) < 500:
            output_file_path = f"path"
            save_contxt_to_file(contxt.replace('\n', '').replace('-', ''), output_file_path)
            i = i + 1
        else:
            parts = split_text(contxt)
            for part in parts:
                output_file_path = f"path"
                save_contxt_to_file(part.replace('\n', '').replace('-', ''), output_file_path)

                # 补充计数
                i = i + 1
                num = str(i).zfill(4)
                print(num)


# 训练数据处理
def create_ans_dataset():
    DATA = []

    # 指定目录路径
    directory_path = 'path'

    # openai
    client = OpenAI(
        api_key='sk',
        base_url=''
    )

    # 信息提取规则
    system_prompt = """1. 注意输出代码框，方便复制提取
    2. 按照要求填补内容安全简报，并且保留⌈⌋、[]符号方便信息提取
    3. 生成 markdown 代码
    4. 所有的“换行”都用“\n”表示

    回复格式:
    # 🐻语义内容安全简报🎏

    + 目标内容类别：?
    + 涉及内容属性：?
    + 安全等级划分：[低、中或者高?]危险等级
    + 划分理由：?

    ## 具体内容分析

    ### 分析点一

    截取目标内容：⌈⌋
    分析结果：?

    ### 分析点二

    截取目标内容：⌈⌋
    分析结果：?

    ### 分析点三

    截取目标内容：⌈⌋
    分析结果：?

    ## 结论总结
    ?
    """

    # 获取文件路径列表
    file_paths = list_files_in_directory(directory_path)

    i = 0
    # 打印文件路径列表
    for file_path in tqdm(file_paths):
        if i <= -1:
            i = i + 1
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        CONTEXT1 = f"""{content}"""

        query_prompt = f"""请分析以下文本:\n{CONTEXT1}"""
        # deepseek
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_prompt}
            ],
            top_p=0.7,
            temperature=0.9
        )

        CONTEXT2 = completion.choices[0].message.content

        _data = {
            'messages': [
                {'role': 'system', 'content': '你是 AI 信息内容安全专家'},
                {'role': 'user', 'content': f'请你分析以下内容:\n```\n{CONTEXT1}\n```\n输出分析结果'},
                {'role': 'assistant', 'content': f'分析结果如下:\n---\n{CONTEXT2}\n---\n'}
            ]
        }

        num = str(i).zfill(4)
        DATA.append(_data)
        with open(f'path/train_full_{num}.json', 'w', encoding='utf-8') as f:
            json.dump(_data, f, ensure_ascii=False, indent=4)

        i = i + 1

    with open(f'path', 'w', encoding='utf-8') as f:
        json.dump(DATA, f, ensure_ascii=False, indent=4)


def synthesis_compo_dataset():
    # open and list the folder
    directory = "path"
    file_paths = list_files_in_directory(directory=directory)

    # setting
    DATA = []
    cnt_1 = 0  # ```markdown
    cnt_2 = 0  # ```
    cnt_3 = 0  # 修改后 -> ```markdown
    cnt_4 = 0  # 修改后 -> ```
    for file_path in file_paths:
        if "```markdown":
            cnt_1 += 1

        with open(file_path, 'r', encoding="utf-8") as f:
            contxt = json.load(f)
            contxt["messages"][2]["content"] = contxt["messages"][2]["content"].replace("```markdown", "")
            if "```":
                cnt_2 += 1
            contxt["messages"][2]["content"] = contxt["messages"][2]["content"].replace("```", "")
            if not "```markdown":
                cnt_3 += 1
            if not "```markdown":
                cnt_4 += 1
            DATA.append(contxt)

    print("cnt_1: ", cnt_1)
    print("cnt_2: ", cnt_2)
    print("cnt_3: ", cnt_3)
    print("cnt_4: ", cnt_4)
    with open(f'path', 'w', encoding='utf-8') as f:
        json.dump(DATA, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    pass
    # create_markdown_dataset()
    # create_ans_dataset()
    # synthesis_compo_dataset()

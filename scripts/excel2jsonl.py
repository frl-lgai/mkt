import os
import random
import json
import openpyxl
import typer

def prompting(*row):

    def p(index):
        cell = str(row[index].value) if row[index].value else ""
        return cell.strip() if cell else ""

    if not (p(0) or p(1) or p(2) or p(3)): return None

    return \
f"""다음의 상품을 홍보하는 창의적인 마케팅 문구를 작성해 보세요.
[회사] {p(2)}
[카테고리] {p(0)}
[분류] {p(3)}
[상품명] {p(1)}
[마케팅 문구] """


def main(
    excel_file: str = "/w/data/mkt/kobaco_mkt.xlsx",
    output_dir: str = "/w/data/mkt",
    num_train: int = 12000,
    num_valid: int = 1000,
):

    workbook = openpyxl.load_workbook(excel_file)
    worksheet = workbook['Sheet1']

    f_train = open(os.path.join(output_dir, "train.jsonl"), 'w')
    f_valid = open(os.path.join(output_dir, "valid.jsonl"), 'w')

    for i, row in enumerate(worksheet.iter_rows()):
        if i == 0: continue

        prompt = prompting(*row)
        target = row[5].value if row[5].value else ""

        if prompt is None: continue

        example = json.dumps({"input": prompt, "label": target}, ensure_ascii=False)

        if random.random() < 0.0002:
            print(prompt + target, end='\n\n')
        
        if i <= num_train:
            f_train.write(example+"\n")
         
        if num_train < i <= num_train + num_valid:
            f_valid.write(example+"\n")


if __name__ == '__main__':
    typer.run(main)

import os
from datasets import load_dataset

fold_name: str = "literature"
os.makedirs(fold_name, exist_ok=True)

# Берем с запасом, так как будем фильтровать по длине текста
dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train[:800]")

count: int = 0
for item in dataset:
    title:str = item.get('title', '') or ''
    abstract:str = item.get('abstract', '') or ''
    
    if not abstract:
        continue
        
    doc_text:str = f"TITLE: {title}\n\nABSTRACT:\n{abstract}"
    
    if len(doc_text) < 200:
        continue
    #TODO: сделать кросплатформенность путей
    filename:str = f"{fold_name}/doc_{count:04d}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(doc_text)
    
    count += 1
    if count >= 500:
        break

file_count:int = len([f for f in os.listdir(fold_name) if f.endswith('.txt')])
if file_count < 500:
    print(f"ПРЕДУПРЕЖДЕНИЕ: В папке '{fold_name}' найдено только {file_count} файлов (меньше 500)!")
else:
    print(f"Проверка пройдена: в папке успешно создано {file_count} файлов.")
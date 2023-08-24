# -*- coding: utf-8 -*-

import os
import openai
import json
import re
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu

"""# Tiền xử lý dữ liệu"""

def preprocess_text(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ ký tự không cần thiết
    cleaned_text = cleaned_text.lower()  # Chuyển thành chữ thường
    return cleaned_text

# Đọc dữ liệu từ tệp JSON
with open('cnn.json', 'r') as json_file:
    data = json.load(json_file)

# Tiền xử lý dữ liệu cho từng đối tượng trong tệp JSON
for key, entry in data.items():
    if 'article' in entry:
        article = entry['article']
        cleaned_article = preprocess_text(article)
        entry['article'] = cleaned_article

# Ghi dữ liệu sau tiền xử lý vào một tệp JSON mới
with open('preprocessed_data.json', 'w') as preprocessed_file:
    json.dump(data, preprocessed_file, indent=4)

"""# Tạo tóm tắt từ dữ liệu gốc"""

openai.api_key = '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'

def generate_summary(content):
   prompt = f"Summarize the following text:\n{content}"
   response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
   )
   summary = response.choices[0].text.strip()
   return summary

# Đọc dữ liệu đã tiền xử lý từ tệp JSON
with open('preprocessed_data.json', 'r') as json_file:
    data = json.load(json_file)

# Tạo tóm tắt từ dữ liệu gốc và cập nhật vào tệp JSON
generated_summaries = []
for _, item in data.items():
    article = item["article"]
    prompt = f"Summarize the following article:\n{article}\nSummary:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=150
    )
    summary = response.choices[0].text.strip()
    generated_summaries.append({
        "id": id,
        "article": article,
        "summ": summary
    })


#Ghi dữ liệu sau tạo tóm tắt vào một tệp JSON mới
with open('summarized_data.json', 'w') as summarized_file:
    json.dump(generated_summaries, summarized_file, indent=4)

"""# Phân tách dữ liệu"""

# Đọc dữ liệu từ tệp JSON
with open('summarized_data.json', 'r') as json_file:
    data = json.load(json_file)

# Tạo danh sách các đối tượng từ dữ liệu đã tạo tóm tắt
object_list = list(data.values())

# Phân tách dữ liệu thành tập đào tạo, tập xác nhận và tập kiểm tra
train_data, temp_data = train_test_split(object_list, test_size=0.3, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Ghi tập đào tạo, tập xác nhận và tập kiểm tra vào các tệp JSON mới
with open('train_data_summarized.json', 'w') as train_file:
    json.dump(train_data, train_file, indent=4)

with open('valid_data_summarized.json', 'w') as valid_file:
    json.dump(valid_data, valid_file, indent=4)

with open('test_data_summarized.json', 'w') as test_file:
    json.dump(test_data, test_file, indent=4)

"""# Đánh giá mô hình

Sử dụng tập dữ liệu kiểm tra để đánh giá mô hình. Đo lường các chỉ số đánh giá như BLEU, ROUGE, hoặc METEOR để đo lường độ chính xác 
của tóm tắt mô hình so với tóm tắt thực tế.
"""

# Đọc dữ liệu từ tệp JSON
with open('test_data_summarized.json', 'r') as json_file:
    test_data = json.load(json_file)

bleu_scores = []

# Tính BLEU score cho mỗi mẫu trong tập kiểm tra
for entry in test_data:
    reference_summary = entry['summ']  # Tóm tắt thực tế
    generated_summary = generate_summary(entry['article'])  # Tóm tắt được tạo bởi mô hình
    reference_tokenized = [reference_summary.split()]  # Chia thành danh sách các từ
    generated_tokenized = generated_summary.split()  # Chia thành danh sách các từ
    bleu_score = sentence_bleu(reference_tokenized, generated_tokenized)
    bleu_scores.append(bleu_score)

average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print("Average BLEU Score:", average_bleu_score)
import tkinter as tk
from tkinter import ttk
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn import CrossEntropyLoss
from transformers import AdamW
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
model = T5ForConditionalGeneration.from_pretrained(r'C:\Users\Administrator\Desktop\nlp\model')
tokenizer = AutoTokenizer.from_pretrained("t5-base",model_max_length=1024)
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model_v1 = SentenceTransformer('bert-base-nli-mean-tokens')
def tom_tat_v1(text, num_sentences,model_v1):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokenzier = []
    for sentence in sentences:
        word_tokens = word_tokenize(sentence)
        tokenzie = [word for word in word_tokens if word.lower() not in stop_words]
        tokenzier.append(tokenzie)

    # Nhúng các câu
    sentence_embeddings = model_v1.encode(tokenzier)
    similarity_matrix = cosine_similarity(sentence_embeddings, sentence_embeddings)
    rank = sorted([(sentences[i], similarity_matrix.sum(axis=1)[i]) for i in range(len(sentences))], key=lambda x: x[1], reverse=True)
    summary_sentences = [sentence[0] for sentence in rank[:num_sentences]]
    summary = ' '.join(summary_sentences)

    return summary


def summarize_text():
    text = text_entry.get("1.0", tk.END)
    inputs = tokenizer.encode_plus(text, return_tensors='pt', max_length=512, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)

    summary1 = tokenizer.decode(output[0], skip_special_tokens=True)
    summary2 = tom_tat_v1(text, 5,model_v1)
    summarized_text_1 = summary1
    summarized_text_2 = summary2

    summary_result_1.delete("1.0", tk.END)
    summary_result_1.insert(tk.END, summarized_text_1)

    summary_result_2.delete("1.0", tk.END)
    summary_result_2.insert(tk.END, summarized_text_2)

root = tk.Tk()
root.title("Ứng dụng tóm tắt văn bản")

# Box để người dùng điền văn bản cần tóm tắt
text_entry = tk.Text(root, height=20, width=100)
text_entry.pack(pady=10)

# Button để tóm tắt văn bản
summarize_button = ttk.Button(root, text="Tóm tắt", command=summarize_text)
summarize_button.pack()

# Tiêu đề cho kết quả tóm tắt cách 1
summary_label_1 = tk.Label(root, text="Tóm tắt theo phương pháp trừu tượng:")
summary_label_1.pack()

# Box để hiển thị kết quả tóm tắt cách 1
summary_result_1 = tk.Text(root,  height=20, width=100)
summary_result_1.pack(pady=10)

# Tiêu đề cho kết quả tóm tắt cách 2
summary_label_2 = tk.Label(root, text="Tóm tắt theo phương pháp trích xuất:")
summary_label_2.pack()

# Box để hiển thị kết quả tóm tắt cách 2
summary_result_2 = tk.Text(root,  height=20, width=100)
summary_result_2.pack(pady=10)

root.mainloop()

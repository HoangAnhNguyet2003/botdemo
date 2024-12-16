import os
import requests
from underthesea import word_tokenize
import nltk
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import math
from flask import Flask, request, jsonify
from flask_cors import CORS

# Tải thư viện NLTK cần thiết
nltk.download('punkt')

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
CORS(app)

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("Assistant_tv/finetune_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("Assistant_tv/finetune_model")

def load_documents(folder_path):
    """Tải tài liệu từ thư mục."""
    documents = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
                filenames.append(filename)
    return documents, filenames

def load_index(filename='Assistant_tv/index.pkl'):
    """Tải chỉ mục từ file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def bm25_search(query, index, documents, vocab, k1=1.5, b=0.75):
    tokenized_query = word_tokenize(query.lower())
    scores = {}
    
    doc_lengths = [len(doc.split()) for doc in documents]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)

    for word in tokenized_query:
        if word in vocab and word in index: 
            for doc_id, freq in index[word].items():
                doc_length = doc_lengths[doc_id]
                
                # Tính IDF
                idf = 1 + math.log((len(documents) - len(index[word]) + 0.5) / (len(index[word]) + 0.5)) if len(index[word]) > 0 else 0
                
                # Tính TF
                tf = freq * (k1 + 1) / (freq + k1 * (1 - b + b * (doc_length / avg_doc_length)))  
                
                # Tính điểm cho tài liệu
                score = idf * tf
                
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += score

    return scores

def test_model_on_new_data(query, document, filename, tokenizer, model, max_len=128):
    """Dự đoán nhãn cho tài liệu mới."""
    encoding = tokenizer(
        query,
        document,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in encoding.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    return predicted_class, probabilities[0][1].item()

def call_gpt_api(prompt):
    """Gọi API GPT để nhận phản hồi."""
    api_key = 'sk-gA2KXQpdq3jE7n9APG5XBNsDufTC7O1cC6TkDFu5NwT3BlbkFJnnjHdF8FrxtwnhX_WPZICvtAnA2c5HyEGmqpE_L4sA'
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512 
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    folder_path = 'Assistant_tv/data'
    documents, filenames = load_documents(folder_path)

    # Tải chỉ mục
    index_file = 'Assistant_tv/index.pkl'
    vocab_file_path = 'Assistant_tv/vocab_ngram.txt'
    vocab = []

    if os.path.exists(index_file):
        index = load_index(index_file)
    else:
        return jsonify({'error': 'Index file not found.'}), 404

    with open(vocab_file_path, 'r', encoding='utf-8') as vocab_file:
        vocab = [line.strip() for line in vocab_file.readlines()]

    scores = bm25_search(query, index, documents, vocab)

    # Lấy 3 văn bản có điểm cao nhất
    top_n = 4
    top_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:top_n]

    all_best_documents = []
    for doc_id in top_indices:
        predicted_class, prob = test_model_on_new_data(query, documents[doc_id], filenames[doc_id], tokenizer, model)
        
        if predicted_class == 1:
            all_best_documents.append(documents[doc_id])

    if all_best_documents:
        prompt = f"Dựa vào thông tin: {', '.join(all_best_documents)}, hãy trả lời {query} như assistant của trường đại học xây dựng. Nếu là lời chào thì hãy trả lời đáp lại, nếu không có thông tin hãy bảo tạm thời tôi chưa có đủ thông tin về câu trả lời này và đưa các thông tin liên hệ liên quan. Nếu trong văn bản không có thông tin mà bạn biết câu trả lời hãy đưa ra câu trả lời đúng"
        response = call_gpt_api(prompt)
        if response:
            return jsonify({'response': response})
    else:
        return jsonify({'message': 'No relevant documents found with label 1.'}), 404

if __name__ == "__main__":
    app.run(debug=True)
trong file app.py có đoạn code sau 
```python
def call_gpt_api(prompt):
    api_key = 'YOUR_API_KEY_HERE'  # Thay thế bằng API key bạn muốn dùng
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",  # Thay thế bằng mô hình bạn muốn dùng
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512  # Thay đổi số lượng token đầu ra theo mong muốn
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

@app.route('/api/chat', methods=['POST'])
```

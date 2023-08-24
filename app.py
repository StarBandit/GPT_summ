from flask import Flask, render_template, request
import openai

app = Flask(__name__)

# Thay thế 'YOUR_API_KEY' bằng API key của bạn
openai.api_key = 'YOUR_API_KEY'

def generate_summary(content):
    prompt = f"Summarize the following text:\n{content}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )
    summary = response.choices[0].text.strip()
    return summary

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        summary = generate_summary(input_text)
        return render_template('index.html', input_text=input_text, summary=summary)
    return render_template('index.html', input_text='', summary='')

if __name__ == '__main__':
    app.run(debug=True)

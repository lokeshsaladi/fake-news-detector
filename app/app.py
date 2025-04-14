from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    vec = vectorizer.transform([news])
    prediction = model.predict(vec)[0]
    result = "Real" if prediction == 1 else "Fake"
    return render_template('index.html', prediction=result, news=news)

if __name__ == '__main__':
    app.run(debug=True)

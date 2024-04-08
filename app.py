from flask import Flask, request, render_template, send_file, jsonify, send_from_directory
import pandas as pd
import pickle

app = Flask(__name__)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
print(model)

predictions_to_labels = {
    0: 'GPD',
    1: 'GRDA',
    2: 'LPD',
    3: 'LRDA',
    4: 'OTHER',
    5: 'SEIZURE'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def predict():
    file = request.files['file']
    df = pd.read_csv(file)
    predictions = model.predict(df).tolist()
    string_predictions = [predictions_to_labels[pred] for pred in predictions]    
    return jsonify({'predictions': string_predictions})


if __name__ == '__main__':
    app.run(debug=True)

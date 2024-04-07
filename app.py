from flask import Flask, request, render_template, send_file, jsonify, send_from_directory
import pandas as pd
from joblib import load
import pickle

app = Flask(__name__)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# model = load('model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def predict():
    file = request.files['file']
    df = pd.read_csv(file)
    # Preprocess your DataFrame as needed
    predictions = model.predict(df)
    
    return jsonify({'predictions': predictions.tolist()})


if __name__ == '__main__':
    app.run(debug=True)



# from flask import Flask, request, render_template, send_from_directory
# import os
# from werkzeug.utils import secure_filename

# app = Flask(__name__)
# model = load('model.joblib')

# # Assuming your index.html is in a folder named 'templates'
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return 'No file part'
#     file = request.files['file']
#     if file.filename == '':
#         return 'No selected file'
#     if file:
#         filename = secure_filename(file.filename)
#         filepath = os.path.join('uploads', filename)
#         file.save(filepath)
        
#         # Process the file with your ML model here
#         result = model.process(filepath)
#         # Example: result = your_model.process(filepath)
#         # Let's assume you simply rename the file as a placeholder for processing
#         processed_filepath = filepath + "_processed"
#         os.rename(filepath, processed_filepath)

#         # Option to return the processed file or results
#         return send_from_directory(directory=os.path.dirname(processed_filepath),
#                                    filename=os.path.basename(processed_filepath),
#                                    as_attachment=True)
#         # Or, return results in another preferred format
#     # return jsonify({'predictions': predictions.tolist()})


# if __name__ == '__main__':
#     # Create uploads directory if it doesn't exist
#     # if not os.path.exists('uploads'):
#     #     os.makedirs('uploads')
    
#     app.run(debug=True)

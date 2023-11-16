from flask import Flask, render_template, request
import subprocess
import test

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        if file:
            file.save('./static/sign.jpg')
            command = ['java', '-cp', 'EdgeDetector', 'Testing', 'static/sign.jpg']
            process = subprocess.Popen(command)
            process.wait()

            pred = test.predict()
            return render_template('index.html', Prediction=pred)

    return render_template('index.html', Prediction='Select Image')
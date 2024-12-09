from flask import Flask, render_template, request, jsonify
import audiotools
import time
import torch
import dac


app = Flask(__name__)


torch.set_num_threads(1)

non_pruned_model = dac.DAC.load('../models/base.pt')
non_pruned_model.eval()

pruned_model = dac.DAC.load('../models/pruned_100_threshold.pt')
pruned_model.eval()


def process_audio(model, signal):

    with torch.no_grad():

        start = time.time()
        z = model.compress(signal)
        compress_time = f'{time.time() - start:.2f}'

        start = time.time()
        y = model.decompress(z)
        decompress_time = f'{time.time() - start:.2f}'

    return y, compress_time, decompress_time


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    file = request.files['audio']
    file.save('static/processed/input.weba')
    input_signal = audiotools.AudioSignal('static/processed/input.weba')
    input_signal.write('static/processed/input.wav')

    a, a1, a2 = process_audio(non_pruned_model, input_signal)
    a.write('static/processed/non_pruned.wav')

    b, b1, b2 = process_audio(pruned_model, input_signal)
    b.write('static/processed/pruned.wav')

    response = {
        "results": [
            {"name": "Base Model", "time1": a1, "time2": a2, "file": "/static/processed/non_pruned.wav"},
            {"name": "Pruned Model (X\% size of base)", "time1": b1, "time2": b2, "file": "/static/processed/pruned.wav"},
        ]
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)

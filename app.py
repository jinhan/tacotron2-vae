#!flask/bin/python

import os, traceback
import hashlib
import argparse
from flask_cors import CORS, cross_origin
from flask import Flask, request, render_template, jsonify, \
        send_from_directory, make_response, send_file

from synthesizer import Synthesizer
from utils import str2bool, makedirs, add_postfix


ROOT_PATH = "web"
AUDIO_DIR = "audio"
AUDIO_PATH = os.path.join(ROOT_PATH, AUDIO_DIR)

base_path = os.path.dirname(os.path.realpath(__file__))
static_path = os.path.join(base_path, 'web/static')

global_config = None
synthesizer = Synthesizer()
app = Flask(__name__, root_path=ROOT_PATH, static_url_path='')
CORS(app)

def generate_audio_response(text, condition_on_ref, ref_audio, ratios):
    hashed_text = hashlib.md5(text.encode('utf-8')).hexdigest()

    relative_dir_path = os.path.join(AUDIO_DIR, 'tacotron2-vae')
    relative_audio_path = os.path.join(
            relative_dir_path, "{}.wav".format(hashed_text))
    real_path = os.path.join(ROOT_PATH, relative_audio_path)
    makedirs(os.path.dirname(real_path))

    if condition_on_ref:
        ref_audio = ref_audio.replace('/uploads', '/data1/jinhan')
        
    try:
        audio = synthesizer.synthesize(text, real_path, condition_on_ref, ref_audio, ratios)
    except Exception as e:
        traceback.print_exc()
        return jsonify(success=False), 400

    return send_file(
            relative_audio_path,
            mimetype="audio/wav", 
            as_attachment=True, 
            attachment_filename=hashed_text + ".wav")

    response = make_response(audio)
    response.headers['Content-Type'] = 'audio/wav'
    response.headers['Content-Disposition'] = 'attachment; filename=sound.wav'
    return response

@app.route('/')
def index():
    text = request.args.get('text') or "듣고 싶은 문장을 입력해 주세요."
    return render_template('index.html', text=text)

@app.route('/generate')
def view_method():
    text = request.args.get('text')
    condition_on_ref = request.args.get('con')    
    
    if text:
        if condition_on_ref=='true':
            ref_audio = request.args.get('ref')
            condition_on_ref = True
            ratios = None
            
            return generate_audio_response(text, condition_on_ref, ref_audio, ratios) # ref_audi, ratios
        else:
            n = float(request.args.get('n'))
            s = float(request.args.get('s'))
            h = float(request.args.get('h'))
            a = float(request.args.get('a'))
            sigma = n+s+h+a
            if sigma:
                ratios = [round(x / sigma * 100)/100 for x in [n, s, h, a]]
            else:
                ratios = [1.0, 0.0, 0.0, 0.0]
            
            ref_audio = None
            condition_on_ref = False
            
            return generate_audio_response(text, condition_on_ref, ref_audio, ratios) # ref_audi, ratios
    else:
        return {}

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory(
            os.path.join(static_path, 'js'), path)

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory(
            os.path.join(static_path, 'css'), path)

@app.route('/audio/<path:path>')
def send_audio(path):
    return send_from_directory(
            os.path.join(static_path, 'audio'), path)

@app.route('/uploads/<path:path>')
def send_uploads(path):
    return send_from_directory(
            os.path.join(static_path, 'uploads'), path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--waveglow_path', required=True)
    parser.add_argument('--port', default=51000, type=int)
    parser.add_argument('--debug', default=False, type=str2bool)
    parser.add_argument('--is_korean', default=True, type=str2bool)
    config = parser.parse_args()

    if os.path.exists(config.checkpoint_path):
        synthesizer.load(config.checkpoint_path, config.waveglow_path)
    else:
        print(" [!] load_path not found: {}".format(config.checkpoint_path))

    app.run(host='10.100.1.119', threaded=True, port=config.port, debug=config.debug)

import os, traceback, json
import hashlib
import argparse
from flask_cors import CORS, cross_origin
from flask import Flask, request, render_template, jsonify, \
        send_from_directory, make_response, send_file
from synthesizer import Synthesizer
from utils import str2bool, makedirs, add_postfix
import base64

ROOT_PATH = "web"
AUDIO_DIR = "audio"
AUDIO_PATH = os.path.join(ROOT_PATH, AUDIO_DIR)

base_path = os.path.dirname(os.path.realpath(__file__))
static_path = os.path.join(base_path, 'web/static')

global_config = None
synthesizer = Synthesizer()
app = Flask(__name__, root_path=ROOT_PATH, static_url_path='')
CORS(app)

def generate_audio_response(args):
    print(args)
    text = args['text']
    print(text)
    condition_on_ref = False
    ref_audio = None
    
    n = float(args['neu'])
    s = float(args['sad'])
    h = float(args['hap'])
    a = float(args['ang'])
    sigma = n+s+h+a
    if sigma:
        ratios = [round(x / sigma * 100)/100 for x in [n, s, h, a]]
    else:
        ratios = [1.0, 0.0, 0.0, 0.0]

    hashed_text = hashlib.md5(text.encode('utf-8')).hexdigest()

    relative_dir_path = os.path.join(AUDIO_DIR, 'tacotron2-vae')
    relative_audio_path = os.path.join(
            relative_dir_path, "{}.wav".format(hashed_text))
    real_path = os.path.join(ROOT_PATH, relative_audio_path)
    makedirs(os.path.dirname(real_path))

    if condition_on_ref:
        ref_audio = ref_audio.replace('/uploads', '/home/jinhan/Storage')
        
    try:
        synthesizer.synthesize(text, real_path, condition_on_ref, ref_audio, ratios)
    except Exception as e:
        traceback.print_exc()
        return jsonify(success=False), 400

    b64_data = base64.b64encode(open(real_path, "rb").read())
    return json.dumps({"params":{
        "text":text,
        "neu": n, "hap": h, "sad": s, "ang": a},
        "data": str(b64_data.decode('utf-8'))})


@app.route('/api', methods=['POST'])
def API():
    args = json.loads(request.data)

    return generate_audio_response(args)

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('--checkpoint_path', required=True)
    main_parser.add_argument('--waveglow_path', required=True)
    main_parser.add_argument('--port', default=51001, type=int)
    main_parser.add_argument('--debug', default=False, type=str2bool)
    main_parser.add_argument('--is_korean', default=True, type=str2bool)
    config = main_parser.parse_args()
    # print(config)
    

    if os.path.exists(config.checkpoint_path):
        synthesizer.load(config.checkpoint_path, config.waveglow_path)
    else:
        print(" [!] load_path not found: {}".format(config.checkpoint_path))

    app.run(host='192.168.0.10', threaded=True, port=config.port, debug=config.debug)

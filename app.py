from flask_cors import CORS
from flask import Flask,request
from MinhashLSH import Minhash
from preprocess.text_pipeline import TextPipeline
import config
import spacy
import json
from preprocess import utils

Minhash_f = Minhash_p = Minhash_s = Minhash_t = None

app = Flask(__name__)


CORS(app)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

@app.route('/', methods=['GET'])
def index():
	response = app.response_class(
		response=json.dumps({'data': "Welcome LSH Entrypoint!"}, indent=4),
		status=200,
		mimetype='application/json'
	)
	return response


@app.route('/query/', methods=['POST'])
def query():
	try:
		query = request.json['data']
		type = request.json['type']
	except:
		return app.response_class(
			response=json.dumps({'error':'query or type is empty'}, indent=4),
			status=505,
			mimetype='application/json'
		)

	try:
		threshold = request.json['threshold']
	except:
		threshold = config.default_threshold

	try:
		maxResults = request.json['max']
	except:
		maxResults = config.num_recommendations
	Minhash_m = None
	T = False
	if type == "Phrase":
		Minhash_m = Minhash_f
	elif type == "Paragraph":
		Minhash_m = Minhash_p
	elif type == "Section":
		Minhash_m = Minhash_s
	elif type == "TriGram":
		T = True
		Minhash_m = Minhash_t

	if Minhash_m == None:
		return app.response_class(
			response=json.dumps({'error': 'type is not validd'}, indent=4),
			status=505,
			mimetype='application/json'
		)

	result = Minhash_m.predict(query,threshold=threshold,N=maxResults,Trigram=T)

	response = app.response_class(
		response=json.dumps(result, indent=4),
		status=200,
		mimetype='application/json'
	)
	return response

@app.route('/connect/', methods=['GET'])
def connect():
	k = str(request.args.get('k', default=3, type=int))
	global Minhash_f
	global Minhash_p
	global Minhash_s
	global Minhash_t

	models = []
	msg = "NOT GOOD"
	# load model phrase

	Minhash_f = Minhash('phrase',k=k)
	Minhash_f.load()
	models.append("Phrase_"+k)

	Minhash_p = Minhash('paragraph',k=k)
	Minhash_p.load()
	models.append("Paragraph_"+k)

	Minhash_s = Minhash('section',k=k)
	Minhash_s.load()
	models.append("Section_"+k)

	Minhash_t = Minhash('trigram',k=k)
	Minhash_t.load()
	models.append("TriGram")


	if len(models) ==  4:
		msg = "All loaded!"
	elif len(models) > 0:
		msg = "loaded"

	response = app.response_class(
		response=json.dumps({
			'data': msg,
			'K': k,
			'models': models,
			'path':config.path_models,
			'wordbased': config.wordBased,
			'ip': config.ip
		}, indent=4, sort_keys=True),
		status=200,
		mimetype='application/json'
	)
	return response

if __name__ == '__main__':
	app.run('0.0.0.0')
 #port="1234"

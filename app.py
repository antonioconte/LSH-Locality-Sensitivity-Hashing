from flask_cors import CORS
from flask import Flask,request
from LSH import LSH
from preprocess.text_pipeline import TextPipeline
import config
import spacy
import json
from preprocess import utils

LSH_f = LSH()
LSH_p = LSH()
LSH_s = LSH()
LSH_t = LSH()

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
	LSH_m = None
	T = False
	if type == "Phrase":
		LSH_m = LSH_f
	elif type == "Paragraph":
		LSH_m = LSH_p
	elif type == "Section":
		LSH_m = LSH_s
	elif type == "TriGram":
		T = True
		LSH_m = LSH_t

	if LSH_m == None:
		return app.response_class(
			response=json.dumps({'error': 'type is not validd'}, indent=4),
			status=505,
			mimetype='application/json'
		)

	result = LSH_m.predict(query,threshold=threshold,N=maxResults,Trigram=T)

	response = app.response_class(
		response=json.dumps(result, indent=4),
		status=200,
		mimetype='application/json'
	)
	return response

@app.route('/connect/', methods=['GET'])
def connect():
	models = []
	msg = "NOT GOOD"
	# load model phrase
	global LSH_f
	LSH_f.load_lsh(config.path_models + "_phrase")
	models.append("Phrase")

	global LSH_p
	LSH_p.load_lsh(config.path_models + "_paragraph")
	models.append("Paragraph")

	global LSH_s
	LSH_s.load_lsh(config.path_models + "_section")
	models.append("Section")

	global LSH_t
	LSH_t.load_lsh(config.path_models + "_trigram")
	models.append("TriGram")


	if len(models) ==  4:
		msg = "All loaded!"
	elif len(models) > 0:
		msg = "loaded"

	response = app.response_class(
		response=json.dumps({
			'data': msg,
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

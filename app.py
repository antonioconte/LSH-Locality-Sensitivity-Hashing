from flask_cors import CORS
from flask import Flask,request
from LSH import LSH
from preprocess.text_pipeline import TextPipeline
import config
import spacy
import json
from preprocess import utils

LSH_f = LSH()
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
	query = request.json['data']
	type = request.json['type']
	try:
		threshold = request.json['threshold']
	except:
		threshold = config.default_threshold

	try:
		maxResults = request.json['max']
	except:
		maxResults = config.num_recommendations

	if type == "Phrase":
		LSH_m = LSH_f
	elif type == "Paragraph":
		LSH_m = None
	elif type == "Section":
		LSH_m = None
	elif type == "3-Gram":
		LSH_m = None

	result = LSH_m.predict(query,threshold=threshold,N=maxResults)

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
	if LSH_f.model == None:
		LSH_f.load_lsh(config.path_models + "_phrase")
		models.append("Phrase")
	else:
		models.append("Phrase")

	if len(models) ==  4:
		msg = "All loaded!"
	elif len(models) > 0:
		msg = "loaded"

	response = app.response_class(
		response=json.dumps({'data': msg, 'models': models, 'path':config.path_models}, indent=4),
		status=200,
		mimetype='application/json'
	)
	return response

if __name__ == '__main__':
	app.run('0.0.0.0')

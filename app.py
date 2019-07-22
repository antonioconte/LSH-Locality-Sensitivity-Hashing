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
	if type == "Phrase":
		LSH_m = LSH_f
	elif type == "Paragraph":
		LSH_m = None
	elif type == "Section":
		LSH_m = None
	elif type == "3-Gram":
		LSH_m = None

	# if(request.json['metric']):
	# 	metric = request.json['metric'] # jac | lev_sim | lev
	result = LSH_m.predict(query,metric="lev")
	# print("RESULT: ")
	# print(json.dumps(result, indent=4, sort_keys=True))
	# print("\n"*2)

	response = app.response_class(
		response=json.dumps(result, indent=4),
		status=200,
		mimetype='application/json'
	)
	return response

@app.route('/connect/', methods=['GET'])
def connect():
	models = []
	msg = "ok"
	# load model phrase
	global LSH_f
	if LSH_f.model == None:
		LSH_f.load_lsh(config.path_models + "_phrase")
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

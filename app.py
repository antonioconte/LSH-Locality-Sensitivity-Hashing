from flask_cors import CORS
from flask import Flask,request
from LSH import load_lsh,predict,train
from preprocess.text_pipeline import TextPipeline
import config
import spacy
import json
from preprocess import utils

LSH_m = None
app = Flask(__name__)
permutations = config.permutations
num_recommendations = config.num_recommendations

nlp = spacy.load('en_core_web_sm')
normalizer = TextPipeline(nlp)

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
	query = utils.cleanhtml(query)
	result = predict(query,permutations,num_recommendations,LSH_m,normalizer)
	print("RESULT: ")
	print(json.dumps(result, indent=4, sort_keys=True))
	print("\n"*4)

	# result = { 'data': query }
	response = app.response_class(
		response=json.dumps(result, indent=4),
		status=200,
		mimetype='application/json'
	)
	return response

@app.route('/connect/', methods=['GET'])
def connect():
	global LSH_m
	msg = "LSH already Loaded"
	if LSH_m == None:
		LSH_m = load_lsh("./model/model"+ "F")
		msg = "LSH loaded!"

	response = app.response_class(
		response=json.dumps({'data': msg}, indent=4),
		status=200,
		mimetype='application/json'
	)
	return response

if __name__ == '__main__':
	app.run('0.0.0.0')

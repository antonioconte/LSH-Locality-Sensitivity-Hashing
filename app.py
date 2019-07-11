import json
from flask_cors import CORS
from flask import Flask,request
from LSH import load_lsh,predict,train
import spacy
from preprocess.text_pipeline import TextPipeline
app = Flask(__name__)
LSH_m = None
permutations = 128
num_recommendations = 5
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

@app.route('/query/', methods=['GET'])
def query():
	title = """On operator in question is Le Seysselan GAEC, Vallod, SEYSSEL."""
	result = predict(title,permutations,num_recommendations,LSH_m,normalizer)
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

virtualenv -p python3.5 venv && source venv/bin/activate && pip --no-cache-dir install -r requirements.txt  && python3 -m spacy download en_core_web_sm && echo $(hostname -i) && python app.py

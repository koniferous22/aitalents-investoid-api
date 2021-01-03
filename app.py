# app.py
import atexit
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from sqlalchemy import create_engine, select, Table, Column, Integer, String, MetaData, ForeignKey, Text, column, desc
from sqlalchemy.dialects.postgresql import TSVECTOR

from os import path, getenv
import torch
import torch.nn as nn
import torch.nn.functional as F
from keras_preprocessing import sequence, text
import numpy as np

dotenv_path = path.join(path.dirname(__file__), '.env')
if path.isfile(dotenv_path):
    load_dotenv(dotenv_path)

metadata = MetaData()
article_companies = Table('article_companies', metadata,
    Column('id', Integer, primary_key=True),
    Column('article_entity', String(90), unique=True, nullable=False),
    Column('nasdaq_entity', String(50)),
    Column('nyse_entity', String(50)),
    Column('nasdaq_label', String(5)),
    Column('nyse_label', String(5)),
    Column('relevance', Integer),
    Column('search_index', TSVECTOR),
)

article_references = Table('article_references', metadata,
  Column('id', Integer, primary_key=True),
  Column('article_entity', Integer, ForeignKey('article_companies.id')),
  Column('title', String(300), nullable=False),
  Column('title', Text, nullable=False),
)

app = Flask(__name__)
engine = create_engine(getenv('DB_CONFIG'))
conn = engine.connect()
def exit_handler():
    print('Closing db connection')
    conn.close()

atexit.register(exit_handler)
CORS(app)

pickles_dir = path.join(path.dirname(path.abspath(__file__)), 'torch-stuff')
tokenizer_path = path.join(pickles_dir, 'tokenizer.pt')
day_model = path.join(pickles_dir, 'model_params_day.pt')
week_model = path.join(pickles_dir, 'model_params_week.pt')

vocabulary = 10000
length = 15
features = 200

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocabulary, embedding_dim=features)
        self.convolutions = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, features)),
                                           nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, features)),
                                           nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(10, features))])
        #self.fc1 = nn.Linear(in_features=64 * len(self.convolutions), out_features=32)
        self.fc2 = nn.Linear(in_features=32 * len(self.convolutions), out_features=2)

    def forward(self, x):
        x = self.embeddings(x)
        x = torch.unsqueeze(x, dim=1)
        xs = []
        for index,convolution in enumerate(self.convolutions, start=1):
            c = F.softplus(convolution(x))
            c = torch.squeeze(c, 3)
            c = F.max_pool1d(c, kernel_size=c.size()[2])
            xs.append(c)
        x = torch.cat(xs, dim=2)
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        return x

# unpickle the Tokenizer
device = 'cpu'
tokenizer = torch.load(tokenizer_path)
day_net = ConvNet().to(device)
day_net.load_state_dict(torch.load(day_model))
week_net = ConvNet().to(device)
week_net.load_state_dict(torch.load(week_model))

# https://docs.sqlalchemy.org/en/14/dialects/postgresql.html#dialect-postgresql

@app.route('/predict_day/<text>', methods=['GET'])
def predict_day(text):
    response = {}

    try:
        x = tokenizer.texts_to_sequences([text])
        x = sequence.pad_sequences(x, maxlen=length, truncating='post', padding='pre')
        tensor = torch.tensor(x[0], dtype=torch.long).unsqueeze(0)
        x = day_net.forward(tensor)
        response['softmaxResult'] = x.tolist()[0]
        status = 200
    except Exception as ex:
        response['error'] = f'Encountered error for the input: {ex}'
        status = 500
    return jsonify(response), status

@app.route('/predict_week/<text>', methods=['GET'])
def predict_week(text):
    response = {}

    try:
        x = tokenizer.texts_to_sequences([text])
        x = sequence.pad_sequences(x, maxlen=length, truncating='post', padding='pre')
        tensor = torch.tensor(x[0], dtype=torch.long).unsqueeze(0)
        x = week_net.forward(tensor)
        response['softmaxResult'] = x.tolist()[0]
        status = 200
    except Exception as ex:
        response['error'] = f'Encountered error for the input: {ex}'
        status = 500
    return jsonify(response), status

@app.route('/search/<text>', methods=['GET'])
def search(text):
    response = {}
    try:
        sql = select([article_companies.c.id]).where(
            article_companies.c.search_index.match(f'{text}:*', postgresql_regconfig='english')
        ).order_by(desc(article_companies.c.relevance))
        response['found_ids'] = [ row[0] for row in conn.execute(sql) ]
        status = 200
    except Exception as ex:
        response['error'] = f'Encountered error for the input: {ex}'
        status = 500
    return jsonify(response), status

def format_sayt(row):
    if not (row[3] is None and row[1]) is None:
        return f'[{row[3]}] {row[1]}'
    if not (row[4] is None and row[2] is None):
        return f'[{row[4]}] {row[2]}'
    return row[0]

@app.route('/sayt/<text>', methods=['GET'])
def sayt(text):
    response = {}
    try:
        sql = select(map(column, ['article_entity', 'nasdaq_entity', 'nyse_entity', 'nasdaq_label', 'nyse_label'])).where(
            article_companies.c.search_index.match(f'{text}:*', postgresql_regconfig='english')
        ).order_by(desc(article_companies.c.relevance)).limit(12)
        response['found_ids'] = [ format_sayt(row) for row in conn.execute(sql) ]
        status = 200
    except Exception as ex:
        response['error'] = f'Encountered error for the input: {ex}'
        status = 500
    return jsonify(response), status

def process_result(response, model):
    titles = list(map(lambda row: row[5], response))
    x = tokenizer.texts_to_sequences(titles)
    x = sequence.pad_sequences(x, maxlen=length, truncating='post', padding='pre')
    tensor = torch.tensor(x, dtype=torch.long)
    x = model.forward(tensor)
    predictions = x.tolist()
    return [
        {
            'foundBy': row[0],
            'entity': row[2] if row[1] is None else row[1],
            'stockLabel': row[4] if row[3] is None else row[3],
            'title': row[5],
            'text': row[6],
            'prediction': prediction
        } for row,prediction in zip(response, predictions)
    ]

@app.route('/results_day', methods=['GET'])
def results_day():
    response = {}
    # ids = request.args.getlist('id')
    ids = request.args.getlist('id[]')
    try:
        sql = select(
            [article_companies.c.article_entity] + list(map(column, ['nasdaq_entity', 'nyse_entity', 'nasdaq_label', 'nyse_label', 'title', 'text']))
        ).select_from(
            article_companies.join(article_references)
        ).where(
            article_companies.c.id.in_(tuple(ids))
        ).order_by(desc(article_companies.c.relevance))
        response = process_result(list(conn.execute(sql)), day_net)
        status = 200
    except Exception as ex:
        response['error'] = f'Encountered error for the input: {ex}'
        status = 500
    return jsonify(response), status

@app.route('/results_week', methods=['GET'])
def results_week():
    response = {}
    # ids = request.args.getlist('id')
    ids = request.args.getlist('id[]')
    try:
        sql = select(
            [article_companies.c.article_entity] + list(map(column, ['nasdaq_entity', 'nyse_entity', 'nasdaq_label', 'nyse_label', 'title', 'text']))
        ).select_from(
            article_companies.join(article_references)
        ).where(
            article_companies.c.id.in_(tuple(ids))
        ).order_by(desc(article_companies.c.relevance))
        response = process_result(list(conn.execute(sql)), week_net)
        status = 200
    except Exception as ex:
        response['error'] = f'Encountered error for the input: {ex}'
        status = 500
    return jsonify(response), status


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)

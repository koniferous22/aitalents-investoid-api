# app.py
from flask import Flask, request, jsonify

from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

from os import path
import torch
# import sys
import torch.nn as nn
import torch.nn.functional as F
# import torch. as optim
# from torch.utils.data import DataLoader, Dataset
# import tensorflow as tf
from keras_preprocessing import sequence, text
import numpy as np


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://postgres:postgres_420@database-2.cbtmhzr7rp4s.eu-central-1.rds.amazonaws.com/postgres'
CORS(app)
db = SQLAlchemy(app)

# class ArticleCompanyModel(db.Model):
#     __tablename__ = 'article_companies'
#     index = db.Column(db.Integer, primary_key=True)
#     article_entity = db.Column(db.String(90), unique=True, nullable=False)
#     nasdaq_entity = db.Column(db.String(50))
#     nyse_entity = db.Column(db.String(50))
#     nasdaq_label = db.Column(db.String(5))
#     nyse_label = db.Column(db.String(5))

#     def __repr__(self):
#         return '<Company %r>' % self.article_entity

# class ArticleReferenceModel(db.Model):
#     __tablename__ = 'article_references'
#     index = db.Column(db.Integer, primary_key=True)
#     article_entity = db.Column(db.String(90), unique=True, nullable=False)
#     nasdaq_entity = db.Column(db.String(50))
#     nyse_entity = db.Column(db.String(50))
#     nasdaq_label = db.Column(db.String(5))
#     nyse_label = db.Column(db.String(5))

#     def __repr__(self):
#         return '<Company %r>' % self.article_entity

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


@app.route('/predict_day/<text>', methods=['GET'])
def predict_day(text):
    response = {}

    try:
        x = tokenizer.texts_to_sequences([text])
        x = sequence.pad_sequences(x, maxlen=length, truncating='post', padding='pre')
        tensor = torch.tensor(x[0], dtype=torch.long).unsqueeze(0)
        x = day_net.forward(tensor)
        predicted_class = torch.max(x, dim=1)[1]
        response['class'] = predicted_class.item()
        status = 200
    except Exception as ex:
        response['error'] = f'Encountered error for the input: {ex}'
        status = 500
    print(response)
    return jsonify(response), status

@app.route('/predict_week/<text>', methods=['GET'])
def predict_week(text):
    response = {}

    try:
        x = tokenizer.texts_to_sequences([text])
        x = sequence.pad_sequences(x, maxlen=length, truncating='post', padding='pre')
        tensor = torch.tensor(x[0], dtype=torch.long).unsqueeze(0)
        x = week_net.forward(tensor)
        predicted_class = torch.max(x, dim=1)[1]
        response['class'] = predicted_class.item()
        status = 200
    except Exception as ex:
        response['error'] = f'Encountered error for the input: {ex}'
        status = 500
    print(response)
    return jsonify(response), status

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)

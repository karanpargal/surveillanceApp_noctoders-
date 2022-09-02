import os
import numpy as np
from flask import Flask, request
from flask_cors import CORS
from keras.models import load_model
from PIL import Image, ImageOps

app = Flask(__name__) # new
CORS(app) # new

@app.route('/upload', methods=['POST'])
def upload():
    pass
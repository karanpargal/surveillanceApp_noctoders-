import os
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

model = load_model('model/model.h5', compile=False)
test_dir='./test_seq'
frames = np.zeros(shape=(20, 10, 128, 128, 1))
count=0
for img_name in sorted(os.listdir(test_dir)):
    img_path=os.path.join(test_dir,img_name)
    if str(img_path[-3:])=='tif':
        img = Image.open(img_path).resize((128, 128))
        img = np.array(img, dtype=np.float32) / 128.0 
        frames[count//10, count%10, :, :, 0] = img
        count += 1

reconstructed_sequences = model.predict(frames,batch_size=4)
sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(frames[i],reconstructed_sequences[i])) for i in range(0,20)])
sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
sr = 1.0 - sa

threshold= 0.8
response= ''
for i in sr:
    if i>=threshold:
        response= "Anomaly Detected !!!"
    else:
        response= "No Anomaly..."
print(response)
#!/usr/bin/env python3

#!/usr/bin/env python3
from flask import Flask, request, jsonify
import base64
import io
import cv2
from imageio import imread
import csv
import time


app = Flask(__name__)

@app.route('/test', methods=['GET', 'POST'])
def add_message():
    content = request.json
    # image = imread(io.BytesIO(base64.b64decode(content['image'])))
    with open('my.csv', 'a+', newline='\n') as csvfile:
        name = time.time()
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow([name, content['coordinate'][0], content['coordinate'][1], 'red', content['image']])
        # cv2.imwrite(f"{name}.jpg", image)

    # cv2.imshow('Result', image)
    # cv2.waitKey(5)
    return 'good'

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)
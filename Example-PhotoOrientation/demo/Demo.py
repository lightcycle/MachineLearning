from flask import Flask, jsonify, render_template, request
import tensorflow as tf
import sys
sys.path.append('../train')
from Model import Model

app = Flask(__name__)

keep_prob_placeholder = tf.constant(1.0, tf.float32)
image_placeholder = tf.placeholder(tf.uint8, [None, 256, 256, 3])
output = Model.create_graph(image_placeholder, keep_prob_placeholder)

sess = tf.Session()

saver = tf.train.Saver(sharded=True)
saver.restore(sess, '../train/model.ckpt')


def infer(image):
    return sess.run(output, feed_dict={image_placeholder: image}).flatten().tolist()


def process_jpeg(stream):
    image = tf.image.decode_jpeg(stream.getvalue(), channels=3)
    image.set_shape([258, 344, 3])
    image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
    image = tf.expand_dims(image, 0)
    return image.eval(session = sess)


@app.route('/api/photo_orientation', methods=['POST'])
def infer_photo_orientation():
    f = request.files['webcam']
    f.save('test.jpg')
    image = process_jpeg(f.stream)
    result = infer(image)
    return jsonify(results = result)


@app.route('/')
def index():
    return render_template('index.html')
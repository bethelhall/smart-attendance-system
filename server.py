import flask
import numpy as np
print('setting up keras...')
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
print('loading facenet...')
from facenet import faceRecoModel, triplet_loss, prepare_database, who_is_it

print('creating model...')
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print('compiling model...')
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
print('loading weights to model...')
FRmodel.load_weights('my_face_model_weights.h5')

print('preparing image database...')
database=prepare_database(FRmodel)

app = flask.Flask(__name__)

@app.route('/recognize', methods=['POST', 'GET'])
def recognizer():
    result = 'uninitialized'
    try:
        # for keys,values in flask.request.files.items():
        #     print(keys)
        #     print(values)
        imageFile=flask.request.files['file'].read()
        npimg=np.fromstring(imageFile,np.uint8)
        img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
        print('img size: ', np.size(img))
        #cv2.imwrite('xx.jpeg', img)

        result = who_is_it(img,database,FRmodel)
        print("Identity: ",result)

    except Exception as e:
        print('exception: ', e)
        result = str(e)

    return str(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3333, debug=False)


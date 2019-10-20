import os
import io
import json
import requests
import logging
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handler(data, context):
    """Handle request.
    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """
    processed_input = _process_input(data, context)
    
    logging.info(context.rest_uri)
    response = requests.post(context.rest_uri, data=processed_input)

    return _process_output(response, context)


def _process_input(data, context):
    if context.request_content_type == 'application/json':
        body = data.read().decode('utf-8')

        logging.info(body)
        
        param = json.loads(body)
        query = param['image']
        
        #初期化
        images = []
        
        # JSON⇒base64エンコード文字列に変換する
        b64byte = _tobase64(query)
        
        # base64エンコード文字列⇒Numpy配列に変換する
        img_array = _base64_to_ndarry(b64byte)
        
        #入力用の配列を作成する
        images.append(img_array)
        
        img_np = np.array(images)
        
        logging.info(img_np.shape)
        
        #JSONにシリアライズする
        # ret = json.dumps(img_np,cls = MyEncoder)
        ret = img_np.tolist()
        
        logging.info(ret)
        
        return json.dumps({
            'inputs': ret
        })
    
        # return json.dumps({
        #    'inputs': ret
        # })

    raise ValueError('{"error": "unsupported content type {}"}'.format(
        context.request_content_type or "unknown"))


def _process_output(data, context):
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = 'application/json'

    body = json.loads(data.content.decode('utf-8'))
    
    logging.info(body)
    predicts = body['outputs']

    # labels_path = '/opt/ml/model/code/labels.pickle'

    # with open(labels_path, mode='rb') as f:
    #     labels = pickle.load(f)
    # rets = _create_response(predicts, labels)
    rets = _create_response(predicts)

    logger.warn(rets)

    return json.dumps(rets), response_content_type

def _create_response(predicts):
    rets = []

    age = 0
    if 'pred_age_1/Softmax:0' in predicts:
        age = predicts['pred_age_1/Softmax:0']
    else:
        age = predicts['pred_age/Softmax:0']
        
    prob = np.argmax(np.array(age))
    
    rets.append({
            'prob': int(prob)
        })

    return rets

def _tobase64(strjson):
    """
    文字列をbase64文字列にエンコードする
    """
    #strjson = enc.decode()
    #x = json.loads(strx)
    y = strjson.encode()
    
    return y
    
def _base64_to_ndarry(img_base64):
    """
    Base64でエンコードされた画像文字列をNumpu配列に変換します。
    """
    im = Image.open(BytesIO(base64.b64decode(img_base64)))
    
    im_size = 64
    # im = im.resize( (32, 32) )
    im = im.resize( (im_size, im_size) )
    img_array = np.asarray(im)
    
    return img_array
    

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
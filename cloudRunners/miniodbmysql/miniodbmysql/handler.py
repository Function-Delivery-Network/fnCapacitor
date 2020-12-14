from minio import Minio
from minio.error import ResponseError
import requests
import json
from time import time



minioClient = Minio('138.246.234.122:9000',
                   access_key='access',
                   secret_key='secretkey',
                   secure=False)


def handle(params):
    start = time()
    r = requests.get('http://ipaddress:31112/function/mysqlside')
    r = r.json()
    data = minioClient.get_object(r['bucket_name'], r['image_name'])
    with open('my-testfile.jpg', 'wb') as file_data:
        for d in data.stream(32 * 1024):
            file_data.write(d)
    lat = time() - start
    ret_val = {}
    ret_val['latency'] = lat
    return ret_val
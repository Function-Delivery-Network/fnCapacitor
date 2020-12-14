import requests
import json
from time import time


def handle(params):
    start = time()
    r = requests.get('http://ipaddress:31112/function/mysqlside')
    r1 = requests.get('http://ipaddress:31112/function/miniodb')
    lat = time() - start
    ret_val = {}
    ret_val['latency'] = lat
    return ret_val
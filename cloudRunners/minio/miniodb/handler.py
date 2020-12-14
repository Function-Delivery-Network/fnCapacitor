from minio import Minio
from minio.error import ResponseError
from PIL import Image, ImageFilter
from time import time


def flip(image):
    img_1 = image.transpose(Image.FLIP_LEFT_RIGHT)
    img_2 = image.transpose(Image.FLIP_TOP_BOTTOM)
    return [img_1, img_2]

def rotate(image):
    img_1 = image.transpose(Image.ROTATE_90)
    img_2 = image.transpose(Image.ROTATE_180)
    img_3 = image.transpose(Image.ROTATE_270)
    return [img_1, img_2, img_3]

def filter(image):
    img_1 = image.filter(ImageFilter.BLUR)
    img_2 = image.filter(ImageFilter.CONTOUR)
    img_3 = image.filter(ImageFilter.SHARPEN)
    return [img_1, img_2, img_3]

def gray_scale(image):
    img = image.convert('L')
    return img

def resize(image):
    img = image.thumbnail((128, 128))
    return img


minioClient = Minio('138.246.234.122:9000',
                   access_key='access',
                   secret_key='secretkey',
                   secure=False)


def handle(params):
    data = minioClient.get_object('test', 'website.jpg')
    with open('my-testfile.jpg', 'wb') as file_data:
        for d in data.stream(32 * 1024):
            file_data.write(d)
    image = Image.open('my-testfile.jpg')
    if image.mode != 'RGB':
        image = image.convert('RGB')

    start = time()
    flip(image)
    rotate(image)
    filter(image)
    gray_scale(image)
    resize(image)
    lat = time() - start

    ret_val = {}

    ret_val['latency'] = lat
    print(ret_val)
    return ret_val
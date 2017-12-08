import base64
import cv2
import numpy as np
from skimage import measure

def decode_image(img, npx):
    img = base64.b64decode(img)
    img = np.fromstring(img, dtype=np.uint8)
    img = cv2.imdecode(img, 0)
    img = cv2.resize(img, (npx, npx), interpolation=cv2.INTER_AREA)
    return np.greater(img, 0).astype(np.float32)

def encode_image(img):
    ret, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf)

def tanh2uint16(img):
    img = np.squeeze(img)
    img = np.clip(img, -1, 1)
    img = (img + 1) * 32767.5
    return img.astype(np.uint16)

def extract_mesh(volume, level):
    spacing = (0.1, 0.1, 0.1)
    # volume = ndimage.filters.gaussian_filter(volume, 1., mode='constant', cval=0.)
    verts, faces, _, _ = measure.marching_cubes_lewiner(volume, level, spacing, gradient_direction='ascent')
    return dict(verts=verts.tolist(), faces=faces.tolist())

def convert2obj(verts, faces):
    text = ''
    for vert in verts:
        text += 'v {0} {1} {2}\n'.format(vert[0], vert[1], vert[2])
    for face in faces:
        text += 'f {0} {1} {2}\n'.format(face[0]+1, face[1]+1, face[2]+1)
    return text

import shutil
import os

""" Las imágenes estaban dentro de carpetas, dentro de la carpeta IncorrectMask o WithMask, por lo que con este script
    hemos juntado todas las imagenes en el mismo directorio respectivamente """

# source = "G:\JUEGOS\IMFD\IncorrectMask"
# destination = "G:\JUEGOS\IMFD\IncorrectMask"

# source = "G:\JUEGOS\CMFD\WithMask"
# destination = "G:\JUEGOS\CMFD\WithMask"

# files = os.listdir(source)

# for r, f in os.walk(source):
#     for files in f:
#         shutil.move(f"{os.path.join(r,files)}", destination)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

    
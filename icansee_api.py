import cv2
import json
import time

from icansee_api_lib import ICanSeeApiLib

icanseeapi = ICanSeeApiLib()

icanseeapi.init()

imageSrc = cv2.imread("image.jpg")

while True:
    result = icanseeapi.process(imageSrc)
    print(json.dumps(result))
    time.sleep(0.5)

icanseeapi.close()
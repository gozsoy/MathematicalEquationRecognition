import sys
import base64
import requests
import json

# put desired file path here
file_path = 'nowtry.png'

image_uri = "data:image/jpg;base64," + base64.b64encode(open(file_path, "rb").read()).decode()
r = requests.post("https://api.mathpix.com/v3/latex",
    data=json.dumps({'src': image_uri}),
    headers={"app_id": 'your-id-here', "app_key": 'your-appkey-here',
            "Content-type": "application/json"})
print(json.dumps(json.loads(r.text), indent=4, sort_keys=True))


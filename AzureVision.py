
########### Python 3.2 #############
import http.client, urllib.request, urllib.parse, urllib.error, base64,json

def getJSON(image_name):

    headers = {
        # Request headers
        'Prediction-Key': 'your-prediction-key-here',
        'Content-Type': 'application/octet-stream',
        'Prediction-key': 'your-prediction-key-here',
    }

    params = urllib.parse.urlencode({
        # Request parameters
        'application': '{string}',
    })

    try:
        data = open(image_name, 'rb').read()
        conn = http.client.HTTPSConnection('northcentralus.api.cognitive.microsoft.com')
        conn.request("POST","/customvision/v3.0/Prediction/your-info-here/detect/iterations/your-projectname-here/image?%s" % params,data, headers)
        response = conn.getresponse()
        data = response.read()
        #print(data)
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))

    parsed_json = json.loads(data)
    prediction_data=parsed_json['predictions']

    return prediction_data
import requests
import demjson


def ocr_space_file(filename, overlay=True, api_key='0d5c85b2d388957', language='eng'):
    """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload,
                          )
    return r.content


def ocr_space_url(url, overlay=False, api_key='helloworld', language='eng'):
    """ OCR.space API request with remote file.
        Python3.5 - not tested on 2.7
    :param url: Image url.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'url': url,
               'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    r = requests.post('https://api.ocr.space/parse/image',
                      data=payload,
                      )
    return r.content.decode()


def image_to_boxes(image, api_key='helloworld'):

    false = False

    true = True

    text_file = ocr_space_file(image, True, api_key)

    text = demjson.decode(text_file)

    words = text['ParsedResults'][0]['TextOverlay']['Lines']

    image_boxes = []

    for i in range(len(words)):

        temp = words[i]['Words']

        for j in range(len(temp)):

            print()

            image_boxes.append([temp[j]['WordText'], temp[j]['Left'], temp[j]['Top'], temp[j]['Left']+temp[j]['Width'],
                                temp[j]['Top']+temp[j]['Height'], 0])

    return image_boxes

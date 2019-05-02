# API Guide

### Request
* Method: POST
* URL: http://ipaddress:51001/api

### Request Parameters

| Key | 설명 | type |
| - | - | - |
| text | 듣고 싶은 문장을 입력 | string |
| neu | neutral 정도 (0.0 ~ 1.0) | string |
| hap | happy 정도 (0.0 ~ 1.0) | string |
| sad | sad 정도 (0.0 ~ 1.0) | string |
| ang| angry 정도 (0.0 ~ 1.0) | string |

### Requests 예제
```
curl --request POST \
 --header 'Content-Type: application/json' \
 --data '{"text":"안녕하세요.", "neu":"1.0", "hap":"0.0", "sad":"0.0", "ang":"0.0"}' \
 http://ipaddress:51001/api
 ```

### Response Parameters
* Response

| Key | 설명 | type |
| - | - | - |
| params | request parameters (text, neu, hap, sad, ang) | object |
| data | TTS 결과 (base64) | string |

* params

| Key | 설명 | type |
| - | - | - |
| text | 듣고 싶은 문장을 입력 | string |
| neu | neutral 정도 (0.0 ~ 1.0) | float |
| hap | happy 정도 (0.0 ~ 1.0) | float |
| sad | sad 정도 (0.0 ~ 1.0) | float |
| ang| angry 정도 (0.0 ~ 1.0) | float |


### Response 예제
```
{
    "params": {
        "text": "\uc548\ub155\ud558\uc138\uc694.", 
        "neu": 1.0, 
        "hap": 0.0, 
        "sad": 0.0, 
        "ang": 0.0
    }, 
    "data": "UklGRj..."
}
```
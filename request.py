import requests


def request_get(url,data):
    response = requests.get(url, json=data)
    if response.status_code == 200:
        data = response.json()
    else:
        raise Exception("Error fetching data from API")

    return data


def request_post(url,data):

    response = requests.post(url, data=data)
    if response.status_code == 200:
        data = response.json()
    else:
        raise Exception("Error fetching data from API")
    return data
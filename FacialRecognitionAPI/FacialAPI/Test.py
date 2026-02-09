import requests

url = "http://127.0.0.1:8000/recognize"
files = {"file": open("person.jpg", "rb")}

response = requests.post(url, files=files)
print(response.json())

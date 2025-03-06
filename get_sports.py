import http.client
import gzip
import json
import pandas as pd
from io import BytesIO

# API Request
conn = http.client.HTTPSConnection("pinnacle-odds.p.rapidapi.com")

headers = {
    'x-rapidapi-key': "93800e0a2bmsh9c218f95919c93ap134635jsn1daab843f20d",
    'x-rapidapi-host': "pinnacle-odds.p.rapidapi.com",
    'Accept-Encoding': 'gzip'  # Ensure server sends gzip response
}

conn.request("GET", "/kit/v1/sports", headers=headers)

res = conn.getresponse()
compressed_data = res.read()  # Read compressed response

# Decompress the response
with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as f:
    decompressed_data = f.read()

# Convert bytes to string and parse JSON
data_str = decompressed_data.decode("utf-8")
data_json = json.loads(data_str)

print(data_json)
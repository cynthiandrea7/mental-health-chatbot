from pymongo import MongoClient
import urllib.parse
import json


with open('credentials_mongodb.json') as f:
    login = json.load(f)
# assign credentials to variables
username = login['username']
password = urllib.parse.quote(login['password'])
host = login['host']
url = "mongodb+srv://{}:{}@{}/?retryWrites=true&w=majority".format(username, password, host)

# 1. Connect to MongoDB
client = MongoClient(url)
db = client["mental_health_db"]
collection = db["intents"]

# 2. Load JSON file
with open("C:\\Projects\\mental-health-chatbot\\KB.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# 3. Upload
if "intents" in data:
    collection.insert_many(data["intents"])
    print(f"✅ Inserted {len(data['intents'])} documents into MongoDB.")
else:
    print("⚠️ JSON file must contain an 'intents' array.")
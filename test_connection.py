"""Test MongoDB Connection"""
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing MongoDB connection...")
print("=" * 50)

try:
    mongo_uri = os.getenv('MONGODB_URI')
    
    if not mongo_uri:
        print("❌ MONGODB_URI not found in .env file!")
        exit(1)
    
    client = MongoClient(mongo_uri)
    client.server_info()
    
    print("✅ MongoDB connected successfully!")
    print(f"✅ Server version: {client.server_info()['version']}")
    print(f"✅ Databases: {client.list_database_names()}")
    
    print("=" * 50)
    print("✅ CONNECTION TEST PASSED!")
    
    client.close()
    
except Exception as e:
    print("=" * 50)
    print(f"❌ ERROR: {e}")
    print("\nCheck:")
    print("1. MONGODB_URI in .env file")
    print("2. MongoDB password is correct")
    print("3. IP whitelist in MongoDB Atlas")
import requests

api_key = "1cwGoU29yCb6DDrXfvwL"  # Your Private API Key
headers = {
    "Authorization": f"Bearer {api_key}"
}

# Attempt to list workspaces via a GET request
response = requests.get("https://api.roboflow.com/my/workspaces", headers=headers)

if response.status_code == 200:
    workspaces = response.json()
    print("Available Workspaces:")
    for workspace in workspaces['workspaces']:
        print(f" - {workspace['name']} (ID: {workspace['id']})")
else:
    print(f"Error: {response.status_code}")
    print(response.json())

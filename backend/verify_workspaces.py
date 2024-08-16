from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow(api_key="1cwGoU29yCb6DDrXfvwL")

# Get and print information about a specific workspace
# Since the method to list all workspaces isn't available, we'll manually access a known workspace
try:
    workspace = rf.workspace("FootballVideoTrackingApp")
    print(f"Workspace Name: {workspace.name}")
    print("Available Projects in this Workspace:")
    for project in workspace.projects():
        print(f" - {project.name}")
except Exception as e:
    print(f"Error: {e}")



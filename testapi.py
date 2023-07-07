import requests

username = 'bho29'
api_token = 'NjkzNzk3MTI0Njg0OiW+9KC+e5LMsLM1F6ZvvVapu6m3'
jira_url = 'https://jira.dxc.com'

auth = (username, api_token)

def update_issue(issue_key):
    url = f"{jira_url}/rest/api/2/issue/{issue_key}"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "fields": {
            "customfield_16029": [
                {
                    "value": "UAT"
                }
            ]
        }
    }
    response = requests.put(url, json=data, auth=auth, headers=headers)
    if response.status_code == 204:
        print(f"Updated issue {issue_key} successfully.")
    else:
        print(f"Failed to update issue {issue_key}. Status code: {response.status_code}")

# Example usage:
issues_to_update = [
    {"key": "PGA-3574"}
    # Add more issues as needed
]

for issue in issues_to_update:
    update_issue(issue['key'])

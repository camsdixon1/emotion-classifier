"""
List all files/folders in Google Drive to find transcripts.
"""
import json
import requests

with open('/tmp/gdrive_token.json') as f:
    tokens = json.load(f)

access_token = tokens['access_token']
headers = {'Authorization': f'Bearer {access_token}'}


def list_files(query=None, page_token=None):
    params = {
        'pageSize': 100,
        'fields': 'nextPageToken, files(id, name, mimeType, size, modifiedTime, parents)',
        'orderBy': 'modifiedTime desc',
    }
    if query:
        params['q'] = query
    if page_token:
        params['pageToken'] = page_token
    resp = requests.get('https://www.googleapis.com/drive/v3/files', headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()


def get_file_content(file_id, mime_type):
    """Download file content."""
    if 'google-apps.document' in mime_type:
        # Export Google Doc as plain text
        resp = requests.get(
            f'https://www.googleapis.com/drive/v3/files/{file_id}/export',
            headers=headers,
            params={'mimeType': 'text/plain'}
        )
    else:
        resp = requests.get(
            f'https://www.googleapis.com/drive/v3/files/{file_id}',
            headers=headers,
            params={'alt': 'media'}
        )
    resp.raise_for_status()
    return resp.text


print("=== ALL FILES IN GOOGLE DRIVE ===\n")
result = list_files()
all_files = result.get('files', [])

# Paginate
while 'nextPageToken' in result:
    result = list_files(page_token=result['nextPageToken'])
    all_files.extend(result.get('files', []))

for f in all_files:
    size = f.get('size', 'N/A')
    print(f"[{f['mimeType'].split('.')[-1][:20]}] {f['name']} (id: {f['id']}, size: {size})")

print(f"\nTotal files: {len(all_files)}")

# Look for transcript-like files
print("\n=== LIKELY TRANSCRIPT FILES ===\n")
keywords = ['transcript', 'call', 'sales', 'recording', 'conversation', 'client', 'meeting', 'demo', 'discovery']
for f in all_files:
    name_lower = f['name'].lower()
    if any(k in name_lower for k in keywords):
        print(f"  >> {f['name']} ({f['mimeType']}, id: {f['id']})")

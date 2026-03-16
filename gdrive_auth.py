"""
Google Drive OAuth2 flow - starts local server on port 8080 to capture auth code.
Run this script and visit the printed URL to authorize access.
"""
import os
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import google.oauth2.credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

CLIENT_ID = os.environ['GOOGLE_CLIENT_ID']
CLIENT_SECRET = os.environ['GOOGLE_CLIENT_SECRET']
REDIRECT_URI = os.environ.get('GOOGLE_REDIRECT_URI', 'http://localhost:8080/callback')

SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/drive.metadata.readonly',
]

client_config = {
    "web": {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uris": [REDIRECT_URI],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
}

auth_code = None
server_done = threading.Event()


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global auth_code
        parsed = urlparse(self.path)
        if parsed.path == '/callback':
            params = parse_qs(parsed.query)
            if 'code' in params:
                auth_code = params['code'][0]
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"<h1>Authorization successful! You can close this tab.</h1>")
                server_done.set()
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"<h1>Error: no code received</h1>")

    def log_message(self, format, *args):
        pass  # suppress logs


def main():
    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )

    auth_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent',
    )

    print("\n" + "="*70)
    print("PLEASE VISIT THIS URL TO AUTHORIZE GOOGLE DRIVE ACCESS:")
    print("="*70)
    print(auth_url)
    print("="*70 + "\n")
    print("Waiting for authorization on port 8080...")

    server = HTTPServer(('0.0.0.0', 8080), CallbackHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    server_done.wait(timeout=300)
    server.shutdown()

    if not auth_code:
        print("Timed out waiting for authorization.")
        return None

    flow.fetch_token(code=auth_code)
    creds = flow.credentials

    # Save tokens
    token_data = {
        'token': creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': list(creds.scopes) if creds.scopes else SCOPES,
    }
    with open('/tmp/gdrive_token.json', 'w') as f:
        json.dump(token_data, f)

    print("Authorization successful! Token saved to /tmp/gdrive_token.json")
    return creds


if __name__ == '__main__':
    main()

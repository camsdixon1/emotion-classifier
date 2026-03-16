"""
Google Drive OAuth2 flow using raw requests (no google-auth library).
"""
import os
import json
import threading
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, urlencode

CLIENT_ID = os.environ['GOOGLE_CLIENT_ID']
CLIENT_SECRET = os.environ['GOOGLE_CLIENT_SECRET']
REDIRECT_URI = os.environ.get('GOOGLE_REDIRECT_URI', 'http://localhost:8080/callback')

SCOPES = 'https://www.googleapis.com/auth/drive.readonly'

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
                error = params.get('error', ['unknown'])[0]
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"<h1>Error: {error}</h1>".encode())
                server_done.set()

    def log_message(self, format, *args):
        pass


def build_auth_url():
    params = {
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'response_type': 'code',
        'scope': SCOPES,
        'access_type': 'offline',
        'prompt': 'consent',
    }
    return 'https://accounts.google.com/o/oauth2/v2/auth?' + urlencode(params)


def exchange_code(code):
    resp = requests.post('https://oauth2.googleapis.com/token', data={
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'redirect_uri': REDIRECT_URI,
        'grant_type': 'authorization_code',
        'code': code,
    })
    resp.raise_for_status()
    return resp.json()


def main():
    auth_url = build_auth_url()

    print("\n" + "="*70)
    print("PLEASE VISIT THIS URL TO AUTHORIZE GOOGLE DRIVE ACCESS:")
    print("="*70)
    print(auth_url)
    print("="*70 + "\n")
    print("Waiting for authorization callback on port 8080...")

    server = HTTPServer(('0.0.0.0', 8080), CallbackHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    server_done.wait(timeout=300)
    server.shutdown()

    if not auth_code:
        print("Timed out or error during authorization.")
        return None

    print("Got auth code, exchanging for tokens...")
    tokens = exchange_code(auth_code)

    with open('/tmp/gdrive_token.json', 'w') as f:
        json.dump(tokens, f, indent=2)

    print("Token saved to /tmp/gdrive_token.json")
    print(f"Access token: {tokens.get('access_token', '')[:20]}...")
    return tokens


if __name__ == '__main__':
    main()

"""
Exchange an auth code for Google Drive tokens.
Usage: python3 gdrive_token_exchange.py <auth_code_or_full_callback_url>
"""
import os
import sys
import json
import requests
from urllib.parse import urlparse, parse_qs

CLIENT_ID = os.environ['GOOGLE_CLIENT_ID']
CLIENT_SECRET = os.environ['GOOGLE_CLIENT_SECRET']
REDIRECT_URI = os.environ.get('GOOGLE_REDIRECT_URI', 'http://localhost:8080/callback')


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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 gdrive_token_exchange.py <auth_code_or_full_url>")
        sys.exit(1)

    arg = sys.argv[1]

    # If they pasted the full redirect URL, extract the code
    if arg.startswith('http'):
        parsed = urlparse(arg)
        params = parse_qs(parsed.query)
        code = params.get('code', [None])[0]
        if not code:
            print("No 'code' parameter found in URL")
            sys.exit(1)
    else:
        code = arg

    print(f"Exchanging code: {code[:20]}...")
    tokens = exchange_code(code)

    with open('/tmp/gdrive_token.json', 'w') as f:
        json.dump(tokens, f, indent=2)

    print("Success! Token saved to /tmp/gdrive_token.json")
    print(f"Scopes: {tokens.get('scope', 'N/A')}")

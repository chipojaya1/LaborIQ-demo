# Running the LaborIQ FastAPI backend from Google Colab with ngrok

The following notebook cells spin up the demo backend in Google Colab and expose it publicly via ngrok so that the Figma Make prototype can reach it.

> ⚠️ **ngrok auth token**: Replace `"YOUR_NGROK_AUTHTOKEN"` with a token from your ngrok dashboard. A free account is sufficient.

## 1. Install dependencies and clone the repo

```python
!pip install --quiet "uvicorn[standard]" fastapi pyngrok==6.0.0 nest_asyncio
!git clone https://github.com/your-org/LaborIQ-demo.git
%cd LaborIQ-demo
!pip install --quiet -r backend/requirements.txt
```

## 2. Launch ngrok and start the FastAPI server

```python
import nest_asyncio
import threading
from pyngrok import ngrok
import uvicorn

# Allow nested event loops inside the notebook runtime
nest_asyncio.apply()

# Start an HTTP tunnel to the notebook's port 8000
ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")
public_url = ngrok.connect(8000, "http")
print(f"Public URL: {public_url.public_url}")

# Boot the FastAPI app in a background thread
config = uvicorn.Config("backend.app.main:app", host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
```

Once the last cell finishes you should see logs showing the server is accepting requests. The `Public URL` printed by ngrok is the address you should use in Figma Make for the chatbot endpoints (for example, `https://<random-subdomain>.ngrok.app/api/chat`).

## 3. (Optional) Verify the deployment

```python
import requests

response = requests.get(f"{public_url.public_url}/health")
response.raise_for_status()
response.json()
```

If everything is configured correctly the cell will return:

```python
{'status': 'ok'}
```

You can now point the Figma Make HTTP Request block to `Public URL + /api/chat` (or the other endpoints) to exercise the complete chatbot experience.

# Connecting the FastAPI backend to the Figma Make chatbot UI

This guide outlines how to plug the FastAPI backend in `backend/app/main.py` into the [LaborIQ Figma Make project](https://www.figma.com/make/60hSiyrWMz0J9tSwU6qUvs/LaborIQ---Figma-Make?node-id=0-1&p=f&t=1bUWuBVGt4MCljT9-0).

## 1. Deploy the backend

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt
   ```
2. **Run the FastAPI server**
   ```bash
   uvicorn backend.app.main:app --reload
   ```
   The API will be available on `http://127.0.0.1:8000` (or your chosen host).

## 2. Figma Make data connections

In Figma Make, data connections let you call external APIs. Configure a REST data connection pointing to the running FastAPI server:

1. Open the **Data** panel and create a new **HTTP Request** data source.
2. Set the **Method** to `POST` and the **URL** to `http://127.0.0.1:8000/api/chat`.
3. Add the following JSON payload template:
   ```json
   {
     "question": "{{ user_input }}",
     "mode": "{{ mode }}"
   }
   ```
4. Define `user_input` and `mode` as data variables in your prototype. `mode` should be either `market_insights` or `skills_analysis` depending on the path a user takes in the flow.
5. Map the response to your UI elements:
   - Bind `response.answer` to the chatbot response bubble.
   - Bind `response.payload.states` to a table or list for state-level metrics.
   - Bind `response.payload.skills` to a pill-list or tag component highlighting top skills.

For dedicated insight views you can connect straight to the specialised endpoints:

| Purpose | Method | URL | Notes |
| --- | --- | --- | --- |
| State-level salary rankings | `POST` | `http://127.0.0.1:8000/api/insights` | Body: `{ "occupation": "Data Scientist", "metric": "average_salary", "top_n": 5 }` |
| Macro trend charts | `GET` | `http://127.0.0.1:8000/api/bls/LNS14000000?periods=12` | Suitable for line charts. |
| Skill extraction | `POST` | `http://127.0.0.1:8000/api/skills` | Body: `{ "job_description": "..." }`. |

## 3. Suggested UI wiring

* **Entry point** – capture the user question in a `Text Input` component and pass it as `user_input`.
* **Mode selection** – attach buttons or a segmented control to set the `mode` variable.
* **Results** –
  * Show the chatbot text response.
  * For market insights, use a stacked list with state name, state code and salary.
  * For skills, render chip components with the returned `skill` values.
* **Sources** – optionally display `response.sources` as small-print references.

## 4. Extending the backend later

The `ChatResponse.payload` structure leaves room for richer analytics (e.g. charts, percentiles) without breaking the UI. When the real search or RAG models are ready, replace the placeholder logic in:

* `backend/app/services/market_insights.py`
* `backend/app/services/skills_analysis.py`

Because the API contract stays the same, no Figma Make wiring changes will be required.

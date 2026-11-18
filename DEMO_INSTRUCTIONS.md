# Run the dashboard

Activate the project environment and run each command in its own terminal from the checkout.

```sh
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

```sh
python -m http.server 8080 --bind 127.0.0.1 --directory ui
```

Open http://localhost:8080 and click **Load trips**. Choose a trip, a model and **Compare both**, then **Match route**. Toggle the GPS, classifier and Viterbi layers or use **Fit trip**. The metrics describe the selected trip. The map uses external Leaflet assets and OpenStreetMap tiles.

## Two laptops

On the API host, set `PATHSNAP_CORS_ORIGINS` to the exact browser origin, including scheme and port. Start Uvicorn with `--host 0.0.0.0`. Serve the dashboard with `--bind 0.0.0.0` if the browser accesses it across the LAN. The API address defaults to the dashboard hostname on port 8000. For separate hosts use the API address field or append `?api=http://BACKEND_HOST:8000` to the dashboard URL. For HTTPS use a compatible HTTPS API origin.

Open `/trips` on the backend from the browser laptop to diagnose routing, firewall and server startup. If that succeeds but the dashboard fails, check the browser origin against the backend allowlist. A 503 indicates missing data or bundle. An unknown trip returns 404. The API root has no dashboard.

For inference-only inputs, set `PATHSNAP_TRIPS` to the directory of numeric trip folders. Ship only track, nodes and arcs. Proxy scoring is available only on the configured student development split. Physical two-device operation needs an on-site check.

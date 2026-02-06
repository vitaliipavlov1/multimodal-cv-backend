# Nginx Reverse Proxy (Reference Configuration)

This directory contains **reference Nginx configuration** for the project.

It is intended for **host-based Nginx** (not Dockerized) acting as:
- HTTPS reverse proxy
- WebSocket proxy
- HTTP → HTTPS redirect

These files are **NOT automatically used** by Nginx.
They serve as **documentation and a reproducible reference** for production deployment.

---

## Files

### `backend.conf`
Reverse proxy for FastAPI backend.

- Proxies HTTP requests to the backend container
- Supports long-lived WebSocket connections (`/ws/inference`)
- Exposes backend health endpoint (`/health`)
- Designed for real-time inference workloads

### `http_redirect.conf`
HTTP → HTTPS redirect.

- Redirects all plain HTTP traffic to HTTPS
- Intended to be used together with `backend.conf`

---

## How to use on a new server

> ⚠️ These steps are **manual** and intended for host-based Nginx.

### 1. Copy configs to Nginx

```bash
sudo cp backend.conf /etc/nginx/sites-available/project_backend
sudo cp http_redirect.conf /etc/nginx/sites-available/project_http
```

---

### 2. Edit domain name

Open both files and replace:

```
YOUR_DOMAIN
```

with your real domain, for example:

```
api.example.com
```

---

### 3. Enable the sites

```bash
sudo ln -s /etc/nginx/sites-available/project_backend /etc/nginx/sites-enabled/
sudo ln -s /etc/nginx/sites-available/project_http /etc/nginx/sites-enabled/
```

---

### 4. Configure SSL (Certbot example)

```bash
sudo certbot --nginx -d api.example.com
```

This automatically:

- obtains HTTPS certificates
- injects `ssl_certificate` directives into Nginx
- enables HTTPS
- configures auto-renewal

SSL certificates and private keys are never stored in this repository.

---

### 5. Test and reload Nginx

```bash
sudo nginx -t
sudo systemctl reload nginx
```

---

## Notes

- WebSocket timeouts are intentionally increased to support long-running inference
- Backend is expected to be reachable at `backend:8000` (Docker Compose network)
- These configs are safe to commit to GitHub (no secrets, no private data)

---

## Production Setup Summary

Client (HTTPS + WS)
        ↓
     Nginx
        ↓
 FastAPI backend
        ↓
 Triton Inference Server (GPU)

This setup ensures:

- Secure HTTPS access
- Stable WebSocket connections
- Clear separation of infrastructure and application logic

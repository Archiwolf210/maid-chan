# Tailscale Setup for Remote Access (v8.2)

Connect to your Digital Human from your phone, tablet or laptop **without** opening any router ports or going through the public internet.

---

## 1. Install Tailscale

1. Download for your OS: https://tailscale.com/download
2. Install on the home PC that runs the bot.
3. Install on every client device (phone, tablet, laptop).
4. Sign in with the **same** account on all of them.

---

## 2. Two remote modes -- pick one

Digital Human v8.2 supports two remote-access models through the same config.

### 2.1 `local_trusted` (default) -- profile picker for trusted LAN / household

`config.json`:
```json
"server": {
  "host": "127.0.0.1",
  "port": 5000,
  "remote_mode": "local_trusted"
}
```

 - UI shows the user switcher -- you can maintain several profiles, each with its own memory.
 - Use when you share the tailnet with trusted humans (family) and want per-person personalities.
 - `X-User-Id` header is honoured from any reachable device.

### 2.2 `tailscale_single_owner` -- solo remote owner (recommended for phone access)

`config.json`:
```json
"server": {
  "host": "127.0.0.1",
  "port": 5000,
  "remote_mode": "tailscale_single_owner"
}
```

 - All requests (local **and** remote) are forced onto the `master` profile.
 - The UI auto-detects this via `/api/status` -> `single_owner: true` and **hides** the profile picker.
 - Use when *only you* connect remotely and you want tamper-proof identity regardless of what the client sends in `X-User-Id`.

Additionally, even in `local_trusted` mode, **every request arriving from a non-loopback address** (i.e. through Tailscale Serve) is treated as single-owner -- so the master profile is protected by default when you expose the bot remotely.

---

## 3. Expose inside your tailnet

After launching the bot with `start.bat`, in PowerShell on the host:

```powershell
tailscale serve --bg localhost:5000
tailscale serve status
```

Your bot URL: `https://<your-pc-name>.<tailnet>.ts.net`

On your phone:
1. Open Tailscale app -> ensure it's **On**.
2. Open a browser -> paste the URL above.
3. Done -- full HTTPS inside your tailnet only.

**Do NOT run `tailscale funnel`**: that makes the bot public on the open internet.

---

## 4. Security model

| Layer | What it does |
|-------|--------------|
| `app_token.txt` | Auto-generated per install. Required for all POST/PUT/DELETE/PATCH via `X-App-Token` header. The UI fetches it from `/api/token` only when requested by loopback. |
| `resolved_uid` dependency | Server-side resolver. In single-owner mode (or any non-loopback request) returns `'master'` regardless of what the client sent. |
| Tailscale identity | `Tailscale-User-Login` header can be read for audit logging. Don't trust the client `X-User-Id` for remote access. |
| CORS | Locked to `http://127.0.0.1:5000` + `http://localhost:5000` by default. Add explicit entries to `server.cors_origins` if you ever proxy externally. |

Operational advice:
 - Enable **MFA** on your Tailscale account.
 - Enable **device approval** in the Tailscale admin panel.
 - Do not add the host to a funnel or public reverse-proxy.
 - Rotate `app_token.txt` by deleting the file and restarting -- a new token is generated.

---

## 5. Verifying the remote_mode takes effect

```powershell
# On the host
curl http://127.0.0.1:5000/api/status
# expect: remote_mode in {local_trusted, tailscale_single_owner}, single_owner=false (local)

# From your phone, same endpoint via the ts.net URL:
#   single_owner should be TRUE because the request is non-loopback.
```

If `single_owner` is `true`, the frontend will:
 - hide the user-picker modal,
 - pin the pill label to "Хозяин",
 - force all API calls onto the `master` profile.

---

## 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| Phone returns 403 from `/api/chat` | Tailscale Serve isn't active, or peer IP is loopback on your phone (bridged adapter). Check `tailscale serve status`. |
| Can't see the user-picker after enabling `tailscale_single_owner` | By design -- profile is pinned to master. Edit `config.json` and restart to switch back. |
| "LLM unavailable" remotely but works locally | llama-server is only listening on 127.0.0.1 (correct). The *bot* is the only thing Tailscale needs to reach; LLM stays internal. |
| Want to stop remote access temporarily | `tailscale serve --bg --off` |

---

## 7. Multi-user mode via Tailscale identity (future)

The `Tailscale-User-Login` header is already captured by `_is_remote_request`. A future update will map that header to a distinct `user_id` so several humans inside your tailnet can each have their own personality evolution. Not required for the personal-companion use case.

# CardScope v0.7 Deployment

Deploy the v0.7 platform to Ubuntu 22.04 or 24.04 with the scripts in
`deployment/server`. The systemd service starts `platform_server.py` on
`127.0.0.1:8765`; Caddy or Nginx terminates HTTPS.

Mutable production data belongs in `/var/lib/cardscope/platform_workspace`.
It contains SQLite state, uploaded images, reference-image assets, inspection
JSON exports and training state. Releases are installed below
`/opt/cardscope/releases`, so release updates do not replace business data.

After installation, verify enterprise login, a traditional image upload, a
reference-registration upload with both user-provided images, authenticated
`result.json` download, a manual correction, and administrator approval into
the training pool.

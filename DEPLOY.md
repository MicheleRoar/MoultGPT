# Deploying the MoultGPT demo on your own server

Target: a public demo at `https://moultgpt.michele-leone.com` (any subdomain works).

Stack (all in `docker-compose.prod.yml`): GROBID, LLM backend, vision backend,
gateway (rate limiting + upload cap + endpoint allowlist), and one entry point
(Cloudflare Tunnel **or** Caddy).

> The vision backend serves the models in `vision/models/`. Before presenting
> numbers publicly, check that those weights match what the paper describes
> (see the project notes on the grouped-split model and the unified classifier).

## 0. Server requirements

- Linux with Docker + Docker Compose v2 (`docker compose version`).
- **8 GB RAM minimum** (GROBID ~4-6 GB, LLM backend loads a ~400 MB taxonomy
  pickle, vision runs YOLO on CPU). 4 vCPU recommended. ~10 GB free disk.
- The server is on a private LAN address (192.168.x.x): it is NOT reachable from
  the internet as-is. Use option A (tunnel, recommended) or B (port forwarding).

## 1. Copy the project and the files that are not in git

```bash
# from your Mac
rsync -av --exclude node_modules --exclude .git --exclude '.venv' \
  --exclude 'vision/data' --exclude 'llm/finetuning' --exclude 'llm/tools' \
  ~/Desktop/Projects/MoultGPT/ USER@192.168.1.130:~/moultgpt/

# the data/model files that git ignores (check they exist locally first)
rsync -av ~/Desktop/Projects/MoultGPT/llm/data/arthropod_taxonomy.csv \
          ~/Desktop/Projects/MoultGPT/llm/data/taxonomy_lookup.pkl \
          ~/Desktop/Projects/MoultGPT/llm/data/moultdb_moulting_ontology_v3_8.owl \
          ~/Desktop/Projects/MoultGPT/llm/data/moultdb_trait_schema.json \
          USER@192.168.1.130:~/moultgpt/llm/data/
```

On the server create the secrets (never copy your real keys through chat or git):

```bash
cd ~/moultgpt
cp llm/.env.example llm/.env      # edit: MISTRAL_API_KEY=..., UNPAYWALL_EMAIL=...
cp .env.prod.example .env         # edit: DOMAIN / TUNNEL_TOKEN
rm -f llm/backend/feedback/feedback.jsonl   # contains a real query, don't ship it
```

Set a **spending limit** on the provider dashboard (Mistral/OpenRouter) as a
second line of defence besides the rate limits.

## 2A. Entry point: Cloudflare Tunnel (recommended)

Needs the domain's DNS on Cloudflare (free plan is enough).

1. Cloudflare dashboard -> Zero Trust -> Networks -> Tunnels -> *Create tunnel*
   (type: cloudflared). Copy the token into `.env` as `TUNNEL_TOKEN`.
2. In the tunnel's *Public hostname* tab: hostname `moultgpt.michele-leone.com`,
   service `http://gateway:8080`.
3. Start:
   ```bash
   docker compose -f docker-compose.prod.yml --profile tunnel up -d --build
   ```

No router configuration, no public IP, HTTPS handled by Cloudflare.

## 2B. Entry point: Caddy (automatic HTTPS)

Needs a public IP: forward ports 80 and 443 on your router to 192.168.1.130,
and create a DNS `A` record `moultgpt.michele-leone.com` -> your public IP
(use a dynamic-DNS updater if the IP changes).

```bash
docker compose -f docker-compose.prod.yml --profile caddy up -d --build
```

## 3. Verify

```bash
docker compose -f docker-compose.prod.yml ps          # all "running"/"healthy"
docker compose -f docker-compose.prod.yml logs -f llm-backend   # taxonomy load takes a while
curl -s https://moultgpt.michele-leone.com/healthz
```

Then in the browser: scan a DOI (exercises Unpaywall + GROBID + the gates) and
upload an image (exercises YOLO + classifier). GROBID needs ~1-2 minutes after
the first start before the DOI/PDF path works.

Check the limits really apply: 6 quick `POST /api/llm/query` calls from the same
IP should return `429` on the 6th.

## 4. Operating it

- Update: `git pull` (or rsync again), then re-run the `up -d --build` command.
- Logs: `docker compose -f docker-compose.prod.yml logs -f <service>`.
- Everything restarts automatically after a reboot (`restart: unless-stopped`).
- Rate limits and caps are env vars on the `gateway` service in the compose file.

## Known limitations of this setup

- Rate limiting is per IP and in memory (resets on restart): fine for a demo,
  not a substitute for authentication if you expect real traffic.
- Single worker per service: concurrent visitors queue. Raise `--threads`
  before adding workers (each LLM worker loads the full taxonomy again).
- The feedback endpoint stores free-text from visitors in a Docker volume;
  review it before using it for anything.

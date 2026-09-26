#!/usr/bin/env python3
"""MPT Cinematic web UI: a minimal local server for customer mode.

A single HTML form (no framework, no build step): highlights, tier,
cinematic look, hook variants, kinetic captions, reference image upload.
Submitting shells out to cli.py with a fixed --task-id and streams the
run log; finished videos (master + 15s/6s cutdowns + hook variants) play
right in the page.

Run:
    cd ~/workspace/money-printer-turbo
    .venv/bin/python cinematic_server.py            # http://127.0.0.1:8765
    .venv/bin/python cinematic_server.py 9000       # custom port

Stdlib only. Binds 127.0.0.1 (local machine) by design.
"""

import html
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(REPO_DIR, "storage", "tasks")
JOBS_DIR = os.path.join(REPO_DIR, "storage", "cinematic_jobs")

# The sandbox proxy breaks httpx (bracketed IPv6 no_proxy entries) and the
# Veo/Google calls must bypass the proxy: same env the CLI needs.
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")

TIERS = ["lite", "fast", "standard"]
LOOKS = ["golden-hour", "neon-noir", "vintage-film", "clean-modern"]

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def parse_multipart(body: bytes, boundary: bytes):
    """Minimal multipart/form-data parser for our known form.

    Returns (fields, files); files are (fieldname, filename, bytes).
    """
    fields: dict[str, str] = {}
    files: list[tuple[str, str, bytes]] = []
    for part in body.split(b"--" + boundary):
        if b"\r\n\r\n" not in part:
            continue
        head, data = part.split(b"\r\n\r\n", 1)
        if data.endswith(b"\r\n"):
            data = data[:-2]
        headers = head.decode("latin-1", "replace")
        name_m = re.search(r'name="([^"]+)"', headers)
        if not name_m:
            continue
        name = name_m.group(1)
        file_m = re.search(r'filename="([^"]*)"', headers)
        filename = file_m.group(1) if file_m else ""
        if filename:
            safe = os.path.basename(filename)
            if safe:
                files.append((name, safe, data))
        else:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


def build_cli_command(fields: dict, ref_paths: list[str], task_id: str) -> list[str]:
    tier = fields.get("veo_tier", "lite").strip() or "lite"
    look = fields.get("cinematic_look", "golden-hour").strip() or "golden-hour"
    try:
        variants = max(1, min(3, int(fields.get("hook_variants", "1") or 1)))
    except ValueError:
        variants = 1
    cmd = [
        sys.executable, "cli.py",
        "--task-id", task_id,
        "--video-subject", fields.get("video_subject", "").strip(),
        "--video-highlights", fields.get("video_highlights", "").strip(),
        "--video-source", "veo",
        "--veo-tier", tier if tier in TIERS else "lite",
        "--cinematic-look", look if look in LOOKS else "golden-hour",
        "--hook-variants", str(variants),
        "--confirm-veo-charge",
        "--video-aspect", "9:16",
        "--voice-name", fields.get("voice_name", "").strip() or "gemini:Charon-Informative",
    ]
    if fields.get("kinetic_captions"):
        cmd.append("--kinetic-captions")
    if ref_paths:
        cmd += ["--reference-images", ",".join(ref_paths)]
    return cmd


def run_job(job_id: str, cmd: list[str], task_id: str):
    with jobs_lock:
        job = jobs[job_id]
    job["log"].append("$ " + " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:  # noqa: BLE001
        job["log"].append(f"failed to start cli.py: {e}")
        job["done"] = True
        job["returncode"] = 127
        return
    job["proc"] = proc
    assert proc.stdout is not None
    for line in proc.stdout:
        job["log"].append(line.rstrip("\n"))
    proc.wait()
    job["returncode"] = proc.returncode
    job["done"] = True
    job["log"].append(f"exit code: {proc.returncode}")


def task_videos(task_id: str) -> list[dict]:
    videos = []
    task_dir = os.path.join(TASKS_DIR, task_id)
    if not os.path.isdir(task_dir):
        return videos
    for name in sorted(os.listdir(task_dir)):
        if name.startswith("final-") and name.endswith(".mp4"):
            videos.append(
                {"name": name, "url": f"/v/{task_id}/{name}"}
            )
    return videos


FORM_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MPT Cinematic</title>
<style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.6rem}label{display:block;margin:.9rem 0 .3rem;font-weight:600}
input[type=text],textarea,select{width:100%;padding:.55rem;border:1px solid #bbb;border-radius:6px;font-size:1rem}
textarea{min-height:110px}button{margin-top:1.2rem;padding:.7rem 1.6rem;font-size:1.05rem;border:0;border-radius:8px;background:#111;color:#fff;cursor:pointer}
button:disabled{opacity:.5}.row{display:flex;gap:1rem}.row>div{flex:1}
#log{background:#111;color:#cfc;font:12px/1.45 monospace;white-space:pre-wrap;padding:1rem;border-radius:8px;max-height:320px;overflow:auto;margin-top:1rem}
video{width:100%;max-width:340px;border-radius:8px;margin:.5rem .5rem 0 0;background:#000}
.note{color:#555;font-size:.9rem}
</style></head><body>
<h1>MPT Cinematic</h1>
<p class="note">Highlights in, finished 30s video out. Veo footage is <b>billed per second</b>; Lite tier is the suggested cheapest.</p>
<form id="f">
<label>Video subject<input type="text" name="video_subject" required placeholder="Sunrise Bakery grand opening"></label>
<label>Highlights (one per line)<textarea name="video_highlights" required placeholder="Fresh sourdough baked every morning at 5am&#10;Family owned since 1998&#10;Free coffee with any pastry this weekend"></textarea></label>
<div class="row"><div><label>Tier<select name="veo_tier"><option value="lite" selected>lite (cheapest, suggested)</option><option value="fast">fast</option><option value="standard">standard</option></select></label></div>
<div><label>Cinematic look<select name="cinematic_look"><option value="golden-hour" selected>golden-hour</option><option value="neon-noir">neon-noir</option><option value="vintage-film">vintage-film</option><option value="clean-modern">clean-modern</option></select></label></div></div>
<div class="row"><div><label>Hook variants<select name="hook_variants"><option value="1" selected>1 (one opening)</option><option value="2">2 alternate openings</option><option value="3">3 alternate openings</option></select></label></div>
<div><label>Voice<input type="text" name="voice_name" value="gemini:Charon-Informative"></label></div></div>
<label><input type="checkbox" name="kinetic_captions" value="1"> Kinetic captions (word-by-word pop-in)</label>
<label>Reference images (max 3: product / logo / character)<input type="file" name="reference_images" accept="image/*" multiple></label>
<label><input type="checkbox" id="confirm" required> I confirm Veo video generation is billed and I approve the charge.</label>
<button type="submit" id="go">Generate video</button>
</form>
<pre id="log" hidden></pre>
<div id="videos"></div>
<script>
const f=document.getElementById('f'),log=document.getElementById('log'),vids=document.getElementById('videos'),go=document.getElementById('go');
f.addEventListener('submit',async e=>{
  e.preventDefault();go.disabled=true;log.hidden=false;log.textContent='starting...\\n';vids.innerHTML='';
  const r=await fetch('/api/run',{method:'POST',body:new FormData(f)});
  const {job}=await r.json();
  const t=setInterval(async()=>{
    const p=await (await fetch('/api/progress?job='+job)).json();
    log.textContent=p.log.slice(-400).join('\\n');log.scrollTop=log.scrollHeight;
    if(p.done){clearInterval(t);go.disabled=false;
      vids.innerHTML='<h2>Finished videos</h2>'+p.videos.map(v=>`<figure style="display:inline-block;margin:0 .8rem .8rem 0"><video src="${v.url}" controls playsinline></video><figcaption>${v.name}</figcaption></figure>`).join('')||'<p>No videos produced.</p>';}
  },2000);
});
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "CinematicServer/1.0"

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, FORM_HTML.encode(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/progress":
            job_id = parse_qs(parsed.query).get("job", [""])[0]
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                self._send(404, b'{"error":"unknown job"}', "application/json")
                return
            with jobs_lock:
                payload = {
                    "done": job["done"],
                    "returncode": job["returncode"],
                    "log": job["log"][-400:],
                    "videos": task_videos(job["task_id"]) if job["done"] else [],
                }
            self._send(200, json.dumps(payload).encode(), "application/json")
            return
        if parsed.path.startswith("/v/"):
            _, _, task_id, name = parsed.path.split("/", 3)
            if ".." in task_id or ".." in name or "/" in name:
                self._send(400, b"bad path", "text/plain")
                return
            fpath = os.path.join(TASKS_DIR, task_id, name)
            if not (os.path.isfile(fpath) and name.endswith(".mp4")):
                self._send(404, b"not found", "text/plain")
                return
            ctype = mimetypes.guess_type(name)[0] or "video/mp4"
            with open(fpath, "rb") as fh:
                self._send(200, fh.read(), ctype)
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self._send(404, b"not found", "text/plain")
            return
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype or "boundary=" not in ctype:
            self._send(400, b'{"error":"multipart required"}', "application/json")
            return
        boundary = ctype.split("boundary=")[1].strip().encode()
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > 60 * 1024 * 1024:
            self._send(400, b'{"error":"bad body"}', "application/json")
            return
        body = self.rfile.read(length)
        fields, files = parse_multipart(body, boundary)
        if not fields.get("video_subject", "").strip() or not fields.get(
            "video_highlights", ""
        ).strip():
            self._send(400, b'{"error":"subject and highlights required"}',
                        "application/json")
            return

        job_id = uuid.uuid4().hex[:12]
        task_id = str(uuid.uuid4())
        job_dir = os.path.join(JOBS_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        ref_paths: list[str] = []
        for fieldname, filename, data in files:
            if fieldname != "reference_images":
                continue
            if len(ref_paths) >= 3:
                break
            dest = os.path.join(job_dir, filename)
            with open(dest, "wb") as fh:
                fh.write(data)
            ref_paths.append(dest)

        cmd = build_cli_command(fields, ref_paths, task_id)
        with jobs_lock:
            jobs[job_id] = {
                "cmd": cmd, "task_id": task_id, "log": [],
                "done": False, "returncode": None, "proc": None,
            }
        thread = threading.Thread(
            target=run_job, args=(job_id, cmd, task_id), daemon=True
        )
        thread.start()
        self._send(200, json.dumps({"job": job_id}).encode(), "application/json")

    def log_message(self, fmt, *args):  # keep stdout clean for the CLI child
        sys.stderr.write("server: " + fmt % args + "\n")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    os.makedirs(JOBS_DIR, exist_ok=True)
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"MPT Cinematic web UI: http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

"""Robust Hugging Face dataset fetcher for a flaky network: retries, redirects, ranged resume."""
import urllib.request, urllib.parse, http.client, ssl, json, time, os, sys

UA = {"User-Agent": "lockstep/0.1"}

def _open(url, headers=None, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers={**UA, **(headers or {})}), timeout=timeout)

def get_bytes(url, tries=30):
    last = None
    for i in range(tries):
        try:
            return _open(url).read()
        except Exception as e:
            last = e; time.sleep(min(1 + i, 6))
    raise last

def get_json(url, tries=30):
    return json.loads(get_bytes(url, tries))

def resolve_location(url, tries=40):
    """Ask huggingface.co where the file lives (one small request), retrying through resets."""
    last = None
    for i in range(tries):
        try:
            u = urllib.parse.urlparse(url)
            c = http.client.HTTPSConnection(u.netloc, timeout=30, context=ssl.create_default_context())
            c.request("HEAD", u.path, headers=UA)
            r = c.getresponse()
            if r.status in (301, 302, 303, 307, 308):
                loc = r.getheader("Location")
                return urllib.parse.urljoin(url, loc), int(r.getheader("X-Linked-Size") or r.getheader("Content-Length") or 0)
            if r.status == 200:
                return url, int(r.getheader("Content-Length") or 0)
            last = RuntimeError(f"HTTP {r.status}")
        except Exception as e:
            last = e
        time.sleep(min(1 + i, 6))
    raise last

def download(repo, path, dest, chunk=8 << 20, tries_per_chunk=25):
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    loc, size = resolve_location(url)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if size and have >= size:
        print(f"  ok (cached) {path} {size/1e6:.1f} MB"); return dest
    with open(dest, "ab") as f:
        while not size or have < size:
            end = (have + chunk - 1) if size else ""
            for i in range(tries_per_chunk):
                try:
                    r = _open(loc, {"Range": f"bytes={have}-{end}"}, timeout=90)
                    data = r.read()
                    if not size:
                        size = int((r.getheader("Content-Range") or "/0").split("/")[-1] or 0)
                    break
                except Exception as e:
                    if i == tries_per_chunk - 1: raise
                    time.sleep(min(1 + i, 6))
                    if i % 5 == 4:  # signed CDN URLs can expire; re-resolve
                        loc, _ = resolve_location(url)
            f.write(data); f.flush(); have += len(data)
            print(f"  {path}: {have/1e6:.1f}/{size/1e6:.1f} MB", flush=True)
            if not data: break
    return dest

if __name__ == "__main__":
    repo = sys.argv[1]
    for p in sys.argv[2:]:
        download(repo, p, os.path.join("data", repo.replace("/", "__"), p))

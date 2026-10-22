#!/usr/bin/env python3
"""Resume official code-skill release ranges, verify every part, then install."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from dsr.paths import repository_root
from dsr.runtime_config import code_asset_root
ROOT = repository_root()
ASSET_ROOT = code_asset_root(required=False)
VENDOR = ROOT / 'vendor/PhysicalRSI_XPolicyLab/policy/physicalRSI'

def atomic(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--proxy', default=None)
    args = p.parse_args()
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    cache = ASSET_ROOT / '.downloads'; cache.mkdir(exist_ok=True)
    chunks = ASSET_ROOT / '.ranges'; chunks.mkdir(exist_ok=True)
    asset = next(a for a in json.loads((VENDOR/'assets.json').read_text())['assets']
                 if a['name'] == "physicalrsi-code-skills")
    proxy = urllib.request.ProxyHandler({'https': args.proxy, 'http': args.proxy}) if args.proxy else urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy)
    redirects = urllib.request.build_opener(proxy, NoRedirect())
    api = "https://api.github.com/repos/yanming03/XPolicyLab/releases/tags/physicalrsi-skill-assets-v1"
    with opener.open(api, timeout=40) as r:
        release = json.load(r)
    urls = {a['name']: a['url'] for a in release['assets']}
    signed = {}; locks = {part['name']: threading.Lock() for part in asset['parts']}
    def url(part):
        with locks[part['name']]:
            record = signed.get(part['name'])
            if record and time.time() - record[0] < 1200:
                return record[1]
            request = urllib.request.Request(urls[part['name']] + '?t=' + str(time.time_ns()),
                                             headers={'Accept': 'application/octet-stream'})
            try:
                response = redirects.open(request, timeout=40)
                raise RuntimeError('GitHub did not redirect asset: ' + str(response.status))
            except urllib.error.HTTPError as exc:
                if exc.code != 302: raise
                location = exc.headers['Location']
            signed[part['name']] = (time.time(), location)
            return location
    size = 4 * 1024**2
    jobs = [(part, start, min(start+size, part['size'])-1)
            for part in asset['parts'] for start in range(0, part['size'], size)]
    total = sum(part['size'] for part in asset['parts'])
    started = time.time(); completed = 0
    def fetch(job):
        part, start, end = job
        target = chunks / (part['name'] + '.' + str(start))
        length = end-start+1
        if target.exists() and target.stat().st_size == length:
            return length
        for attempt in range(12):
            try:
                tmp = target.with_suffix(target.suffix + '.partial')
                offset=tmp.stat().st_size if tmp.exists() else 0
                if offset>length:tmp.unlink();offset=0
                if offset==length:
                    tmp.replace(target);return length
                req = urllib.request.Request(url(part) + '&dsr_range=' + str(start+offset),
                                             headers={'Range': f'bytes={start+offset}-{end}'})
                with opener.open(req, timeout=60) as r:
                    expected = f'bytes {start+offset}-{end}/{part["size"]}'
                    if r.status != 206 or r.headers.get('Content-Range') != expected:
                        raise ValueError('Invalid ranged response')
                    with tmp.open('ab') as out:
                        remaining = length-offset
                        while remaining:
                            block = r.read(min(256*1024, remaining))
                            if not block: raise IOError('Truncated response')
                            out.write(block); remaining -= len(block)
                    tmp.replace(target)
                return length
            except Exception:
                if attempt == 11: raise
                time.sleep(min(10, 1+attempt))
        raise AssertionError('unreachable')
    def status(state, **extra):
        atomic(ASSET_ROOT/'download_status.json', dict(status=state, completed_bytes=completed,
            total_bytes=total, elapsed_seconds=time.time()-started, workers=args.workers, **extra))
    status('DOWNLOADING')
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, job) for job in jobs]
            last = 0
            for future in as_completed(futures):
                completed += future.result()
                if time.time()-last > 10:
                    status('DOWNLOADING'); last=time.time()
        status('VERIFYING')
        for part in asset['parts']:
            target = cache/part['name']; tmp=target.with_suffix('.assembling')
            h=hashlib.sha256()
            with tmp.open('wb') as out:
                for start in range(0,part['size'],size):
                    with (chunks/(part['name']+'.'+str(start))).open('rb') as src:
                        for block in iter(lambda:src.read(1024**2),b''):
                            h.update(block); out.write(block)
            if h.hexdigest()!=part['sha256']:
                raise ValueError('SHA256 mismatch: '+part['name'])
            tmp.replace(target)
        status('INSTALLING')
        spec=importlib.util.spec_from_file_location('physicalrsi_asset_installer',VENDOR/'download_assets.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        destination=ASSET_ROOT/asset['destination']
        if not destination.exists():mod.install(asset,ASSET_ROOT)
        if not (destination/'programs.json').is_file():
            import sys
            sys.path.insert(0,str(ROOT/'vendor/PhysicalRSI'))
            from PhysicalRSI_baselines.robodojo.prepare_code_programs import prepare
            prepare(destination)
        status('VERIFIED_INSTALLED', destination=str(destination),
               part_sha256={p['name']:p['sha256'] for p in asset['parts']})
    except BaseException as exc:
        status('FAILED',error=str(exc));raise

if __name__=='__main__':
    main()

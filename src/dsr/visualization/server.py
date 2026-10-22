from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from dsr.evidence import contained, inventory
from dsr.paths import repository_root

PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DSR · Research evidence</title><style>
:root{color-scheme:dark;font:16px/1.5 system-ui,sans-serif;background:#111820;color:#e3e9ee}body{max-width:1300px;margin:0 auto;padding:36px}
header{border-bottom:1px solid #354652;margin-bottom:28px}small,.muted{color:#a7bac9}h1{font-size:34px;letter-spacing:-1px;margin:0}h2{font-size:20px}
.cards{display:flex;gap:14px;flex-wrap:wrap}.card{background:#1a2631;border:1px solid #344655;border-radius:8px;padding:16px;min-width:160px;flex:1}.value{font-size:32px;font-weight:600;color:#76d7c4}
nav{display:flex;gap:8px;flex-wrap:wrap;margin:28px 0}button,input{background:#1b2a38;color:#e3e9ee;border:1px solid #40586b;padding:9px 14px;border-radius:6px}button{cursor:pointer}button.active{background:#356859}
table{border-collapse:collapse;width:100%;text-align:left}td,th{padding:12px;border-bottom:1px solid #304353;vertical-align:top}th{color:#9fb9ca;font-size:13px;text-transform:uppercase}td{font-size:14px}a{color:#76d7c4}pre{white-space:pre-wrap;overflow-wrap:anywhere}video{max-width:650px;width:100%}.empty{padding:30px;background:#1a2631;border-radius:8px}#search{width:310px;margin-bottom:14px}
</style><header><small>DSR / LOCAL RESEARCH WORKSPACE</small><h1>Execution & evolution evidence</h1><p class="muted">Source coverage, measured execution, and unresolved requirements are tracked separately.</p></header>
<div class="cards" id="cards"></div><nav id="nav"></nav><input id="search" placeholder="Filter records" aria-label="Filter records"><main id="content"></main><p class="muted">This browser displays recorded files. Interactive Isaac Sim availability is reported separately.</p>
<script>
let data,tab='Tasks';const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const titles={official_task_definitions:'Official task definitions',native_configuration_definitions:'Native configurations',declared_code_programs:'Declared code programs',completed_native_episodes:'Completed native episodes',validated_distinct_skill_executions:'Validated skill executions'};
function draw(){document.querySelectorAll('nav button').forEach(b=>b.className=b.textContent===tab?'active':'');const filter=document.getElementById('search').value.toLowerCase();let rows=[];
if(tab==='Tasks')rows=data.tasks;if(tab==='Policies')rows=data.policies;if(tab==='Rollouts')rows=data.episodes;if(tab==='Skills & clips')rows=data.clips;
if(tab==='Capabilities'){let c=data.capabilities;rows=Array.isArray(c)?c:(c.capabilities||c.items||[])}
if(tab==='Evolution & memory'){document.getElementById('content').innerHTML='<pre>'+esc(JSON.stringify(data.research,null,2))+'</pre>';return;}
rows=rows.filter(r=>JSON.stringify(r).toLowerCase().includes(filter));let keys=tab==='Tasks'?['task_family','native_configuration','category','episodes_per_seed','execution_validation']:tab==='Policies'?['name','description','action_type','validation']:rows.length?Object.keys(rows[0]).filter(k=>!['configuration','raw'].includes(k)).slice(0,7):[];
document.getElementById('content').innerHTML=rows.length?'<table><thead><tr>'+keys.map(k=>'<th>'+esc(k.replaceAll('_',' '))+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+keys.map(k=>'<td>'+esc(typeof r[k]==='object'?JSON.stringify(r[k]):r[k])+'</td>').join('')+'</tr>'+(tab==='Skills & clips'?'<tr><td colspan="7"><video controls preload="metadata" src="/media/'+encodeURIComponent(r.path)+'"></video></td></tr>':'')).join('')+'</tbody></table>':'<div class="empty">No matching execution records. Unexecuted capabilities are not counted as validated results.</div>';
}
fetch('/api/index').then(r=>r.json()).then(d=>{data=d;document.getElementById('cards').innerHTML=Object.entries(titles).map(([k,v])=>'<div class="card"><div class="value">'+esc(d.counts[k])+'</div><small>'+esc(v)+'</small></div>').join('');document.getElementById('nav').innerHTML=['Tasks','Policies','Capabilities','Rollouts','Skills & clips','Evolution & memory'].map(t=>'<button>'+t+'</button>').join('');document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{tab=b.textContent;draw()});draw()}).catch(e=>document.getElementById('content').textContent='Cannot read evidence: '+e);
document.getElementById('search').oninput=()=>data&&draw();
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        try:
            if path == "/":
                body, content_type = PAGE.encode(), "text/html; charset=utf-8"
            elif path == "/api/index":
                body, content_type = json.dumps(inventory(self.server.root)).encode(), "application/json"
            elif path.startswith("/media/"):
                name = path[len("/media/"):]
                known = {x["path"] for x in inventory(self.server.root)["clips"]}
                if name not in known:
                    self.send_error(404); return
                media = contained(self.server.root, name)
                total = media.stat().st_size
                start, end = 0, total - 1
                partial = self.headers.get("Range")
                if partial:
                    import re
                    match = re.fullmatch(r"bytes=(\d+)-(\d*)", partial)
                    if not match:
                        self.send_error(416); return
                    start = int(match[1]); end = min(int(match[2]) if match[2] else end, end)
                    if start > end:
                        self.send_error(416); return
                self.send_response(206 if partial else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                if partial:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                self.end_headers()
                with media.open("rb") as stream:
                    stream.seek(start); remaining = end - start + 1
                    while remaining:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk: break
                        self.wfile.write(chunk); remaining -= len(chunk)
                return
            else:
                self.send_error(404); return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        except (OSError, ValueError, KeyError) as error:
            self.send_error(500, str(error))

    def log_message(self, format, *args):
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local evidence browser; forward its port in VS Code")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=repository_root())
    args = parser.parse_args(argv)
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.root = args.root.resolve()
        print(f"DSR evidence browser: http://127.0.0.1:{server.server_port}", flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass
    return 0

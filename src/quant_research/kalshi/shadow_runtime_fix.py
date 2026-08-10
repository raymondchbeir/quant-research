from __future__ import annotations
import json, urllib.request
from pathlib import Path
from . import shadow_trader as S

def _coinbase_loop(self):
    while not self.stop_event.is_set():
        try:
            req=urllib.request.Request(S.COINBASE_TICKER_URL,headers={"Cache-Control":"no-cache","User-Agent":"kalshi-shadow-research/1.0"})
            with urllib.request.urlopen(req,timeout=5) as r:x=json.loads(r.read().decode("utf-8"))
            self._btc_append(x.get("time"),x.get("price"))
        except Exception as exc:self.emit("ERROR",detail=f"Coinbase REST: {exc!r}")
        self.stop_event.wait(1)

def _start_shadow_trader(session_dir):
    session_dir=Path(session_dir)
    if S._SHADOW is not None and any(t.is_alive() for t in S._SHADOW.threads):
        if Path(S._SHADOW.session_dir).resolve()==session_dir.resolve():
            print("Shadow trader is already running for this session.");return S._SHADOW
        S._SHADOW.stop();S._SHADOW=None
    S._SHADOW=S.ShadowTrader(session_dir).start();return S._SHADOW

def _stop_shadow_trader():
    if S._SHADOW is None:print("Shadow trader is not running.");return None
    out=S._SHADOW.stop();S._SHADOW=None;return out

S.ShadowTrader._coinbase_loop=_coinbase_loop
S.start_shadow_trader=_start_shadow_trader
S.stop_shadow_trader=_stop_shadow_trader

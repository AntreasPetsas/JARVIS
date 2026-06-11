"""Spotify Web API — richer playback control (search & play, volume, now playing).

This is the optional upgrade over the media-key path in `spotify.py`. It uses
the OAuth Authorization Code flow: credentials come from `.env`
(`SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET`), and the refresh token is cached
in `jarvis/data/spotify_token.json` so you only authorize once.

Everything degrades gracefully — if the API isn't configured/authorized, or
there's no active Spotify Connect device, the router falls back to Windows media
keys. Calls speak plain HTTP via httpx, matching the rest of Jarvis (no SDK).
"""
from __future__ import annotations

import asyncio
import base64
import json
import random
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

from ..config import Config

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API = "https://api.spotify.com/v1"
SCOPES = ("user-read-playback-state user-modify-playback-state "
          "user-read-currently-playing user-library-read "
          "playlist-read-private playlist-read-collaborative")

DATA = Path(__file__).resolve().parent.parent / "data"
TOKEN_STORE = DATA / "spotify_token.json"
# A small rolling stack of the playlists/albums Jarvis has started, so "play the
# previous playlist" can return to the last context. Spotify has no API for this.
CONTEXT_STORE = DATA / "spotify_contexts.json"
CONTEXT_MAX = 20


class SpotifyWeb:
    """Thin async wrapper over the Spotify Web API playback endpoints."""

    def __init__(self, client_id: str, client_secret: str,
                 redirect_uri: str, default_device: str = ""):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.default_device = default_device

    # ---- configuration / auth state -------------------------------------
    def configured(self) -> bool:
        """True once the .env credentials are present."""
        return bool(self.client_id and self.client_secret)

    def authorized(self) -> bool:
        """True once the user has linked their account (we hold a refresh token)."""
        return bool(self._tokens().get("refresh_token"))

    def _tokens(self) -> dict:
        if not TOKEN_STORE.exists():
            return {}
        try:
            return json.loads(TOKEN_STORE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_tokens(self, data: dict) -> None:
        DATA.mkdir(parents=True, exist_ok=True)
        TOKEN_STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ---- context history (for "play the previous playlist") -------------
    def _contexts(self) -> list[dict]:
        if not CONTEXT_STORE.exists():
            return []
        try:
            data = json.loads(CONTEXT_STORE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save_contexts(self, items: list[dict]) -> None:
        DATA.mkdir(parents=True, exist_ok=True)
        CONTEXT_STORE.write_text(json.dumps(items[-CONTEXT_MAX:], indent=2), encoding="utf-8")

    def _record_context(self, uri: str, name: str, kind: str, extra: dict | None = None) -> None:
        """Push a started context onto the history stack (no consecutive dupes).

        `kind` is normally playlist/album/artist; "liked" is a sentinel for Liked
        Songs, which has no context URI. `extra` carries kind-specific state (e.g.
        the shuffle flag) so the context can be replayed faithfully."""
        if not uri:
            return
        items = self._contexts()
        if items and items[-1].get("uri") == uri:
            return
        rec = {"uri": uri, "name": name, "kind": kind}
        if extra:
            rec.update(extra)
        items.append(rec)
        self._save_contexts(items)

    def _basic_auth(self) -> str:
        raw = f"{self.client_id}:{self.client_secret}".encode()
        return "Basic " + base64.b64encode(raw).decode()

    # ---- OAuth ----------------------------------------------------------
    def auth_url(self, state: str) -> str:
        q = urlencode({
            "response_type": "code",
            "client_id": self.client_id,
            "scope": SCOPES,
            "redirect_uri": self.redirect_uri,
            "state": state,
        })
        return f"{AUTH_URL}?{q}"

    async def exchange_code(self, code: str) -> bool:
        """Trade the callback `code` for an access + refresh token, then cache it."""
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
        }
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(TOKEN_URL, data=data,
                             headers={"Authorization": self._basic_auth()})
        if r.status_code != 200:
            return False
        tok = r.json()
        self._save_tokens({
            "refresh_token": tok.get("refresh_token", ""),
            "access_token": tok.get("access_token", ""),
            "expires_at": time.time() + tok.get("expires_in", 3600) - 60,
        })
        return True

    async def _access_token(self) -> str | None:
        tokens = self._tokens()
        if not tokens.get("refresh_token"):
            return None
        if tokens.get("access_token") and tokens.get("expires_at", 0) > time.time():
            return tokens["access_token"]
        # Refresh using the long-lived refresh token.
        data = {"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"]}
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(TOKEN_URL, data=data,
                             headers={"Authorization": self._basic_auth()})
        if r.status_code != 200:
            return None
        tok = r.json()
        tokens["access_token"] = tok.get("access_token", "")
        tokens["expires_at"] = time.time() + tok.get("expires_in", 3600) - 60
        if tok.get("refresh_token"):  # Spotify occasionally rotates it
            tokens["refresh_token"] = tok["refresh_token"]
        self._save_tokens(tokens)
        return tokens["access_token"]

    # ---- low-level request ----------------------------------------------
    async def _api(self, method: str, path: str, *, params=None, json_body=None):
        """Return (status_code | None | 0, body_dict). None = not authorized, 0 = transport error.

        Honours Spotify's 429 rate-limit once: on a 429 we wait for the
        `Retry-After` header (capped) and retry a single time before giving up.
        """
        token = await self._access_token()
        if not token:
            return None, {}
        headers = {"Authorization": f"Bearer {token}"}
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=15) as c:
                    r = await c.request(method, f"{API}{path}", headers=headers,
                                        params=params, json=json_body)
            except httpx.HTTPError:
                return 0, {}
            if r.status_code == 429 and attempt == 0:
                try:
                    wait = int(r.headers.get("Retry-After", "1"))
                except ValueError:
                    wait = 1
                await asyncio.sleep(min(wait, 5))
                continue
            break
        body: dict = {}
        if r.content:
            try:
                body = r.json()
            except ValueError:
                body = {}
        return r.status_code, body

    def _result(self, status, ok_msg: str, body: dict) -> dict:
        """Map a Spotify status code onto a spoken result the router can use."""
        if status is None:
            return {"ok": False, "needs_auth": True,
                    "message": "Spotify isn't linked yet."}
        if status in (200, 202, 204):
            return {"ok": True, "message": ok_msg}
        if status == 404:
            return {"ok": False, "no_device": True,
                    "message": "No active Spotify device. Open Spotify and start "
                               "playing once, then try again."}
        if status == 403:
            err = body.get("error") or {}
            reason = err.get("reason", "")
            msg = (err.get("message") or "").lower()
            if reason == "PREMIUM_REQUIRED":
                return {"ok": False, "message": "That needs Spotify Premium, I'm afraid."}
            if "scope" in msg:
                return {"ok": False, "needs_auth": True,
                        "message": "I need updated Spotify permissions for that — "
                                   "say 'connect Spotify' to relink."}
            return {"ok": False, "message": "Spotify wouldn't allow that one."}
        if status == 401:
            return {"ok": False, "needs_auth": True,
                    "message": "Spotify authorization expired — say 'connect Spotify' to relink."}
        msg = (body.get("error") or {}).get("message", "")
        return {"ok": False, "message": f"Spotify error: {msg or status}."}

    # ---- devices --------------------------------------------------------
    async def _devices(self) -> list[dict]:
        status, body = await self._api("GET", "/me/player/devices")
        return body.get("devices", []) if status == 200 else []

    async def _target_device(self) -> str | None:
        """Pick a device to act on: the active one, else a preferred/first one."""
        devices = await self._devices()
        if not devices:
            return None
        active = next((d for d in devices if d.get("is_active")), None)
        if active:
            return active.get("id")
        if self.default_device:
            named = next((d for d in devices
                          if d.get("name", "").lower() == self.default_device.lower()), None)
            if named:
                return named.get("id")
        return devices[0].get("id")

    async def list_devices(self) -> dict:
        """Spoken summary of the available Spotify Connect devices."""
        devices = await self._devices()
        if not devices:
            return {"ok": False, "no_device": True,
                    "message": "No Spotify devices are available. Open Spotify somewhere first."}
        names = [d.get("name", "?") for d in devices]
        active = next((d.get("name") for d in devices if d.get("is_active")), None)
        msg = "Available devices: " + ", ".join(names) + "."
        if active:
            msg += f" {active} is active."
        return {"ok": True, "devices": devices, "message": msg}

    async def transfer_to(self, name_query: str, play: bool = True) -> dict:
        """Move playback to the device whose name best matches `name_query`."""
        name_query = (name_query or "").strip().lower()
        devices = await self._devices()
        if not devices:
            return {"ok": False, "no_device": True,
                    "message": "No Spotify devices are available. Open Spotify somewhere first."}
        match = next((d for d in devices if name_query in (d.get("name", "").lower())), None)
        if not match:
            names = ", ".join(d.get("name", "?") for d in devices)
            return {"ok": False, "message": f"I couldn't find a device called "
                                            f"'{name_query}'. You have: {names}."}
        status, body = await self._api("PUT", "/me/player",
                                       json_body={"device_ids": [match["id"]], "play": play})
        return self._result(status, f"Playing on {match.get('name', 'that device')}.", body)

    # ---- transport ------------------------------------------------------
    async def play(self) -> dict:
        dev = await self._target_device()
        params = {"device_id": dev} if dev else None
        status, body = await self._api("PUT", "/me/player/play", params=params)
        return self._result(status, "Resuming playback.", body)

    async def pause(self) -> dict:
        status, body = await self._api("PUT", "/me/player/pause")
        return self._result(status, "Paused.", body)

    async def toggle(self) -> dict:
        np = await self.now_playing()
        if np.get("ok") and np.get("is_playing"):
            return await self.pause()
        return await self.play()

    async def next(self) -> dict:
        status, body = await self._api("POST", "/me/player/next")
        return self._result(status, "Skipping ahead.", body)

    async def previous(self) -> dict:
        status, body = await self._api("POST", "/me/player/previous")
        return self._result(status, "Going back.", body)

    # ---- volume ---------------------------------------------------------
    async def set_volume(self, pct: int) -> dict:
        pct = max(0, min(100, int(pct)))
        status, body = await self._api("PUT", "/me/player/volume",
                                       params={"volume_percent": pct})
        label = "Muted." if pct == 0 else f"Volume {pct} percent."
        return self._result(status, label, body)

    async def nudge_volume(self, delta: int) -> dict:
        status, body = await self._api("GET", "/me/player")
        if status is None:
            return {"ok": False, "needs_auth": True, "message": "Spotify isn't linked yet."}
        cur = (body.get("device") or {}).get("volume_percent") if status == 200 else None
        if cur is None:
            return {"ok": False, "message": "I can't read the current volume — try an "
                                            "exact level, like 'volume 50'."}
        return await self.set_volume(cur + delta)

    # ---- shuffle / repeat -----------------------------------------------
    async def set_shuffle(self, on: bool) -> dict:
        dev = await self._target_device()
        params = {"state": "true" if on else "false"}
        if dev:
            params["device_id"] = dev
        status, body = await self._api("PUT", "/me/player/shuffle", params=params)
        return self._result(status, "Shuffle on." if on else "Shuffle off.", body)

    async def set_repeat(self, mode: str) -> dict:
        mode = mode if mode in ("off", "track", "context") else "context"
        dev = await self._target_device()
        params = {"state": mode}
        if dev:
            params["device_id"] = dev
        status, body = await self._api("PUT", "/me/player/repeat", params=params)
        label = {"off": "Repeat off.", "track": "Repeating this track.",
                 "context": "Repeating the playlist."}[mode]
        return self._result(status, label, body)

    # ---- seek -----------------------------------------------------------
    async def seek(self, position_ms: int) -> dict:
        position_ms = max(0, int(position_ms))
        dev = await self._target_device()
        params = {"position_ms": position_ms}
        if dev:
            params["device_id"] = dev
        status, body = await self._api("PUT", "/me/player/seek", params=params)
        label = "Back to the start." if position_ms == 0 else "Jumped."
        return self._result(status, label, body)

    async def seek_relative(self, delta_ms: int) -> dict:
        status, body = await self._api("GET", "/me/player")
        if status is None:
            return {"ok": False, "needs_auth": True, "message": "Spotify isn't linked yet."}
        if status != 200:
            return {"ok": False, "message": "I can't read the current position right now."}
        progress = body.get("progress_ms") or 0
        duration = (body.get("item") or {}).get("duration_ms")
        target = progress + delta_ms
        if duration is not None:
            target = min(target, duration)
        return await self.seek(max(0, target))

    # ---- queue ----------------------------------------------------------
    async def add_to_queue(self, uri: str, label: str = "Added to your queue.") -> dict:
        # Spotify's queue endpoint takes a single track/episode URI (no context)
        # and only appends — there's no true "play next"/insert-at-front API.
        if not uri:
            return {"ok": False, "message": "There's nothing to queue."}
        dev = await self._target_device()
        params = {"uri": uri}
        if dev:
            params["device_id"] = dev
        status, body = await self._api("POST", "/me/player/queue", params=params)
        return self._result(status, label, body)

    async def queue_search(self, query: str, kind: str = "track") -> dict:
        status, body = await self._api("GET", "/search",
                                       params={"q": query, "type": "track", "limit": 5})
        if status != 200:
            return self._result(status, "", body)
        # Spotify occasionally returns null entries in the items array — drop them.
        items = [it for it in ((body.get("tracks") or {}).get("items") or []) if it]
        if not items:
            return {"ok": False, "message": f"I couldn't find '{query}' on Spotify."}
        item = self._pick_best(items, query, "track")
        artist = ", ".join(a["name"] for a in item.get("artists", []))
        label = f"Queued {item.get('name', 'that')}" + (f" by {artist}." if artist else ".")
        return await self.add_to_queue(item["uri"], label)

    # ---- search & play --------------------------------------------------
    @staticmethod
    def _pick_best(items: list[dict], query: str, kind: str) -> dict:
        """Choose the closest match instead of blindly taking the first result:
        exact name, then prefix, then (for tracks) most popular, else first."""
        q = query.strip().lower()
        # Spotify can return null entries (e.g. unavailable playlists) — ignore them.
        items = [it for it in items if it]
        if not items:
            return {}
        exact = [it for it in items if (it.get("name") or "").lower() == q]
        if exact:
            if kind == "track":
                return max(exact, key=lambda it: it.get("popularity", 0))
            return exact[0]
        prefix = [it for it in items if (it.get("name") or "").lower().startswith(q)]
        if prefix:
            return prefix[0]
        return items[0]

    # ---- the user's own playlists ---------------------------------------
    async def _my_playlists(self) -> list[dict]:
        """All playlists the user owns or follows (paged). Needs the
        playlist-read-private scope — relink Spotify if it returns nothing."""
        out: list[dict] = []
        for offset in range(0, 200, 50):  # up to 200 playlists, 4 pages
            status, body = await self._api("GET", "/me/playlists",
                                           params={"limit": 50, "offset": offset})
            if status != 200:
                break
            batch = [p for p in (body.get("items") or []) if p]
            out.extend(batch)
            if len(batch) < 50:
                break
        return out

    async def _find_my_playlist(self, query: str) -> dict | None:
        """Best name match among the user's own playlists: exact, then prefix, then substring."""
        q = query.strip().lower()
        pls = await self._my_playlists()
        for test in (lambda n: n == q, lambda n: n.startswith(q), lambda n: q in n):
            hit = next((p for p in pls if test((p.get("name") or "").lower())), None)
            if hit and hit.get("uri"):
                return hit
        return None

    async def search_and_play(self, query: str, kind: str = "track",
                              shuffle: bool = False) -> dict:
        kind = kind if kind in ("track", "album", "artist", "playlist") else "track"
        # A named playlist is almost always one of the user's own. The public /search
        # endpoint can't see private playlists and outranks personal ones, so look in
        # the user's library first and only fall back to the catalog if there's no match.
        if kind == "playlist":
            mine = await self._find_my_playlist(query)
            if mine:
                if shuffle:
                    await self.set_shuffle(True)
                dev = await self._target_device()
                params = {"device_id": dev} if dev else None
                st, b = await self._api("PUT", "/me/player/play", params=params,
                                        json_body={"context_uri": mine["uri"]})
                res = self._result(st, "", b)
                if res["ok"]:
                    res["message"] = self._now_phrase(mine, "playlist")
                    self._record_context(mine["uri"], mine.get("name", ""), "playlist")
                return res
        status, body = await self._api("GET", "/search",
                                       params={"q": query, "type": kind, "limit": 5})
        if status != 200:
            return self._result(status, "", body)
        # Spotify occasionally returns null entries in the items array — drop them.
        items = [it for it in ((body.get(kind + "s") or {}).get("items") or []) if it]
        if not items:
            return {"ok": False, "message": f"I couldn't find a {kind} for '{query}' on Spotify."}
        item = self._pick_best(items, query, kind)
        if not item.get("uri"):
            return {"ok": False, "message": f"I couldn't find a {kind} for '{query}' on Spotify."}
        # For a context (album/artist/playlist), set shuffle first so it starts shuffled.
        if shuffle and kind != "track":
            await self.set_shuffle(True)
        dev = await self._target_device()
        params = {"device_id": dev} if dev else None
        payload = {"uris": [item["uri"]]} if kind == "track" else {"context_uri": item["uri"]}
        st, b = await self._api("PUT", "/me/player/play", params=params, json_body=payload)
        res = self._result(st, "", b)
        if res["ok"]:
            res["message"] = self._now_phrase(item, kind)
            if kind in ("album", "artist", "playlist"):
                self._record_context(item["uri"], item.get("name", ""), kind)
        return res

    async def play_previous_context(self) -> dict:
        """Replay the playlist/album started just before the current one. Toggles
        between the last two contexts on repeated calls."""
        items = self._contexts()
        current = items[-1] if items else None
        cur_uri = current.get("uri") if current else None
        prev = next((c for c in reversed(items[:-1]) if c.get("uri") != cur_uri), None)
        if not prev or not prev.get("uri"):
            return {"ok": False, "message": "I don't have an earlier playlist to go back to, sir."}
        # Liked Songs has no context URI — replay it via play_liked (which re-records it).
        if prev.get("kind") == "liked":
            return await self.play_liked(shuffle=bool(prev.get("shuffle")))
        dev = await self._target_device()
        params = {"device_id": dev} if dev else None
        st, b = await self._api("PUT", "/me/player/play", params=params,
                                json_body={"context_uri": prev["uri"]})
        res = self._result(st, "", b)
        if res["ok"]:
            res["message"] = self._now_phrase(prev, prev.get("kind", "playlist"))
            self._record_context(prev["uri"], prev.get("name", ""), prev.get("kind", "playlist"))
        return res

    # ---- liked / saved songs --------------------------------------------
    async def play_liked(self, shuffle: bool = False) -> dict:
        """Play the user's Liked Songs. These are saved tracks, not a real
        playlist, so there's no searchable URI — we fetch them and play the URIs."""
        status, body = await self._api("GET", "/me/tracks", params={"limit": 50})
        if status is None:
            return {"ok": False, "needs_auth": True, "message": "Spotify isn't linked yet."}
        if status != 200:
            return self._result(status, "", body)
        uris = [it["track"]["uri"] for it in body.get("items", [])
                if it.get("track") and it["track"].get("uri")]
        if not uris:
            return {"ok": False, "message": "You don't have any Liked Songs yet."}
        if shuffle:
            random.shuffle(uris)
        dev = await self._target_device()
        params = {"device_id": dev} if dev else None
        st, b = await self._api("PUT", "/me/player/play", params=params,
                                json_body={"uris": uris})
        msg = "Shuffling your Liked Songs." if shuffle else "Playing your Liked Songs."
        res = self._result(st, msg, b)
        if res["ok"]:
            self._record_context("liked", "Liked Songs", "liked", {"shuffle": shuffle})
        return res

    # ---- now playing ----------------------------------------------------
    async def now_playing(self) -> dict:
        status, body = await self._api("GET", "/me/player/currently-playing")
        if status is None:
            return {"ok": False, "needs_auth": True, "message": "Spotify isn't linked yet."}
        if status == 204 or not body:
            return {"ok": True, "is_playing": False,
                    "message": "Nothing's playing on Spotify right now."}
        if status != 200:
            return self._result(status, "", body)
        item = body.get("item") or {}
        card = self._track_card(item, body.get("is_playing", False), body.get("progress_ms"))
        card["ok"] = True
        card["message"] = (f"{card['title']} by {card['artist']}." if item
                           else "Nothing's playing right now.")
        return card

    # ---- shaping --------------------------------------------------------
    @staticmethod
    def _track_card(item: dict, is_playing: bool, progress_ms: int | None = None) -> dict:
        album = item.get("album", {}) or {}
        images = album.get("images", [])
        return {
            "title": item.get("name", "—"),
            "artist": ", ".join(a["name"] for a in item.get("artists", [])) or "—",
            "album": album.get("name", ""),
            "image": images[0]["url"] if images else "",
            "is_playing": is_playing,
            "duration_ms": item.get("duration_ms"),
            "progress_ms": progress_ms,
            "url": (item.get("external_urls") or {}).get("spotify", ""),
            "uri": item.get("uri", ""),
        }

    @staticmethod
    def _now_phrase(item: dict, kind: str) -> str:
        name = item.get("name", "that")
        artist = ", ".join(a["name"] for a in item.get("artists", []))
        if kind == "artist":
            return f"Playing {name}."
        if kind == "album":
            return f"Playing the album {name}" + (f" by {artist}." if artist else ".")
        if kind == "playlist":
            return f"Playing the playlist {name}."
        return f"Playing {name}" + (f" by {artist}." if artist else ".")


def get_spotify(cfg: Config) -> SpotifyWeb:
    """Build a SpotifyWeb client from config + .env (mirrors get_llm)."""
    return SpotifyWeb(
        client_id=Config.env("SPOTIFY_CLIENT_ID", "") or "",
        client_secret=Config.env("SPOTIFY_CLIENT_SECRET", "") or "",
        redirect_uri=cfg.get("spotify.redirect_uri",
                             "http://127.0.0.1:8765/spotify/callback"),
        default_device=cfg.get("spotify.default_device", "") or "",
    )

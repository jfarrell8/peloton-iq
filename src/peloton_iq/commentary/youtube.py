"""
peloton_iq.commentary.youtube
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
YouTubeCacheManager — builds and refreshes the local video cache.

The v3 design insight: download all channel uploads once (~few hundred
quota units), then do all race matching locally at zero quota cost.
This replaces the v2 pattern of searching YouTube per-race which
exhausted the daily quota after ~25 races.

Usage:
    from peloton_iq.commentary.youtube import YouTubeCacheManager

    mgr = YouTubeCacheManager()
    mgr.build_cache()           # one-time full download
    mgr.refresh_recent(days=30) # lightweight weekly refresh
    df  = mgr.load_cache()      # load for local matching
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Optional

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from peloton_iq.config import settings, YOUTUBE_CACHE_PATH

log = logging.getLogger(__name__)

# Channel priority tiers — higher = better match candidate
CHANNEL_TIER = {
    "NBC Sports":         3,
    "GCN Racing":         2,
    "TNT Sports Cycling": 1,
    "GCN":                1,
}

# Races where NBC Sports is strongly preferred
NBC_PRIORITY_RACES = [
    "tour de france",
    "vuelta a espana",
    "vuelta españa",
]

# Races where GCN Racing / TNT are more likely to have the best content
GCN_PRIORITY_RACES = [
    "giro d'italia",
    "giro d italia",
    "paris-roubaix",
    "paris roubaix",
    "tour of flanders",
    "ronde van vlaanderen",
    "liege-bastogne-liege",
    "milan-san remo",
    "milan san remo",
    "il lombardia",
]


class YouTubeCacheManager:
    """
    Manages the local YouTube video metadata cache.

    Builds a parquet file of all video metadata from the configured
    channels. Subsequent race matching happens locally against this
    cache — zero quota cost.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_path=None,
        channels: list[dict] | None = None,
        max_pages: int | None = None,
    ) -> None:
        self._api_key    = api_key    or settings.youtube_api_key
        self._cache_path = cache_path or YOUTUBE_CACHE_PATH
        self._channels   = channels   or settings.youtube_channels
        self._max_pages  = max_pages  or settings.youtube_cache_max_pages
        self._youtube    = None

    # ------------------------------------------------------------------
    # YouTube client (lazy)
    # ------------------------------------------------------------------

    @property
    def youtube(self):
        if self._youtube is None:
            if not self._api_key:
                raise ValueError(
                    "YouTube API key not set. "
                    "Set PELOTON_YOUTUBE_API_KEY in your .env file."
                )
            self._youtube = build("youtube", "v3", developerKey=self._api_key)
        return self._youtube

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def load_cache(self) -> Optional[pd.DataFrame]:
        """Load the local cache from disk. Returns None if not built yet."""
        if not self._cache_path.exists():
            log.warning("Cache not found at %s — run build_cache() first", self._cache_path)
            return None
        df      = pd.read_parquet(self._cache_path)
        age_hrs = (
            datetime.datetime.utcnow() -
            datetime.datetime.fromtimestamp(self._cache_path.stat().st_mtime)
        ).seconds / 3600
        log.info(
            "Cache loaded: %d videos (last updated %.1fh ago)",
            len(df), age_hrs,
        )
        return df

    def build_cache(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Download all channel uploads and save to parquet.
        Safe to re-run — loads from disk unless force_refresh=True.

        Quota cost: ~1 unit per 50 videos fetched — effectively free
        for the full channel history.
        """
        if self._cache_path.exists() and not force_refresh:
            log.info(
                "Cache already exists. Pass force_refresh=True to re-download. "
                "Use refresh_recent() for incremental updates."
            )
            return self.load_cache()

        log.info("Building full video cache — downloading all channel uploads...")
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

        all_dfs = []
        for ch in self._channels:
            log.info("  Fetching: %s", ch["name"])
            ch_df = self._fetch_channel_videos(ch)
            log.info("    → %d videos", len(ch_df))
            all_dfs.append(ch_df)
            time.sleep(1.0)

        df = pd.concat(all_dfs, ignore_index=True)
        df.to_parquet(self._cache_path)
        log.info("Cache saved: %d videos → %s", len(df), self._cache_path)
        return df

    def refresh_recent(self, days_back: int | None = None) -> pd.DataFrame:
        """
        Lightweight refresh: re-fetch the first few pages of each channel
        to pick up new uploads without re-downloading the full history.

        Quota cost: ~8 units total (one per channel × 2 pages).
        """
        days_back = days_back or settings.youtube_refresh_days

        if not self._cache_path.exists():
            log.warning("No cache found — run build_cache() first")
            return pd.DataFrame()

        existing = pd.read_parquet(self._cache_path)
        cutoff   = pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)
        new_rows = []

        for ch in self._channels:
            playlist_id = self._get_upload_playlist(ch["id"])
            if not playlist_id:
                continue
            page_token = None
            for _ in range(4):  # ≤4 pages = 200 newest videos per channel
                resp = self._fetch_playlist_page(playlist_id, page_token)
                stop = False
                for item in resp.get("items", []):
                    s      = item["snippet"]
                    vid_id = s.get("resourceId", {}).get("videoId")
                    if not vid_id:
                        continue
                    pub = pd.to_datetime(s["publishedAt"], utc=True)
                    if pub < cutoff:
                        stop = True
                        break
                    if vid_id not in existing["video_id"].values:
                        new_rows.append({
                            "video_id":   vid_id,
                            "title":      s["title"],
                            "published":  pub,
                            "channel":    ch["name"],
                            "channel_id": ch["id"],
                        })
                page_token = resp.get("nextPageToken")
                if not page_token or stop:
                    break
            time.sleep(0.5)

        if new_rows:
            fresh  = pd.DataFrame(new_rows)
            merged = pd.concat([existing, fresh], ignore_index=True).drop_duplicates("video_id")
            merged.to_parquet(self._cache_path)
            log.info("Refresh: added %d new videos (%d total)", len(new_rows), len(merged))
            return merged
        else:
            log.info("Refresh: no new videos found")
            return existing

    # ------------------------------------------------------------------
    # Internal YouTube API helpers
    # ------------------------------------------------------------------

    def _get_upload_playlist(self, channel_id: str) -> Optional[str]:
        """Get the uploads playlist ID for a channel. Costs 1 quota unit."""
        try:
            resp  = self.youtube.channels().list(part="contentDetails", id=channel_id).execute()
            items = resp.get("items", [])
            if not items:
                return None
            return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        except HttpError as e:
            log.error("Failed to get upload playlist for %s: %s", channel_id, e)
            return None

    def _fetch_playlist_page(self, playlist_id: str, page_token=None) -> dict:
        """Fetch one page of playlist items. Costs 1 quota unit."""
        kwargs = dict(part="snippet", playlistId=playlist_id, maxResults=50)
        if page_token:
            kwargs["pageToken"] = page_token
        return self.youtube.playlistItems().list(**kwargs).execute()

    def _fetch_channel_videos(self, channel: dict) -> pd.DataFrame:
        """Download all video metadata from a channel's uploads playlist."""
        playlist_id = self._get_upload_playlist(channel["id"])
        if not playlist_id:
            log.warning("No uploads playlist for %s", channel["name"])
            return pd.DataFrame()

        rows, page_token, pages = [], None, 0
        while pages < self._max_pages:
            resp       = self._fetch_playlist_page(playlist_id, page_token)
            pages     += 1
            for item in resp.get("items", []):
                s      = item["snippet"]
                vid_id = s.get("resourceId", {}).get("videoId")
                if not vid_id or s.get("title") == "Private video":
                    continue
                rows.append({
                    "video_id":   vid_id,
                    "title":      s["title"],
                    "published":  s["publishedAt"],
                    "channel":    channel["name"],
                    "channel_id": channel["id"],
                })
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
            time.sleep(0.1)

        df = pd.DataFrame(rows)
        if not df.empty:
            df["published"] = pd.to_datetime(df["published"], utc=True)
        return df
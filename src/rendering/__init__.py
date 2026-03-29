import os
import json
import hashlib
import shutil
import logging

logger = logging.getLogger(__name__)


class RenderCache:
    """Skip re-rendering identical code+quality combos."""

    def __init__(self, cache_dir=None, max_entries=200):
        default = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", ".render_cache")
        self.cache_dir = cache_dir or os.path.abspath(default)
        self.max_entries = max_entries
        self.index_path = os.path.join(self.cache_dir, "index.json")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._index = self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_index(self):
        try:
            with open(self.index_path, "w") as f:
                json.dump(self._index, f, indent=2)
        except OSError as e:
            logger.warning(f"Cache index write failed: {e}")

    def _make_key(self, code, quality):
        raw = f"{code.strip()}|{quality}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, code, quality):
        """Return cached video path if it exists, else None."""
        key = self._make_key(code, quality)
        entry = self._index.get(key)
        if not entry:
            return None

        path = entry.get("path", "")
        if os.path.exists(path):
            logger.info(f"Render cache hit: {key}")
            return path

        # stale entry
        del self._index[key]
        self._save_index()
        return None

    def put(self, code, quality, video_path):
        """Cache a successful render result."""
        if not os.path.exists(video_path):
            return

        key = self._make_key(code, quality)

        # copy video into cache dir so it survives temp cleanup
        ext = os.path.splitext(video_path)[1] or ".mp4"
        cached_path = os.path.join(self.cache_dir, f"{key}{ext}")
        try:
            shutil.copy2(video_path, cached_path)
        except OSError as e:
            logger.warning(f"Cache copy failed: {e}")
            return

        self._index[key] = {"path": cached_path, "quality": quality}

        # evict oldest if over limit
        if len(self._index) > self.max_entries:
            oldest_key = next(iter(self._index))
            old_path = self._index[oldest_key].get("path", "")
            if os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except OSError:
                    pass
            del self._index[oldest_key]

        self._save_index()
        logger.info(f"Render cached: {key}")

    def clear(self):
        """Wipe the cache."""
        for entry in self._index.values():
            p = entry.get("path", "")
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        self._index = {}
        self._save_index()
        logger.info("Render cache cleared")

    def stats(self):
        valid = sum(1 for e in self._index.values() if os.path.exists(e.get("path", "")))
        return {"total": len(self._index), "valid": valid, "dir": self.cache_dir}

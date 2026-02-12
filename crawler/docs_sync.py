"""
Sync Manim documentation from GitHub for RAG indexing.
Downloads RST files, converts to markdown, tracks changes via commit hash.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import requests

from .rst_converter import RSTtoMarkdownConverter

logger = logging.getLogger(__name__)


class ManimDocsSync:
    """Fetches and converts Manim docs from GitHub."""
    
    REPO = "ManimCommunity/manim"
    BRANCH = "main"
    DOCS_PATH = "docs/source"
    
    API_BASE = "https://api.github.com"
    RAW_BASE = "https://raw.githubusercontent.com"
    
    def __init__(self, output_dir=None, cache_dir=None):
        base = Path(__file__).parent
        self.output_dir = Path(output_dir) if output_dir else base / "docs_output"
        self.cache_dir = Path(cache_dir) if cache_dir else base / ".sync_cache"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.cache_dir / "sync_metadata.json"
        self.converter = RSTtoMarkdownConverter()
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        if self.metadata_file.exists():
            try:
                return json.load(open(self.metadata_file))
            except:
                pass
        return {
            "last_sync": None, "last_commit": None, "last_commit_date": None,
            "files_synced": 0, "total_size_kb": 0, "sync_history": []
        }
    
    def _save_metadata(self):
        try:
            json.dump(self.metadata, open(self.metadata_file, 'w'), indent=2)
        except Exception as e:
            logger.error(f"Couldn't save metadata: {e}")
    
    def _api_request(self, endpoint, timeout=10):
        """Make a GitHub API request."""
        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "ManimDocsSync"}
        try:
            r = requests.get(f"{self.API_BASE}{endpoint}", headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.error(f"API error: {e}")
            return None
    
    def get_latest_commit(self):
        """Get info about the latest commit on main branch."""
        data = self._api_request(f"/repos/{self.REPO}/commits/{self.BRANCH}")
        if not data:
            return None
        return {
            "sha": data["sha"],
            "date": data["commit"]["committer"]["date"],
            "message": data["commit"]["message"][:100]
        }
    
    def check_for_updates(self):
        """Check if there are new commits since last sync."""
        latest = self.get_latest_commit()
        if not latest:
            return False
        
        last = self.metadata.get("last_commit")
        if not last:
            logger.info("First sync - update needed")
            return True
        
        if latest["sha"] != last:
            logger.info(f"New commit: {last[:8]} -> {latest['sha'][:8]}")
            return True
        
        logger.info("Docs up to date")
        return False
    
    def get_docs_tree(self):
        """Get list of RST files from the docs folder."""
        data = self._api_request(f"/repos/{self.REPO}/git/trees/{self.BRANCH}?recursive=1", timeout=30)
        if not data:
            return None
        
        files = []
        for item in data.get("tree", []):
            path = item.get("path", "")
            if path.startswith(self.DOCS_PATH) and path.endswith(".rst"):
                files.append({"path": path, "sha": item.get("sha"), "size": item.get("size", 0)})
        
        logger.info(f"Found {len(files)} RST files")
        return files
    
    def download_file(self, filepath):
        """Download a single file from the repo."""
        try:
            url = f"{self.RAW_BASE}/{self.REPO}/{self.BRANCH}/{filepath}"
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            return None
    
    def sync_docs(self, force=False):
        """
        Main sync method. Downloads RST files, converts to MD, saves to output dir.
        Returns (success, message) tuple.
        """
        if not force and not self.check_for_updates():
            return True, "Already up to date"
        
        commit = self.get_latest_commit()
        if not commit:
            return False, "Couldn't get commit info"
        
        files = self.get_docs_tree()
        if not files:
            return False, "Couldn't get file list"
        
        # Process files
        markdown_parts = []
        count = 0
        total_size = 0
        
        for i, f in enumerate(files):
            name = Path(f["path"]).stem
            if (i + 1) % 10 == 0:
                logger.info(f"Processing {i+1}/{len(files)}...")
            
            content = self.download_file(f["path"])
            if not content:
                continue
            
            md = self.converter.convert(content)
            if md:
                title = name.replace('_', ' ').title()
                markdown_parts.append(f"\n\n# {title}\n\n{md}")
                count += 1
                total_size += len(md)
        
        if count == 0:
            return False, "No files processed"
        
        # Save combined output
        combined = "\n\n---\n\n".join(markdown_parts)
        
        out_file = self.output_dir / "manim_docs_combined.md"
        out_file.write_text(combined, encoding='utf-8')
        
        # Also update legacy file for backward compatibility
        legacy = Path(__file__).parent / "markdown_output.md"
        try:
            legacy.write_text(combined, encoding='utf-8')
        except:
            pass
        
        # Update metadata
        self.metadata.update({
            "last_sync": datetime.now().isoformat(),
            "last_commit": commit["sha"],
            "last_commit_date": commit["date"],
            "files_synced": count,
            "total_size_kb": round(total_size / 1024, 2)
        })
        self.metadata["sync_history"].append({
            "date": datetime.now().isoformat(),
            "commit": commit["sha"][:8],
            "files": count
        })
        self.metadata["sync_history"] = self.metadata["sync_history"][-10:]
        self._save_metadata()
        
        msg = f"Synced {count}/{len(files)} files ({self.metadata['total_size_kb']}KB)"
        logger.info(msg)
        return True, msg
    
    def get_status(self):
        """Get current sync status."""
        latest = self.get_latest_commit()
        return {
            "last_sync": self.metadata.get("last_sync"),
            "last_commit": self.metadata.get("last_commit"),
            "files_synced": self.metadata.get("files_synced", 0),
            "total_size_kb": self.metadata.get("total_size_kb", 0),
            "latest_commit": latest["sha"] if latest else None,
            "update_available": latest and latest["sha"] != self.metadata.get("last_commit"),
            "output_dir": str(self.output_dir)
        }
    
    def get_output_path(self):
        return str(self.output_dir / "manim_docs_combined.md")


# Module-level helpers
_instance = None

def get_docs_sync():
    global _instance
    if not _instance:
        _instance = ManimDocsSync()
    return _instance

def sync_manim_docs(force=False):
    return get_docs_sync().sync_docs(force)

def check_docs_update():
    return get_docs_sync().check_for_updates()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync Manim docs from GitHub")
    parser.add_argument("--force", action="store_true", help="Force sync")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--check", action="store_true", help="Check for updates")
    args = parser.parse_args()
    
    sync = ManimDocsSync()
    
    if args.status:
        for k, v in sync.get_status().items():
            print(f"{k}: {v}")
    elif args.check:
        print(f"Update available: {sync.check_for_updates()}")
    else:
        ok, msg = sync.sync_docs(force=args.force)
        print(f"{'OK' if ok else 'FAILED'}: {msg}")

"""
Scheduler for automatic Manim docs syncing.
Can run continuously (daemon mode) or once (for cron/Task Scheduler).
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta

# Setup path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(SCRIPT_DIR, 'sync.log'))
    ]
)
logger = logging.getLogger(__name__)


def run_sync(force=False):
    """Execute a single sync operation."""
    try:
        from docs_sync import ManimDocsSync
        sync = ManimDocsSync()
        return sync.sync_docs(force=force)
    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        return False, str(e)


def next_run_time(hour=3, minute=0):
    """Calculate when the next scheduled run should be."""
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def run_loop(hour=3, minute=0, interval=24):
    """Main scheduler loop. Runs until interrupted."""
    logger.info(f"Scheduler started - interval: {interval}h, target time: {hour:02d}:{minute:02d}")
    
    while True:
        try:
            # Calculate wait time
            if interval == 24:
                target = next_run_time(hour, minute)
                wait = (target - datetime.now()).total_seconds()
            else:
                wait = interval * 3600
            
            logger.info(f"Next sync in {wait/3600:.1f} hours")
            time.sleep(wait)
            
            # Do the sync
            ok, msg = run_sync()
            logger.info(f"Sync {'completed' if ok else 'failed'}: {msg}")
            
            time.sleep(60)  # brief pause before next cycle
            
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(300)  # wait 5 min on error


def main():
    parser = argparse.ArgumentParser(description="Manim docs sync scheduler")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--force", action="store_true", help="Force sync")
    parser.add_argument("--interval", type=int, default=24, help="Hours between syncs")
    parser.add_argument("--hour", type=int, default=3, help="Target hour (0-23)")
    parser.add_argument("--minute", type=int, default=0, help="Target minute (0-59)")
    args = parser.parse_args()
    
    if args.once:
        ok, msg = run_sync(force=args.force)
        print(f"{'OK' if ok else 'FAILED'}: {msg}")
        sys.exit(0 if ok else 1)
    else:
        print("Starting scheduler (Ctrl+C to stop)...")
        run_loop(hour=args.hour, minute=args.minute, interval=args.interval)


if __name__ == "__main__":
    main()

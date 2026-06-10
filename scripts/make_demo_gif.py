# scripts/make_demo_gif.py
# Generates docs/assets/demo.gif — the README demo animation.
#
# The outputs shown in the GIF are REAL: this script runs the actual Chronos
# engine (remember / update / recall / query_at / consolidate) against a
# temporary database, backdates a few rows so the time-travel beat has
# history to reconstruct, and renders whatever the engine actually returned.
# Regenerate after any API change so the demo never lies:
#
#   pip install pillow          (dev-only; not a runtime dependency)
#   python scripts/make_demo_gif.py
#
# Output: docs/assets/demo.gif (~800x500, dark GitHub theme)

import os
import sys
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["CHRONOS_DB_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="chronos_demo_"), "demo.db"
)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from chronos.beliefs import BeliefEngine  # noqa: E402
from chronos.consolidation import ConsolidationEngine  # noqa: E402
from chronos.db import get_db, init_db  # noqa: E402
from chronos.memory import MemoryStore  # noqa: E402

# --- theme ------------------------------------------------------------------

W, H, PAD, LINE_H = 800, 640, 22, 24
BG      = "#0d1117"
FG      = "#e6edf3"
DIM     = "#8b949e"
GREEN   = "#7ee787"
CYAN    = "#79c0ff"
ORANGE  = "#ffa657"
COMMENT = "#6e7681"

FONT_PATH = r"C:\Windows\Fonts\consola.ttf"
FONT_BOLD = r"C:\Windows\Fonts\consolab.ttf"


def _font(path, size=15):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


FONT = _font(FONT_PATH)
BOLD = _font(FONT_BOLD)


def _backdate(memory_id: str, days: int) -> str:
    """Shift a memory's created_at/updated_at into the past. Returns the ISO ts."""
    past = (datetime.now() - timedelta(days=days)).isoformat()
    with get_db() as db:
        db.execute(
            "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
            (past, past, memory_id),
        )
        db.commit()
    return past


# --- run the real engine ----------------------------------------------------

def run_scenario() -> list:
    """Execute real Chronos calls; return display steps [(cmd, [out lines])]."""
    init_db()
    store, beliefs = MemoryStore(), BeliefEngine()
    consolidation = ConsolidationEngine(beliefs)

    # Beat 1: remember, ~115 days ago
    m = store.remember("API rate limit is 100 requests/min", project="api")
    _backdate(m["id"], days=115)
    short_id = m["id"][:8]

    # Off-screen corpus (stored before recall so BM25 IDF is meaningful —
    # a single-document corpus scores every term ~0) + consolidation fodder.
    store.remember("Deploy pipeline runs on GitHub Actions runners")
    dup = store.remember("Deploy pipeline runs on GitHub Actions runners")
    doomed = store.remember("Old note nobody trusts anymore")
    beliefs.set_confidence(doomed["id"], 0.05, "unverified")
    _backdate(doomed["id"], days=90)
    _backdate(dup["id"], days=1)

    # Beat 2: the fact changed — update (old content snapshotted)
    store.update(m["id"], "API rate limit is 500 requests/min")

    # Beat 3: recall sees the present
    now_hit = store.recall("rate limit", recency_weight=0.0)["results"][0]

    # Beat 4: time-travel sees the past
    as_of = (datetime.now() - timedelta(days=100)).isoformat(timespec="seconds")
    past_hit = store.query_at("rate limit", timestamp=as_of)["results"][0]

    # Beat 5: consolidation over everything stored above
    report = consolidation.consolidate(auto_merge=True)

    orient = report["orient"]
    gathered = report["gather"]["duplicates_found"]
    merged = report["consolidate"]["duplicates_merged"]
    decayed = report["consolidate"]["memories_decayed"]
    prune = report["prune"]

    return [
        ('remember("API rate limit is 100 requests/min", project="api")',
         [(f"  [ok] stored  id={short_id}...  ({m['token_estimate']} tokens)", GREEN)]),
        ("# ... months pass, the limit changes ...", None),
        (f'update_memory("{short_id}...", "API rate limit is 500 requests/min")',
         [("  [ok] updated -- previous version snapshotted automatically", GREEN)]),
        ('recall("rate limit")',
         [(f'  1. "{now_hit["content"]}"', FG),
          (f'     score {now_hit["score"]:.2f}   confidence '
           f'{now_hit.get("confidence", 0.5):.2f}   source {now_hit["source"]}', DIM)]),
        (f'query_at("rate limit", "{as_of}")',
         [(f'  1. "{past_hit["content"]}"', ORANGE),
          (f"     reconstructed as of {as_of[:10]} -- time-travel", DIM)]),
        ("# ... a few more memories accumulate over the weeks ...", None),
        ("consolidate_memories(auto_merge=True)",
         [(f"  orient   {orient['total_active']} active · "
           f"avg confidence {orient['avg_confidence']:.2f} · "
           f"{orient['stale_count']} stale", CYAN),
          (f"  gather   {gathered} duplicate pair -> merged {merged}, "
           "survivor confidence boosted", CYAN),
          (f"  decay    {decayed} unreviewed memories lost confidence", CYAN),
          (f"  prune    {prune['prune_candidates']} candidate "
           f"(confidence {prune['prune_details'][0]['confidence']:.2f}, "
           f"retention {prune['prune_details'][0]['retention']:.2f}) -- dry run", CYAN)]),
    ]


# --- render -----------------------------------------------------------------

def draw_frame(lines, caret=False):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # window chrome
    d.rounded_rectangle([6, 6, W - 6, H - 6], radius=10, outline="#30363d", width=1)
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        d.ellipse([PAD + i * 22, 16, PAD + 12 + i * 22, 28], fill=c)
    d.text((W // 2 - 90, 14), "chronos -- temporal memory", font=FONT, fill=COMMENT)

    y = 52
    for text, color in lines:
        d.text((PAD, y), text, font=FONT, fill=color)
        y += LINE_H
    if caret and lines:
        last_text = lines[-1][0]
        x = PAD + d.textlength(last_text, font=FONT)
        d.rectangle([x + 3, y - LINE_H + 3, x + 12, y - 5], fill=FG)
    return img


def build_frames(steps):
    frames, durations, shown = [], [], []

    def emit(img, ms):
        frames.append(img)
        durations.append(ms)

    for cmd, out in steps:
        if out is None:  # comment beat
            shown.append((cmd, COMMENT))
            emit(draw_frame(shown), 1100)
            shown.append(("", FG))
            continue
        # type the command in three increments
        for frac in (0.45, 1.0):
            partial = cmd[: max(1, int(len(cmd) * frac))]
            emit(draw_frame(shown + [("> " + partial, FG)], caret=True), 260)
        shown.append(("> " + cmd, FG))
        emit(draw_frame(shown, caret=True), 350)
        for line in out:
            shown.append(line)
        emit(draw_frame(shown), 1700)
        shown.append(("", FG))

    emit(draw_frame(shown), 4500)  # hold the ending
    return frames, durations


def main():
    steps = run_scenario()
    frames, durations = build_frames(steps)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, "demo.gif"))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_kb = os.path.getsize(out_path) // 1024
    print(f"wrote {out_path} ({len(frames)} frames, {size_kb} KB)")


if __name__ == "__main__":
    main()

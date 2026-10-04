"""Keyboard labeller for the review queue.

Replaces the F2/LeftArrow/key/Tab rename dance with straight sequential typing.

Each crop is shown twice: the full ROI upscaled, and the 20x32 NEAREST downsample
that `prepare_joe_data.py` will actually feed the model. Judge legibility against
the *second* one -- if the glyph is gone there, the sample carries no signal no
matter how it looks at full size.

Frame mode (C_uncertain, filenames `<stamp>_<pos>_saw-<label>.jpg`) walks a cursor
left to right across dig2..dig6. One keystroke per position, commits on the fifth:

    0-9         set this digit                      (number row or numpad)
    -           NaN                                 (`n` also works)
    Space       original value was fine -- advance
    Backspace   drop this crop from the sample set  (`Del` also works)
    Left        step back one position
    Enter       accept every remaining position as-is and commit
    Esc         restart this frame

So an all-8s screen is `88888`, and `saw 5,1,5,2,N` that should read `5,7,5,2,7`
is `<space>7<space><space>7`.

Single mode (A/B/E tiers, filenames `<label>_main_<pos>_<stamp>.jpg`) is the same
minus the cursor: one keystroke decides the crop and advances.

Always available:  u undo last frame   s toggle NEAREST/LANCZOS   q save and quit

Usage:
    python tools/label_frames.py --in work/review/queue/C_uncertain --out work/labeled
    python tools/label_frames.py --in work/review/queue/A_conflict  --out work/labeled
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageTk

FRAME_RE = re.compile(
    r"^(?P<stamp>\d{8}-\d{6})_(?P<pos>dig\d)_saw-(?P<saw>10|[0-9]|N)\.jpg$", re.I)
SINGLE_RE = re.compile(
    r"^(?P<label>10|[0-9]|N|REVIEW)_main_(?P<pos>dig\d)_(?P<stamp>\d{8}-\d{6})\.jpg$",
    re.I)

POSITIONS = ("dig2", "dig3", "dig4", "dig5", "dig6")
MODEL_INPUT = (20, 32)  # what prepare_joe_data.py resizes to
PREVIEW_H = 240
MODEL_VIEW_H = 192

KEEP, DROP = "\x00keep", "\x00drop"

# With NumLock OFF, Windows sends navigation keysyms from the numpad instead of
# digits. Map both spellings so the keypad works either way.
KP_NUMLOCK_OFF = {
    "KP_Insert": "0", "KP_End": "1", "KP_Down": "2", "KP_Next": "3",
    "KP_Left": "4", "KP_Begin": "5", "KP_Right": "6", "KP_Home": "7",
    "KP_Up": "8", "KP_Prior": "9",
}


def key_value(ev) -> str | None:
    """Translate a keypress into a label, KEEP, DROP, or None."""
    sym = ev.keysym
    if sym in KP_NUMLOCK_OFF:
        return KP_NUMLOCK_OFF[sym]
    if sym.startswith("KP_") and sym[3:].isdigit():
        return sym[3:]
    if len(sym) == 1 and sym.isdigit():
        return sym
    if ev.char and ev.char.isdigit():
        return ev.char
    if sym in ("minus", "KP_Subtract", "n", "N", "period", "KP_Decimal"):
        return "N"
    if sym == "space":
        return KEEP
    if sym in ("BackSpace", "Delete", "KP_Delete"):
        return DROP
    return None


@dataclass
class Item:
    path: Path
    pos: str
    stamp: str
    saw: str
    derived: str = ""


@dataclass
class Group:
    key: str
    items: list[Item] = field(default_factory=list)


def load_manifest(root: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for parent in [root, *root.parents]:
        man = parent / "MANIFEST.csv"
        if man.is_file():
            with man.open(newline="", encoding="utf8") as fh:
                for row in csv.DictReader(fh):
                    if row.get("derived_label"):
                        out[(row["stamp"], row["pos"])] = row["derived_label"]
            break
    return out


def discover(root: Path) -> tuple[str, list[Group]]:
    man = load_manifest(root)
    frame_items: dict[str, list[Item]] = defaultdict(list)
    single: list[Group] = []
    mode = None

    for p in sorted(root.rglob("*.jpg")):
        m = FRAME_RE.match(p.name)
        if m:
            mode = mode or "frame"
            it = Item(p, m.group("pos").lower(), m.group("stamp"),
                      m.group("saw").upper())
            it.derived = man.get((it.stamp, it.pos), "")
            frame_items[it.stamp].append(it)
            continue
        m = SINGLE_RE.match(p.name)
        if m:
            mode = mode or "single"
            it = Item(p, m.group("pos").lower(), m.group("stamp"),
                      m.group("label").upper())
            it.derived = man.get((it.stamp, it.pos), it.saw)
            single.append(Group(key=p.name, items=[it]))

    if mode == "frame":
        return "frame", [
            Group(key=s, items=sorted(frame_items[s], key=lambda i: i.pos))
            for s in sorted(frame_items)
        ]
    return "single", single


class Labeller:
    def __init__(self, mode: str, groups: list[Group], out: Path, smooth: bool):
        self.mode, self.groups, self.out = mode, groups, out
        self.smooth = smooth
        self.i = 0
        self.cursor = 0
        self.work: dict[str, str] = {}
        self.decisions: dict[str, dict[str, str]] = {}
        self.history: list[str] = []
        self._imgs: list[ImageTk.PhotoImage] = []

        self.out.mkdir(parents=True, exist_ok=True)
        self.journal = self.out / ".label_journal.json"
        if self.journal.is_file():
            self.decisions = json.loads(self.journal.read_text("utf8"))
            while self.i < len(self.groups) and self.groups[self.i].key in self.decisions:
                self.i += 1

        self.root = tk.Tk()
        self.root.title("label_frames")
        self.root.configure(bg="#1e1e1e")
        self.header = tk.Label(self.root, font=("Consolas", 13), fg="#eee", bg="#1e1e1e")
        self.header.pack(pady=(8, 4))
        self.canvas = tk.Frame(self.root, bg="#1e1e1e")
        self.canvas.pack(padx=10)
        self.footer = tk.Label(self.root, font=("Consolas", 11), fg="#9cf",
                               bg="#1e1e1e", justify="left")
        self.footer.pack(pady=(6, 10))

        self.root.bind("<Key>", self.on_key)
        self.root.focus_force()  # without this the window can start unfocused
        self.load_group()

    # ---------- state ----------
    def load_group(self) -> None:
        self.cursor = 0
        self.work = {}
        if self.i < len(self.groups):
            for it in self.groups[self.i].items:
                self.work[it.pos] = it.derived or it.saw
        self.render()

    def positions(self) -> list[str]:
        if self.i >= len(self.groups):
            return []
        return [it.pos for it in self.groups[self.i].items]

    def commit(self) -> None:
        g = self.groups[self.i]
        self.decisions[g.key] = dict(self.work)
        self.history.append(g.key)
        self.i += 1
        self.save()
        self.load_group()

    # ---------- rendering ----------
    def render(self) -> None:
        for w in self.canvas.winfo_children():
            w.destroy()
        self._imgs.clear()

        if self.i >= len(self.groups):
            self.header.config(text="done -- all groups decided.   q to quit")
            self.footer.config(text=f"{len(self.decisions)} decisions -> {self.out}")
            return

        g = self.groups[self.i]
        pos_list = self.positions()
        resample = Image.Resampling.LANCZOS if self.smooth else Image.Resampling.NEAREST

        for col, it in enumerate(g.items):
            active = self.mode == "frame" and col == self.cursor
            cell = tk.Frame(self.canvas, bg="#ffd24a" if active else "#1e1e1e",
                            padx=3, pady=3)
            cell.grid(row=0, column=col, padx=5)

            im = Image.open(it.path).convert("L")
            w = max(1, int(im.width * PREVIEW_H / im.height))
            big = ImageTk.PhotoImage(im.resize((w, PREVIEW_H), resample))
            self._imgs.append(big)
            tk.Label(cell, image=big, bd=1, relief="solid").pack()

            small = im.resize(MODEL_INPUT, Image.Resampling.NEAREST)
            mw = max(1, int(MODEL_INPUT[0] * MODEL_VIEW_H / MODEL_INPUT[1]))
            mv = ImageTk.PhotoImage(
                small.resize((mw, MODEL_VIEW_H), Image.Resampling.NEAREST))
            self._imgs.append(mv)
            tk.Label(cell, image=mv, bd=1, relief="solid").pack(pady=(4, 0))

            cur = self.work.get(it.pos, "?")
            shown = {KEEP: it.saw, DROP: "DROP"}.get(cur, cur)
            changed = cur not in (it.derived or it.saw, KEEP)
            tk.Label(
                cell, text=f"{it.pos}  saw={it.saw}", font=("Consolas", 10),
                fg="#333" if active else "#bbb",
                bg="#ffd24a" if active else "#1e1e1e",
            ).pack()
            tk.Label(
                cell, text=f" {shown} ", font=("Consolas", 22, "bold"),
                fg=("#c00" if shown == "DROP" else "#333" if active
                    else "#ffd24a" if changed else "#8f8"),
                bg="#ffd24a" if active else "#1e1e1e",
            ).pack()

        done = len(self.decisions)
        self.header.config(
            text=f"[{self.i+1}/{len(self.groups)}]  {g.key}   done={done}   "
                 f"(top: ROI   bottom: 20x32 model view)")

        row = " ".join(
            {KEEP: ".", DROP: "X"}.get(self.work.get(p, "?"), self.work.get(p, "?"))
            for p in pos_list
        )
        at = pos_list[self.cursor] if self.cursor < len(pos_list) else "-"
        if self.mode == "frame":
            self.footer.config(
                text=f"row: {row}     cursor: {at}\n"
                     f"0-9 set   - NaN   Space keep   BackSpace drop crop   "
                     f"Left back   Enter commit   Esc restart   u undo   q save+quit")
        else:
            self.footer.config(
                text="0-9 set   - NaN   Space keep   BackSpace drop   "
                     "u undo   s smooth   q save+quit")

    # ---------- input ----------
    def on_key(self, ev) -> None:
        sym = ev.keysym

        if sym == "q":
            self.save()
            self.root.destroy()
            return
        if sym == "s":
            self.smooth = not self.smooth
            self.render()
            return
        if sym == "u" and self.history:
            key = self.history.pop()
            self.decisions.pop(key, None)
            self.i = next((j for j, g in enumerate(self.groups) if g.key == key), self.i)
            self.save()
            self.load_group()
            return
        if self.i >= len(self.groups):
            return
        if sym == "Escape":
            self.load_group()
            return
        if sym in ("Return", "KP_Enter"):
            self.commit()
            return
        if sym == "Left":
            self.cursor = max(0, self.cursor - 1)
            self.render()
            return
        if sym == "Right":
            self.cursor = min(len(self.positions()) - 1, self.cursor + 1)
            self.render()
            return

        val = key_value(ev)
        if val is None:
            return

        pos_list = self.positions()
        if self.mode == "single":
            it = self.groups[self.i].items[0]
            self.work[it.pos] = it.derived or it.saw if val is KEEP else val
            self.commit()
            return

        if self.cursor >= len(pos_list):
            return
        pos = pos_list[self.cursor]
        it = self.groups[self.i].items[self.cursor]
        if val == KEEP:
            self.work[pos] = it.derived or it.saw
        else:
            self.work[pos] = val
        self.cursor += 1
        if self.cursor >= len(pos_list):
            self.commit()
        else:
            self.render()

    # ---------- output ----------
    def save(self) -> None:
        self.journal.write_text(json.dumps(self.decisions, indent=1), "utf8")

    def run(self) -> None:
        self.root.mainloop()
        self.write_out()

    def write_out(self) -> None:
        kept = dropped = 0
        for g in self.groups:
            dec = self.decisions.get(g.key)
            if not dec:
                continue
            for it in g.items:
                lab = dec.get(it.pos)
                if not lab or lab in (DROP, "REVIEW"):
                    dropped += 1
                    continue
                if lab == KEEP:
                    lab = it.derived or it.saw
                shutil.copy2(it.path, self.out / f"{lab}_main_{it.pos}_{it.stamp}.jpg")
                kept += 1
        print(f"wrote {kept} labelled crops to {self.out}  ({dropped} dropped)")
        print(f"journal: {self.journal}  (delete it to start over)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smooth", action="store_true",
                    help="LANCZOS preview instead of NEAREST (default NEAREST)")
    args = ap.parse_args()

    mode, groups = discover(args.src)
    if not groups:
        raise SystemExit(f"no labellable images under {args.src}")
    print(f"{mode} mode: {len(groups)} groups from {args.src}")
    Labeller(mode, groups, args.out, args.smooth).run()


if __name__ == "__main__":
    main()

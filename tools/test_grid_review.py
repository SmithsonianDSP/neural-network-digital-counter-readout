"""Self-test for tools/grid_review.py pure logic (no GUI).

    python tools/test_grid_review.py
    python tools/grid_review.py --selftest
"""
from __future__ import annotations

import csv
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

import grid_review as gr  # noqa: E402


def _img(path: Path, v: int) -> str:
    Image.new("RGB", (94, 202), (v, v, v)).save(path)
    return str(path)


def _write_queue(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=gr.QUEUE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in gr.QUEUE_FIELDS})
    return path


def _fixture(tmp: Path):
    """Queue A: cell (dig6,'1',flash) with 50 items of varied brightness,
    cell (dig5,'7',day) with 3 items. Queue B: 2 complete frames dig2..dig6."""
    imgs = tmp / "img"
    imgs.mkdir()
    rows_a = []
    for k in range(50):
        stamp = f"20260920-01{k // 60:02d}{k % 60:02d}"
        b = (k * 37) % 50          # scrambled brightness
        rows_a.append(dict(queue="A", item_id=f"{stamp}_dig6", stamp=stamp, pos="6",
                           src=_img(imgs / f"1_main_dig6_{stamp}.jpg", 40 + b),
                           proposed="1", bucket="flash", brightness=str(b),
                           context=f"ctx {k}"))
    for k in range(3):
        stamp = f"20260920-12000{k}"
        rows_a.append(dict(queue="A", item_id=f"{stamp}_dig5", stamp=stamp, pos="dig5",
                           src=_img(imgs / f"7_main_dig5_{stamp}.jpg", 100 + k),
                           proposed="7", bucket="day", brightness=str(100 + k)))
    rows_b = []
    for stamp in ("20260920-130000", "20260920-130200"):
        for p, lab in zip(range(2, 7), ["5", "7", "10", "3", "9"]):
            rows_b.append(dict(queue="B", item_id=f"{stamp}_dig{p}", stamp=stamp, pos=str(p),
                               src=_img(imgs / f"{lab}_main_dig{p}_{stamp}.jpg", 120),
                               proposed=lab, bucket="day", brightness="",
                               context="reading 57103"))
    qa = _write_queue(tmp / "qa.csv", rows_a)
    qb = _write_queue(tmp / "qb.csv", rows_b)
    return qa, qb


def t_load_and_normalise(tmp):
    qa, qb = _fixture(tmp)
    items, warns = gr.load_queues([qa, qb])
    assert len(items) == 63, len(items)
    assert not warns, warns
    by = {it.item_id: it for it in items}
    assert by["20260920-130000_dig4"].proposed == "N"          # '10' -> 'N'
    assert by["20260920-120000_dig5"].pos == 5                 # 'dig5' -> 5
    assert by["20260920-130000_dig2"].brightness is None
    n = gr.fill_brightness(items)
    assert n == 10 and abs(by["20260920-130000_dig2"].brightness - 120) < 1.5
    # duplicate item ids across queues: first wins, warned
    items2, warns2 = gr.load_queues([qa, qa])
    assert len(items2) == 53 and len(warns2) == 53


def t_paging_and_sorting(tmp):
    qa, qb = _fixture(tmp)
    items, _ = gr.load_queues([qa, qb])
    gr.fill_brightness(items)
    s = gr.Session(items, tmp / "ledger.csv", page_size=48)
    assert s.cells[0] == ("A", 6, "1", "flash"), s.cells
    assert s.cells[-1][0] == "B"                                # queue order kept
    pg = s.next_page()
    # 50 items, page size 48 -> balanced 25 + 25, not 48 + 2
    assert pg.kind == "grid" and len(pg.items) == 25, len(pg.items)
    br = [it.brightness for it in pg.items]
    assert br == sorted(br) and br[0] == 0, br                  # dark -> bright
    assert s.pages_left_in_cell(pg.cell) == 2
    # cell order options
    s2 = gr.Session(items, tmp / "ledger2.csv", cell_order="small")
    assert s2.cells[0] == ("A", 5, "7", "day"), s2.cells[0]
    s3 = gr.Session(items, tmp / "ledger3.csv", cell_order="sorted")
    assert [c[1] for c in s3.cells if c[0] == "A"] == [5, 6]


def t_commit_rows(tmp):
    qa, qb = _fixture(tmp)
    items, _ = gr.load_queues([qa, qb])
    led = tmp / "ledger.csv"
    s = gr.Session(items, led, page_size=48)
    pg = s.next_page()
    a, b, c, d, e, f = pg.items[:6]
    pg.mark(a, "artifact")
    pg.mark(b, "drift")
    pg.mark(c, "illegible")
    pg.mark(d, "ok_derived")
    pg.mark(e, "7")                     # relabel
    pg.mark(f, "1")                     # relabel to proposed == no-op
    assert f.item_id not in pg.marks
    pg.mark(a, "artifact")              # toggle off ...
    assert a.item_id not in pg.marks
    pg.mark(a, "artifact")              # ... and on again
    pg.mark(b, "clear")
    assert b.item_id not in pg.marks
    pg.mark(b, "drift")
    pid = s.commit(pg)
    rows = gr.read_ledger_rows(led)
    # 25 tiles + frame-wide drift for b's 4 other positions (not loaded, not on disk)
    assert len(rows) == 29 and all(r["page_id"] == pid for r in rows), len(rows)
    extra = rows[25:]
    assert {r["item_id"] for r in extra} == {f"{b.stamp}_dig{p}" for p in (2, 3, 4, 5)}
    assert all(r["verdict"] == "drift" and r["queue"] == "A" and r["final"] == ""
               for r in extra)
    rows = rows[:25]
    got = {r["item_id"]: (r["verdict"], r["final"]) for r in rows}
    assert got[a.item_id] == ("artifact", "")
    assert got[b.item_id] == ("drift", "")
    assert got[c.item_id] == ("illegible", "")
    assert got[d.item_id] == ("ok_derived", "1")
    assert got[e.item_id] == ("relabeled", "7")
    assert got[f.item_id] == ("ok", "1")
    assert sum(v == ("ok", "1") for v in got.values()) == 20
    r0 = next(r for r in rows if r["item_id"] == e.item_id)
    assert r0["proposed"] == "1" and r0["queue"] == "A" and r0["pos"] == "6"
    assert r0["src"] == e.src and r0["stamp"] == e.stamp and "T" in r0["reviewed_at"]
    hdr = led.read_text("utf-8").splitlines()[0]
    assert hdr.split(",") == gr.LEDGER_FIELDS
    assert s.stats["ok"] == 20 and s.committed == 29 and s.stats["drift"] == 5
    # second page is the rest of the cell
    pg2 = s.next_page()
    assert len(pg2.items) == 25 and not {i.item_id for i in pg2.items} & set(got)


def t_resume(tmp):
    qa, qb = _fixture(tmp)
    items, _ = gr.load_queues([qa, qb])
    led = tmp / "ledger.csv"
    s = gr.Session(items, led)
    p1 = s.next_page()
    s.commit(p1)
    first = {it.item_id for it in p1.items}
    # fresh process
    items2, _ = gr.load_queues([qa, qb])
    s2 = gr.Session(items2, led)
    assert s2.preexisting == 25
    p2 = s2.next_page()
    assert not first & {it.item_id for it in p2.items}
    assert s2.cell_progress(p2.cell) == (25, 50)
    assert s2.overall_progress() == (25, 63)
    s2.commit(p2)
    p3 = s2.next_page()
    assert p3.cell == ("A", 5, "7", "day")
    s2.commit(p3)
    p4 = s2.next_page()                     # queue B, one cell per position
    assert p4.items[0].queue == "B"
    size_before = led.stat().st_size
    text_before = led.read_bytes()
    s2.commit(p4)
    assert led.read_bytes()[:size_before] == text_before     # append-only


def t_undo(tmp):
    qa, qb = _fixture(tmp)
    items, _ = gr.load_queues([qa, qb])
    led = tmp / "ledger.csv"
    s = gr.Session(items, led)
    p1 = s.next_page()
    pid1 = s.commit(p1)
    p2 = s.next_page()
    p2.mark(p2.items[3], "5")
    p2.mark(p2.items[4], "artifact")
    pid2 = s.commit(p2)
    before = led.read_bytes()
    ok, msg = s.undo()
    assert ok, msg
    after = led.read_bytes()
    assert after[:len(before)] == before                      # tombstones appended
    rows = gr.read_ledger_rows(led)
    tomb = [r for r in rows if r["verdict"] == gr.UNDONE]
    assert len(tomb) == 25 and all(r["page_id"] == pid2 for r in tomb)
    eff = gr.load_ledger(led)
    assert len(eff) == 25 and all(r["page_id"] == pid1 for r in eff.values())
    assert s.committed == 25 and s.stats["relabeled"] == 0 and s.stats["artifact"] == 0
    # the undone page comes back next, with its marks
    p2b = s.next_page()
    assert p2b.restored_from == pid2
    assert {i.item_id for i in p2b.items} == {i.item_id for i in p2.items}
    assert p2b.marks == {p2.items[3].item_id: ("relabeled", "5"),
                         p2.items[4].item_id: ("artifact", "")}
    p2b.mark(p2.items[4], "clear")                            # fix the mistake
    pid3 = s.commit(p2b)
    eff = gr.load_ledger(led)
    assert len(eff) == 50
    assert eff[p2.items[4].item_id]["verdict"] == "ok"
    assert eff[p2.items[3].item_id]["page_id"] == pid3
    # undo works across sessions on the last page ...
    s_new = gr.Session(gr.load_queues([qa, qb])[0], led)
    ok, _ = s_new.undo()
    assert ok and len(gr.load_ledger(led)) == 25
    # ... and refuses when that page's items aren't loaded
    s_a = gr.Session(gr.load_queues([qa])[0], led)
    pga = s_a.next_page()
    s_a.commit(pga)
    s_b = gr.Session(gr.load_queues([qb])[0], led)
    ok, msg = s_b.undo()
    assert not ok and "not in" in msg, msg
    s_empty = gr.Session(gr.load_queues([qb])[0], tmp / "empty.csv")
    assert s_empty.undo() == (False, "nothing to undo")


def t_ledger_semantics(tmp):
    led = tmp / "l.csv"
    rows = [
        dict(item_id="x", verdict="ok", final="1", page_id="P1"),
        dict(item_id="y", verdict="artifact", final="", page_id="P1"),
        dict(item_id="x", verdict="relabeled", final="7", page_id="P2"),
        dict(item_id="y", verdict="ok", final="3", page_id="P3"),
        dict(item_id="y", verdict=gr.UNDONE, final="", page_id="P3"),
        dict(item_id="z", verdict="drift", final="", page_id="P4"),
    ]
    w = gr.LedgerWriter(led)
    w.append(rows)
    with led.open("ab") as fh:                    # simulate a torn last line
        fh.write(b"q,2026,6,/x,A,1,1,o")
    eff = gr.load_ledger(led)
    assert set(eff) == {"x", "y", "z"}, eff
    assert eff["x"]["final"] == "7"                    # latest wins
    assert eff["y"]["verdict"] == "artifact"           # P3 undone -> P1 row effective
    assert list(gr.effective_pages(led)) == ["P1", "P2", "P4"]
    gr.LedgerWriter(led).append([dict(item_id="w", verdict="ok", final="2",
                                      page_id="P5")])
    eff = gr.load_ledger(led)
    assert eff["w"]["final"] == "2" and "q" not in eff
    assert gr.load_ledger(tmp / "nope.csv") == {}


def t_frame_mode(tmp):
    qa, qb = _fixture(tmp)
    items, _ = gr.load_queues([qb])
    gr.fill_brightness(items)
    s = gr.Session(items, tmp / "led.csv", mode="frame", frames_per_page=1)
    pg = s.next_page()
    assert pg.kind == "frame" and len(pg.frames) == 1
    assert [it.pos for it in pg.items] == [2, 3, 4, 5, 6]
    assert pg.frames[0][0] == ("B", "20260920-130000")
    pg.mark(pg.items[2], "4")
    s.commit(pg)
    pg2 = s.next_page()
    assert pg2.frames[0][0] == ("B", "20260920-130200")
    s.commit(pg2)
    assert s.next_page() is None
    eff = gr.load_ledger(tmp / "led.csv")
    assert eff["20260920-130000_dig4"]["verdict"] == "relabeled"
    assert eff["20260920-130000_dig4"]["proposed"] == "N"
    # neighbours: loaded queue items first, else same-dir glob by stamp
    sa = gr.Session(gr.load_queues([qa])[0], tmp / "led2.csv")
    it = sa.by_id["20260920-120000_dig5"]
    nb = sa.frame_neighbours(it)
    assert set(nb) == {5}, nb                        # no other positions on disk
    sb = gr.Session(items[:1], tmp / "led3.csv")     # only dig2 loaded
    nb = sb.frame_neighbours(items[0])
    assert set(nb) == {2, 3, 4, 5, 6} and nb[4][1] == "N", nb


def t_sample_queue(tmp):
    raw = tmp / "raw" / "01"
    raw.mkdir(parents=True)
    for stamp in ("20260920-010000", "20260920-010200", "20260920-070400"):
        for p, lab in zip(range(2, 7), ["5", "7", "N", "0", "10"]):
            _img(raw / f"{lab}_main_dig{p}_{stamp}.jpg", 60)
    out = tmp / "q.csv"
    n = gr.make_sample_queue(out, tmp / "raw", n=12)
    assert n == 12
    items, warns = gr.load_queues([out])
    assert len(items) == 12 and not warns
    assert {it.bucket for it in items} <= {"flash", "day"}
    assert all(it.proposed == "N" for it in items if it.pos in (4, 6))   # '10' -> N
    assert all(it.brightness is not None for it in items)



# ---------------------------------------------------------------- new: frame drift


def _frame_imgs(d: Path, stamp: str, labels=("5", "8", "10", "3", "1")) -> dict[int, str]:
    """All five crops of one capture on disk (filename labels as the collector writes)."""
    d.mkdir(parents=True, exist_ok=True)
    return {p: _img(d / f"{lab}_main_dig{p}_{stamp}.jpg", 90 + p)
            for p, lab in zip(range(2, 7), labels)}


def t_frame_drift_marks(tmp):
    """GUI marks: d marks the whole frame on the page; any change clears the frame."""
    qa, qb = _fixture(tmp)
    items, _ = gr.load_queues([qb])
    s = gr.Session(items, tmp / "led.csv", mode="frame", frames_per_page=2)
    pg = s.next_page()
    f1 = [it for it in pg.items if it.stamp == "20260920-130000"]
    f2 = [it for it in pg.items if it.stamp == "20260920-130200"]
    pg.mark(f1[1], "drift")
    assert {k for k, v in pg.marks.items() if v[0] == "drift"} == {i.item_id for i in f1}
    assert not any(i.item_id in pg.marks for i in f2)
    pg.mark(f1[3], "drift")                       # d again on a mate: clears the frame
    assert not pg.marks, pg.marks
    pg.mark(f1[0], "drift")
    pg.mark(f1[4], "7")                           # relabel a mate: frame drift cleared
    assert pg.marks == {f1[4].item_id: ("relabeled", "7")}, pg.marks
    pg.mark(f1[2], "drift")
    pg.mark(f1[2], "clear")
    assert not any(v[0] == "drift" for v in pg.marks.values())
    pg.mark(f1[2], "drift")
    pid = s.commit(pg)
    rows = [r for r in gr.read_ledger_rows(tmp / "led.csv") if r["page_id"] == pid]
    assert len(rows) == 10                        # all 5 were on the page: no extras
    v = {r["item_id"]: r["verdict"] for r in rows}
    assert all(v[i.item_id] == "drift" for i in f1) and all(v[i.item_id] == "ok" for i in f2)


def t_frame_drift_grid_and_offqueue(tmp):
    """Grid page: drift on one crop writes drift rows for the frame's other positions,
    both loaded (other cells) and not loaded (found on disk); they are skipped on
    resume; undo removes them and restores only the original page."""
    raw = tmp / "raw" / "20260920" / "01"
    st1, st2, st3 = "20260920-010000", "20260920-010400", "20260920-010800"
    paths = {st: _frame_imgs(raw, st) for st in (st1, st2, st3)}
    # queue X: dig6 of all three frames + dig3 of st1 and st2 (other cell)
    rows = [dict(queue="X", item_id=f"{st}_dig6", stamp=st, pos="6", src=paths[st][6],
                 proposed="1", bucket="flash", brightness=str(k)) for k, st in
            enumerate((st1, st2, st3))]
    rows += [dict(queue="X", item_id=f"{st}_dig3", stamp=st, pos="3", src=paths[st][3],
                  proposed="8", bucket="flash", brightness="5") for st in (st1, st2)]
    q = _write_queue(tmp / "x.csv", rows)
    led = tmp / "led.csv"
    s = gr.Session(gr.load_queues([q])[0], led)
    pg = s.next_page()
    assert pg.cell == ("X", 6, "1", "flash") and len(pg.items) == 3
    it1 = next(i for i in pg.items if i.stamp == st1)
    pg.mark(it1, "drift")
    assert list(pg.marks) == [it1.item_id]       # mates are in other cells
    pid = s.commit(pg)
    rows = [r for r in gr.read_ledger_rows(led) if r["page_id"] == pid]
    assert len(rows) == 3 + 4, len(rows)
    extra = {r["item_id"]: r for r in rows[3:]}
    assert set(extra) == {f"{st1}_dig{p}" for p in (2, 3, 4, 5)}
    assert all(r["verdict"] == "drift" and r["queue"] == "X" for r in extra.values())
    assert extra[f"{st1}_dig3"]["src"] == paths[st1][3]          # loaded item
    assert extra[f"{st1}_dig4"]["src"] == paths[st1][4]          # found on disk
    assert extra[f"{st1}_dig4"]["proposed"] == "N"               # filename '10' -> N
    assert extra[f"{st1}_dig2"]["proposed"] == "5"
    # the dig3 cell now only offers st2 (st1 is a drift frame)
    pg2 = s.next_page()
    assert pg2.cell == ("X", 3, "8", "flash")
    assert [i.stamp for i in pg2.items] == [st2]
    # resume in a fresh session: same
    s2 = gr.Session(gr.load_queues([q])[0], led)
    assert s2.done(s2.by_id[f"{st1}_dig3"]) and st1 in s2.drift
    assert [i.stamp for i in s2.next_page().items] == [st2]
    assert gr.load_effective(led)[f"{st1}_dig5"]["rejected"] == "drift"
    # undo: allowed although dig2/4/5 are not loaded; removes all 7 rows
    ok, msg = s2.undo()
    assert ok, msg
    eff = gr.load_effective(led)
    assert not any(k.startswith(st1) for k in eff), eff.keys()
    assert st1 not in s2.drift
    back = s2.next_page()
    assert back.restored_from == pid
    assert {i.item_id for i in back.items} == {f"{st}_dig6" for st in (st1, st2, st3)}
    assert back.marks == {it1.item_id: ("drift", "")}
    # a re-commit writes the same frame rows again
    s2.commit(back)
    assert sum(1 for e in gr.load_effective(led).values()
               if e["screen_verdict"] == "drift") == 5


def t_propagate_frame_drift(tmp):
    raw = tmp / "raw"
    st1, st2 = "20260818-042958", "20260818-043358"
    p1 = _frame_imgs(raw, st1)
    p2 = _frame_imgs(raw, st2)
    led = tmp / "led.csv"
    w = gr.LedgerWriter(led)

    def r(st, pos, src, v, page, final=""):
        return dict(item_id=f"{st}_dig{pos}", stamp=st, pos=pos, src=src, queue="holdout",
                    proposed="5", final=final, verdict=v, page_id=page,
                    reviewed_at="2026-10-03T12:00:00-05:00")
    w.append([r(st1, 3, p1[3], "drift", "f1"), r(st1, 4, p1[4], "ok", "f1", "N"),
              r(st2, 2, p2[2], "drift", "f2"),
              r(st2, 2, "", gr.UNDONE, "f2")])        # st2's drift page was undone
    before = led.read_bytes()
    assert gr.propagate_frame_drift(led, dry_run=True) == 4
    assert led.read_bytes() == before
    n = gr.propagate_frame_drift(led)
    assert n == 4, n
    assert led.read_bytes()[:len(before)] == before                  # append-only
    eff = gr.load_effective(led)
    assert all(eff[f"{st1}_dig{p}"]["screen_verdict"] == "drift" for p in range(2, 7))
    assert eff[f"{st1}_dig4"]["rejected"] == "drift"                  # ok overridden
    assert eff[f"{st1}_dig6"]["src"] == p1[6] and eff[f"{st1}_dig6"]["queue"] == "holdout"
    assert eff[f"{st1}_dig6"]["proposed"] == "1"                     # filename label
    assert not any(k.startswith(st2) for k in eff)
    assert gr.propagate_frame_drift(led) == 0                        # idempotent
    assert gr.propagate_frame_drift(tmp / "none.csv", dry_run=True) == 0


# ---------------------------------------------------------------- new: two stages


def t_two_stage_semantics(tmp):
    led = tmp / "l.csv"

    def r(iid, v, final, page, prop="1"):
        return dict(item_id=iid, stamp=iid[:15], pos=iid[-1], verdict=v, final=final,
                    page_id=page, proposed=prop, queue="holdout")
    a, b, c, d, e, f, g, h = (f"20260901-12000{k}_dig6" for k in range(8))
    gr.LedgerWriter(led).append([
        r(a, "ok", "1", "S1"), r(a, "label_ok", "1", "V1"),             # ok + label_ok
        r(b, "ok", "1", "S1"), r(b, "label_fixed", "7", "V1"),          # ok + fixed
        r(c, "label_fixed", "7", "V1"), r(c, "relabeled", "3", "S2"),   # fixed beats screen
        r(d, "ok", "1", "S1"), r(d, "label_unsure", "", "V1"),          # unsure rejects
        r(e, "label_ok", "1", "V1"), r(e, "artifact", "", "S2"),        # screen rejects
        r(f, "ok", "1", "S1"), r(f, "label_fixed", "4", "V2"),
        r(f, "", "", "V2") | {"verdict": gr.UNDONE},                    # V2 undone
        r(g, "label_ok", "1", "V1"),                                    # label only
        r(h, "relabeled", "9", "S1"),                                   # screen only
    ])
    eff = gr.load_effective(led)
    assert (eff[a]["final"], eff[a]["rejected"]) == ("1", "")
    assert (eff[b]["final"], eff[b]["label_verdict"], eff[b]["screen_verdict"]) == \
        ("7", "label_fixed", "ok")
    assert eff[c]["final"] == "7" and eff[c]["screen_final"] == "3"
    assert eff[d]["rejected"] == "label_unsure" and eff[d]["final"] == ""
    assert eff[e]["rejected"] == "artifact" and eff[e]["final"] == ""
    assert eff[f]["final"] == "1" and eff[f]["label_verdict"] == ""   # V2 undone
    assert eff[g]["screen_verdict"] == "" and eff[g]["final"] == "1"
    assert eff[h]["final"] == "9" and eff[h]["label_verdict"] == ""
    # raw view still = latest row of either stage
    assert gr.load_ledger(led)[c]["verdict"] == "relabeled"


def _vq(tmp: Path, name="v.csv"):
    """Queue holdout: 3 frames; needs_verify on dig6 of every frame and on st1 dig5.
    dig6 is proposed 1 while the model saw 9."""
    raw = tmp / "raw" / "20260901" / "04"
    sts = ["20260901-040000", "20260901-040400", "20260901-040800"]
    rows = []
    for st in sts:
        paths = _frame_imgs(raw, st, labels=("5", "8", "1", "2", "9"))
        for p, lab in zip(range(2, 7), ("5", "8", "1", "2", "1")):
            nv = (p == 6) or (st == sts[0] and p == 5)
            rows.append(dict(queue="holdout", item_id=f"{st}_dig{p}", stamp=st, pos=str(p),
                             src=paths[p], proposed=lab, bucket="flash", brightness="50",
                             context=f"reading_est 58{p}12; model saw {'9' if p == 6 else lab}; "
                                     f"{'disagree' if nv else 'agree'}",
                             preflag="model_disagrees" if p == 6 else "", group=st,
                             needs_verify="1" if nv else "0"))
    return _write_queue(tmp / name, rows), sts


def t_verify_and_screen_modes(tmp):
    q, (st1, st2, st3) = _vq(tmp)
    items, warns = gr.load_queues([q])
    assert not warns
    by = {i.item_id: i for i in items}
    assert by[f"{st1}_dig6"].needs_verify and not by[f"{st1}_dig2"].needs_verify
    assert by[f"{st1}_dig6"].model_label == "9" and by[f"{st1}_dig2"].model_label == "5"
    led = tmp / "led.csv"
    # 1. the holdout was screened first in classic frame mode (like the real ledger):
    #    st3 is a drift frame, the rest ok
    s0 = gr.Session(items, led, mode="frame", frames_per_page=3)
    pg = s0.next_page()
    pg.mark(next(i for i in pg.items if i.item_id == f"{st3}_dig4"), "drift")
    s0.commit(pg)
    assert s0.next_page() is None
    # 2. screen mode now: everything already screened, nothing pending, none unverified
    ss = gr.Session(gr.load_queues([q])[0], led, task="screen")
    # (the 3 screened-but-unverified nv crops are reported; st3 is a drift frame)
    assert ss.next_page() is None and ss.unverified_skipped == 3
    # 3. verify mode: only nv items, minus the drift frame (st3)
    sv = gr.Session(gr.load_queues([q])[0], led, task="verify", cell_order="size")
    assert len(sv.items) == 4 and sv.not_needing_verify == 11
    assert sv.overall_progress() == (1, 4)                # st3 dig6: drift frame
    p1 = sv.next_page()
    assert p1.cell == ("holdout", 6, "1", "flash") and len(p1.items) == 2
    assert p1.task == "verify" and p1.resolved(p1.items[0]) == ("label_ok", "1")
    i1 = next(i for i in p1.items if i.stamp == st1)
    i2 = next(i for i in p1.items if i.stamp == st2)
    p1.mark(i1, "9")
    assert p1.marks[i1.item_id] == ("label_fixed", "9")
    p1.mark(i2, "unsure")
    p1.mark(i2, "unsure")                                  # toggles off
    assert i2.item_id not in p1.marks
    pid = sv.commit(p1)
    assert pid.startswith("gv")
    p2 = sv.next_page()
    assert p2.cell == ("holdout", 5, "2", "flash")
    p2.mark(p2.items[0], "unsure")
    sv.commit(p2)
    assert sv.next_page() is None
    eff = gr.load_effective(led)
    e1, e2, e5 = eff[f"{st1}_dig6"], eff[f"{st2}_dig6"], eff[f"{st1}_dig5"]
    assert (e1["screen_verdict"], e1["label_verdict"], e1["final"]) == ("ok", "label_fixed", "9")
    assert (e2["screen_verdict"], e2["label_verdict"], e2["final"]) == ("ok", "label_ok", "1")
    assert e5["rejected"] == "label_unsure"
    # resume: nothing left to verify
    assert gr.Session(gr.load_queues([q])[0], led, task="verify").next_page() is None
    # 4. cross-mode undo is refused
    ss = gr.Session(gr.load_queues([q])[0], led, task="screen")
    ok, msg = ss.undo()
    assert not ok and "verify mode" in msg, msg
    ok, msg = gr.Session(gr.load_queues([q])[0], led, task="verify").undo()
    assert ok, msg
    assert gr.load_effective(led)[f"{st1}_dig5"]["label_verdict"] == ""


def t_screen_after_verify(tmp):
    """Fresh queue (no screen verdicts): screen skips unverified items, then shows
    verified ones under their verified label; label_unsure stays out."""
    q, (st1, st2, st3) = _vq(tmp)
    led = tmp / "led.csv"
    ss = gr.Session(gr.load_queues([q])[0], led, task="screen")
    assert ss.unverified_skipped == 4 and len(ss.items) == 11
    assert all(not i.needs_verify for i in ss.items)
    sv = gr.Session(gr.load_queues([q])[0], led, task="verify", cell_order="size")
    p = sv.next_page()                                     # dig6 of st1..st3, "is this a 1?"
    assert len(p.items) == 3
    byst = {i.stamp: i for i in p.items}
    p.mark(byst[st1], "9")                                 # fixed -> 9
    p.mark(byst[st3], "unsure")
    sv.commit(p)                                           # st2 -> label_ok 1
    ss = gr.Session(gr.load_queues([q])[0], led, task="screen")
    assert ss.unverified_skipped == 1                      # st1 dig5 still unverified
    ids = {i.item_id for i in ss.items}
    assert f"{st1}_dig5" not in ids and f"{st1}_dig6" in ids
    fixed = ss.by_id[f"{st1}_dig6"]
    assert fixed.proposed == "9" and fixed.orig_proposed == "1"
    assert ("holdout", 6, "9", "flash") in ss.cell_items      # grouped under its new label
    assert ss.done(ss.by_id[f"{st3}_dig6"])                   # label_unsure: skipped
    # screen until the dig6 '1' cell: only st2; mark drift -> whole frame st2
    while True:
        pg = ss.next_page()
        if pg.cell[1:3] == (6, "1"):
            break
        ss.commit(pg)
    assert [i.stamp for i in pg.items] == [st2]
    assert pg.resolved(pg.items[0]) == ("ok", "1")
    pg.mark(pg.items[0], "drift")
    pid = ss.commit(pg)
    assert pid.startswith("gs")
    eff = gr.load_effective(led)
    assert all(eff[f"{st2}_dig{p}"]["rejected"] == "drift" for p in range(2, 7))
    # the fixed crop, once screened ok, keeps its verified label
    while (pg := ss.next_page()) is not None:
        ss.commit(pg)
    eff = gr.load_effective(led)
    assert eff[f"{st1}_dig6"]["screen_verdict"] == "ok"
    assert eff[f"{st1}_dig6"]["final"] == "9" and eff[f"{st1}_dig6"]["screen_final"] == "9"
    assert eff[f"{st1}_dig2"]["final"] == "5"
    assert f"{st1}_dig5" not in eff                         # never verified -> never screened
    # verify mode no longer offers st2 (drift frame) -- st1 dig5 is the only one left
    sv = gr.Session(gr.load_queues([q])[0], led, task="verify")
    assert [i.item_id for i in sv.next_page().items] == [f"{st1}_dig5"]


def t_keys_and_badges(tmp):
    assert gr.action_allowed("verify", "unsure") and gr.action_allowed("verify", "drift")
    assert not gr.action_allowed("verify", "illegible")
    assert not gr.action_allowed("verify", "ok_derived")
    assert not gr.action_allowed("screen", "unsure")
    assert not gr.action_allowed("screen", "illegible") and gr.action_allowed("screen", "7")
    assert gr.action_allowed("classic", "illegible") and not gr.action_allowed("classic", "unsure")

    class Ev:
        def __init__(self, k):
            self.keysym = k
    assert gr.key_action(Ev("x")) == "unsure" and gr.key_action(Ev("question")) == "unsure"
    assert gr.key_action(Ev("n")) == "N" and gr.key_action(Ev("KP_End")) == "1"
    assert gr.badge_text("model_disagrees|drift?", "verify") == "!drift?"
    assert gr.badge_text("model_disagrees", "verify") == ""
    assert gr.badge_text("model_disagrees", "screen") == ""
    assert gr.badge_text("drift?", "screen") == "!drift?"
    assert gr.badge_text("model_disagrees", "classic") == "!model"
    assert gr.model_from_context("9002 says 10 @0.97; label 8") == "N"


def t_time_strip(tmp):
    rows = []
    for k in range(10):
        st = f"20260901-04{k * 4:02d}00"
        rows.append(dict(stamp=st, pos="dig6", path=f"/x/{k}.jpg", model_label=str(k % 10),
                         truth="?" if k == 5 else str(k % 10),
                         status="agree", screen="dash" if k == 3 else "reading",
                         reading_est=str(58000 + k)))
    der = gr.DeriveIndex(rows=rows)
    assert len(der) == 10 and len(der.reading[6]) == 9
    it = gr.QItem("q", "20260901-042000_dig6", "20260901-042000", 6, "/x/me.jpg", "5")
    s = gr.Session([it], tmp / "led.csv", task="verify", derive=der)
    strip = s.time_strip(it, 3)
    st = [e["stamp"][-6:] for e in strip]
    # k=5 is the item; k=3 is a dash screen -> skipped
    assert st == ["040400", "040800", "041600", "042000", "042400", "042800", "043200"], st
    assert [e["current"] for e in strip] == [False] * 3 + [True] + [False] * 3
    assert strip[3]["path"] == "/x/me.jpg"
    assert strip[3]["lines"][1].startswith("t ?") and "e 5" in strip[3]["lines"][1]
    assert strip[0]["lines"][0].startswith("04:04") and "58001" in strip[0]["lines"][0]
    assert "reading_est 58005" in s.frame_reading(it)
    # non-b3 fallback: same-directory crops of that position
    d = tmp / "leg"
    d.mkdir()
    for k in range(6):
        _img(d / f"{k}_main_dig6_20260803-0{k}0000.jpg", 80)
    it2 = gr.QItem("legacy", "20260803-030000_dig6", "20260803-030000", 6,
                   str(d / "3_main_dig6_20260803-030000.jpg"), "3", model_label="8")
    s2 = gr.Session([it2], tmp / "led2.csv", task="verify", derive=der)
    strip = s2.time_strip(it2, 3)
    assert [e["stamp"][-6:-4] for e in strip] == ["00", "01", "02", "03", "04", "05"]
    assert strip[3]["current"] and "m 8" in strip[3]["lines"][1]
    assert strip[0]["lines"][1] == "file 0"


def t_needs_verify_rule(tmp):
    import build_queues as bq
    nv = bq.needs_verify_b3
    assert nv("agree", "5", "5", False, "reading", 6) == 0
    assert nv("agree_soft", "8", "9", True, "reading", 6) == 1          # skip-inferred
    assert nv("ambiguous", "1", "1", False, "reading", 5) == 1
    assert nv("disagree", "3", "4", False, "reading", 4) == 1
    assert nv("disagree", "8", "9", False, "reading", 3) == 0          # series pins dig3
    assert nv("ambiguous", "8", "9", False, "reading", 3) == 1
    assert nv("uncertain", "N", "5", False, "reading", 2) == 1         # uncertain stays
    assert nv("disagree", "8", "0", False, "test0", 3) == 0            # 00000 pins it
    assert nv("disagree", "9", "8", False, "test8", 6) == 0
    assert nv("disagree", "5", "N", False, "dash", 2) == 0
    assert nv("dash_unverified", "8", "9", False, "dash", 6) == 1
    assert nv("dash_unverified", "0", "0", False, "dash", 5) == 1
    assert bq.needs_verify_legacy("9002 says 8 @0.97; label 5") == 1
    assert bq.needs_verify_legacy("derived-only in corpus") == 0


# ---------------------------------------------------------------- new: consistency

GS = "gs20261003-100000-aaaa-0001"
GS2 = "gs20261003-100500-aaaa-0002"
GV = "gv20261003-090000-bbbb-0001"
FH = "f20261003-080000-cccc-0001"
CA, CB, CC, CD = "20260901-030000", "20260901-120000", "20260901-060000", "20260901-031000"
CE, CF, CH, CG = "20260901-121000", "20260901-032000", "20260901-033000", "20260902-040000"


def _cfix(tmp: Path):
    """Ledger after verify + screen passes. Frames (dig2..dig6 labels):
      A night_fill flash  br 30   N 0 9 8 0      (dig5 later rejected: artifact)
      B day_fill   day    br 150  N 0 9 8 0      (dig5 label_unsure; dig4 screen-relabeled 7)
      C holdout    trans. br 90   N 0 9 8 0      (classic frame-mode page)
      D night_err  flash  br 20   N 0 9 8 6      (dig6 label_fixed 6 -> 5 in verify)
      E day_fill   day    (not in b3_derive) N 0 9 8 5   (dig6 label_ok 5 in verify)
      F night_fill flash  br 40   ... dig6 9
      H night_fill  b3 bucket 'transition'  br 50   ... dig6 8
      G night_fill drift frame (all five rejected)
    Queue file qa.csv carries A only (bucket flash, model context)."""
    raw = tmp / "raw" / "20260901"
    spec = {CA: ("night_fill", 30.0, "0"), CB: ("day_fill", 150.0, "0"),
            CC: ("holdout", 90.0, "0"), CD: ("night_err", 20.0, "6"),
            CE: ("day_fill", None, "5"), CF: ("night_fill", 40.0, "9"),
            CH: ("night_fill", 50.0, "8"), CG: ("night_fill", 60.0, "1")}
    paths, rows, drows = {}, [], []
    for st, (q, br, d6) in spec.items():
        labs = ("N", "0", "9", "8", d6)
        paths[st] = _frame_imgs(raw, st, labels=labs)
        for p, lab in zip(range(2, 7), labs):
            v, fin = ("drift", "") if st == CG else ("ok", lab)
            rows.append(dict(item_id=f"{st}_dig{p}", stamp=st, pos=p, src=paths[st][p],
                             queue=q, proposed=lab, final=fin, verdict=v,
                             page_id=FH if q == "holdout" else GS, reviewed_at="t"))
            if br is not None:
                drows.append(dict(stamp=st, pos=f"dig{p}", path=paths[st][p], model_label=lab,
                                  truth=lab, status="agree", screen="reading",
                                  reading_est="58000",
                                  bucket="transition" if st == CH else
                                  gr.bucket_for_hour(int(st[9:11])),
                                  frame_brightness=str(br)))

    def r(st, p, v, fin, page, prop):
        return dict(item_id=f"{st}_dig{p}", stamp=st, pos=p, src=paths[st][p],
                    queue=spec[st][0], proposed=prop, final=fin, verdict=v, page_id=page,
                    reviewed_at="t")
    rows += [r(CA, 5, "artifact", "", GS2, "8"), r(CB, 5, "label_unsure", "", GV, "8"),
             r(CD, 6, "label_fixed", "5", GV, "6"), r(CE, 6, "label_ok", "5", GV, "5"),
             r(CB, 4, "relabeled", "7", GS2, "9")]
    led = tmp / "led.csv"
    gr.LedgerWriter(led).append(rows)
    qa = _write_queue(tmp / "qa.csv", [
        dict(queue="night_fill", item_id=f"{CA}_dig{p}", stamp=CA, pos=str(p),
             src=paths[CA][p], proposed=lab, bucket="flash", brightness="31",
             context=f"reading_est 58000; model saw {lab}", needs_verify="0")
        for p, lab in zip(range(2, 7), ("N", "0", "9", "8", "0"))])
    return led, qa, gr.DeriveIndex(rows=drows), paths


def _csession(led, qa, der, positions=None, page_size=48):
    items, loaded, info = gr.consistency_items(led, [qa], der, positions)
    gr.fill_brightness(items)
    return gr.Session(items, led, task="consistency", page_size=page_size, derive=der,
                      extra_loaded_ids=loaded), info


def _cell(pos, lab):
    return ("consistency", pos, lab, "all")


def t_consistency_items_and_groups(tmp):
    led, qa, der, paths = _cfix(tmp)
    items, loaded, info = gr.consistency_items(led, [qa], der)
    by = {i.item_id: i for i in items}
    # 8 frames x 5 = 40; drift frame G (5) and A dig5 / B dig5 rejected -> 33
    assert len(items) == 33 and info["rejected_before"] == 7, (len(items), info)
    assert f"{CA}_dig5" not in by and f"{CB}_dig5" not in by
    assert not any(i.stamp == CG for i in items)
    assert {f"{CA}_dig5", f"{CG}_dig6"} <= loaded                   # undo scope
    assert by[f"{CC}_dig6"].queue == "holdout"                      # holdout included
    d6 = by[f"{CD}_dig6"]
    assert (d6.proposed, d6.prior, d6.orig_proposed) == ("5", "f", "6")
    assert by[f"{CE}_dig6"].prior == "v" and by[f"{CA}_dig6"].prior == ""
    assert by[f"{CB}_dig4"].proposed == "7"                         # screen relabel
    # buckets: queue file > b3_derive > hour rule; brightness from b3 frame_brightness
    assert by[f"{CA}_dig6"].bucket == "flash" and by[f"{CA}_dig6"].model_label == "0"
    assert by[f"{CH}_dig6"].bucket == "transition"                  # b3 (hour 03 says flash)
    assert by[f"{CE}_dig6"].bucket == "day" and by[f"{CE}_dig6"].brightness is None
    assert by[f"{CA}_dig6"].brightness == 30.0                      # b3, not queue (31)
    assert by[f"{CA}_dig6"].src == paths[CA][6]
    s, _ = _csession(led, qa, der)
    assert abs(s.by_id[f"{CE}_dig6"].brightness - 96) < 1.5          # computed crop luma
    assert [c[1:3] for c in s.cells] == [
        (6, "0"), (6, "8"), (6, "9"), (6, "5"), (5, "8"), (4, "9"), (4, "7"),
        (3, "0"), (2, "N")], s.cells
    assert s.overall_progress() == (0, 33)
    pg = s.next_page()
    assert pg.cell == _cell(6, "0") and pg.task == "consistency"
    assert [i.stamp for i in pg.items] == [CA, CC, CB]               # dark -> bright
    assert {i.bucket for i in pg.items} == {"flash", "transition", "day"}   # mixed
    assert pg.resolved(pg.items[0]) == ("label_ok", "0")
    pg5 = [s.cell_items[_cell(6, "5")]][0]
    assert [i.stamp for i in pg5] == [CD, CE]
    # --positions
    assert gr.parse_positions("dig6,dig5") == (6, 5) and gr.parse_positions("") == (6, 5, 4, 3, 2)
    s3, _ = _csession(led, qa, der, positions=gr.parse_positions("dig5,6"))
    assert {c[1] for c in s3.cells} == {5, 6} and s3.cells[0] == _cell(6, "0")
    todo = s3.todo_counts()
    assert todo[(6, "0")] == 3 and todo[(5, "8")] == 5 and sum(todo.values()) == 12, todo
    assert "dig6:" in gr.consistency_report(s3) and "dig4" not in gr.consistency_report(s3)
    assert gr.action_allowed("consistency", "unsure") and gr.action_allowed("consistency", "drift")
    assert not gr.action_allowed("consistency", "illegible")
    # balanced pages + header numbers (commits: keep last)
    s2, _ = _csession(led, qa, der, page_size=2)
    p = s2.next_page()
    assert len(p.items) == 2 and s2.group_page_numbers(p.cell) == (1, 2)
    s2.commit(p)
    p = s2.next_page()
    assert p.cell == _cell(6, "0") and len(p.items) == 1
    assert s2.group_page_numbers(p.cell) == (2, 2)


def t_consistency_writes(tmp):
    led, qa, der, paths = _cfix(tmp)
    s, _ = _csession(led, qa, der)
    p1 = s.next_page()                                    # dig6 '0': A, C, B
    byst = {i.stamp: i for i in p1.items}
    p1.mark(byst[CC], "8")                                # holdout crop: really an 8
    assert p1.marks == {byst[CC].item_id: ("label_fixed", "8")}
    pid1 = s.commit(p1)
    assert pid1.startswith("gc") and gr.page_task(pid1) == "consistency"
    rows = [r for r in gr.read_ledger_rows(led) if r["page_id"] == pid1]
    got = {r["stamp"]: (r["verdict"], r["proposed"], r["final"]) for r in rows}
    assert got == {CA: ("label_ok", "0", "0"), CB: ("label_ok", "0", "0"),
                   CC: ("label_fixed", "0", "8")}, got
    eff = gr.load_effective(led)
    ec = eff[f"{CC}_dig6"]
    # the fix updates the final and does NOT un-accept the item
    assert (ec["final"], ec["rejected"], ec["screen_verdict"], ec["label_verdict"]) == \
        ("8", "", "ok", "label_fixed")
    assert eff[f"{CA}_dig6"]["final"] == "0" and eff[f"{CA}_dig6"]["label_verdict"] == "label_ok"
    # the fixed crop moved to the '8' group, counted done, not re-shown
    assert s.by_id[f"{CC}_dig6"].proposed == "8"
    assert s.cell_progress(_cell(6, "8")) == (1, 2)
    p2 = s.next_page()
    assert p2.cell == _cell(6, "8") and [i.stamp for i in p2.items] == [CH]
    p2.mark(p2.items[0], "unsure")
    s.commit(p2)
    p3 = s.next_page()                                    # dig6 '9': F
    p3.mark(p3.items[0], "artifact")
    s.commit(p3)
    p4 = s.next_page()                                    # dig6 '5': D (fixed earlier), E
    assert p4.cell == _cell(6, "5")
    e_it = next(i for i in p4.items if i.stamp == CE)
    p4.mark(e_it, "drift")
    pid4 = s.commit(p4)
    eff = gr.load_effective(led)
    assert eff[f"{CH}_dig6"]["rejected"] == "label_unsure"
    assert eff[f"{CF}_dig6"]["rejected"] == "artifact" and eff[f"{CF}_dig6"]["final"] == ""
    assert eff[f"{CF}_dig6"]["screen_page_id"].startswith("gc")
    assert all(eff[f"{CE}_dig{p}"]["rejected"] == "drift" for p in range(2, 7))
    rows4 = [r for r in gr.read_ledger_rows(led) if r["page_id"] == pid4]
    assert len(rows4) == 2 + 4                              # frame-wide drift rows
    assert eff[f"{CD}_dig6"]["final"] == "5" and eff[f"{CD}_dig6"]["label_verdict"] == "label_ok"
    # E's other crops are now skipped (drift frame)
    assert s.done(s.by_id[f"{CE}_dig5"])
    assert s.stats["label_fixed"] == 1 and s.stats["drift"] == 5


def t_consistency_resume(tmp):
    led, qa, der, paths = _cfix(tmp)
    s, _ = _csession(led, qa, der)
    p1 = s.next_page()
    p1.mark(next(i for i in p1.items if i.stamp == CC), "8")
    s.commit(p1)
    p2 = s.next_page()
    p2.mark(p2.items[0], "unsure")                         # H -> label_unsure
    s.commit(p2)
    # fresh process
    s2, info = _csession(led, qa, der)
    assert info["rejected_in_consistency"] == 1
    assert s2.overall_progress() == (4, 33) and s2.preexisting == 4   # total unchanged
    c6 = s2.by_id[f"{CC}_dig6"]
    assert c6.proposed == "8" and c6.prior == "" and s2.done(c6)       # moved, done
    assert s2.cell_progress(_cell(6, "8")) == (2, 2)
    assert s2.cell_progress(_cell(6, "0")) == (2, 2)
    assert s2.next_page().cell == _cell(6, "9")
    # verify/screen passes don't count as consistency-done
    assert not s2.done(s2.by_id[f"{CE}_dig6"])                         # label_ok from gv
    # another mode's later label row makes the item due again
    gr.LedgerWriter(led).append([dict(item_id=f"{CA}_dig6", stamp=CA, pos=6, src="",
                                      queue="night_fill", proposed="0", final="0",
                                      verdict="label_ok", page_id="gv20261004-000000-x-0001",
                                      reviewed_at="t")])
    s3, _ = _csession(led, qa, der)
    assert not s3.done(s3.by_id[f"{CA}_dig6"]) and s3.by_id[f"{CA}_dig6"].prior == "v"


def t_consistency_undo(tmp):
    led, qa, der, paths = _cfix(tmp)
    s, _ = _csession(led, qa, der)
    ok, msg = s.undo()                                       # last page is a verify page
    assert not ok and "verify mode" in msg, msg
    p1 = s.next_page()
    byst = {i.stamp: i for i in p1.items}
    p1.mark(byst[CC], "8")
    p1.mark(byst[CA], "unsure")
    pid1 = s.commit(p1)
    # verify mode refuses to undo a consistency page
    ok, msg = gr.Session(gr.load_queues([qa])[0], led, task="verify").undo()
    assert not ok and "consistency mode" in msg, msg
    # a dig5-only session doesn't have these items loaded
    s5, _ = _csession(led, qa, der, positions=(5,))
    ok, msg = s5.undo()
    assert not ok and "not in" in msg, msg
    # fresh consistency session: A is rejected (unsure) but still loaded -> undo works
    s2, _ = _csession(led, qa, der)
    before = led.read_bytes()
    ok, msg = s2.undo()
    assert ok, msg
    assert led.read_bytes()[:len(before)] == before          # append-only tombstones
    eff = gr.load_effective(led)
    assert eff[f"{CC}_dig6"]["final"] == "0" and eff[f"{CC}_dig6"]["label_verdict"] == ""
    assert eff[f"{CA}_dig6"]["rejected"] == "" and eff[f"{CA}_dig6"]["final"] == "0"
    back = s2.next_page()
    assert back.restored_from == pid1 and back.cell == _cell(6, "0")
    assert {i.stamp for i in back.items} == {CA, CB, CC}
    assert s2.by_id[f"{CC}_dig6"].proposed == "0"              # moved back to its group
    assert back.marks == {f"{CC}_dig6": ("label_fixed", "8"), f"{CA}_dig6": ("label_unsure", "")}
    back.mark(s2.by_id[f"{CA}_dig6"], "clear")
    s2.commit(back)
    eff = gr.load_effective(led)
    assert eff[f"{CA}_dig6"]["label_verdict"] == "label_ok" and eff[f"{CC}_dig6"]["final"] == "8"
    # drift page undo across sessions (E's frame rejected by a consistency page)
    while (pg := s2.next_page()).cell != _cell(6, "5"):
        s2.commit(pg)
    pg.mark(next(i for i in pg.items if i.stamp == CE), "drift")
    s2.commit(pg)
    s3, _ = _csession(led, qa, der)
    assert s3.done(s3.by_id[f"{CE}_dig6"])
    ok, msg = s3.undo()
    assert ok, msg
    assert not any(e["rejected"] for k, e in gr.load_effective(led).items() if k.startswith(CE))
    assert s3.next_page().marks == {f"{CE}_dig6": ("drift", "")}


def t_audit_mode(tmp):
    """audit re-shows crops that already carry a label verdict, and resumes on audit pages."""
    led, _qa, der, paths = _cfix(tmp)
    qa = _write_queue(tmp / "audit.csv", [
        dict(queue="audit", item_id=f"{st}_dig6", stamp=st, pos="6", src=paths[st][6],
             proposed=lab, bucket="day", brightness="", context="OOF model says 6 @0.97",
             needs_verify="1") for st, lab in ((CE, "5"), (CD, "5"))])
    items = gr.load_queues([qa])[0]
    s = gr.Session(items, led, task="audit")
    ce = s.by_id[f"{CE}_dig6"]                     # label_ok from a verify page
    assert not s.done(ce), "an earlier verify verdict must not count as audited"
    while True:
        pg = s.next_page()
        if pg is None:
            break
        if ce in pg.items:
            pg.mark(ce, "6")
        s.commit(pg)
    eff = gr.load_effective(led)
    e = eff[f"{CE}_dig6"]
    assert e["label_verdict"] == "label_fixed" and e["label_final"] == "6", e
    assert gr.page_task(e["label_page_id"]) == "audit"
    assert eff[f"{CD}_dig6"]["label_verdict"] == "label_ok"   # confirmed its earlier fix
    s2 = gr.Session(gr.load_queues([qa])[0], led, task="audit")
    d, t = s2.overall_progress()
    assert d == t == 2, (d, t)


TESTS = [t_load_and_normalise, t_paging_and_sorting, t_commit_rows, t_resume,
         t_undo, t_ledger_semantics, t_frame_mode, t_sample_queue,
         t_frame_drift_marks, t_frame_drift_grid_and_offqueue, t_propagate_frame_drift,
         t_two_stage_semantics, t_verify_and_screen_modes, t_screen_after_verify,
         t_keys_and_badges, t_time_strip, t_needs_verify_rule,
         t_consistency_items_and_groups, t_consistency_writes, t_consistency_resume,
         t_consistency_undo, t_audit_mode]


def main() -> int:
    fails = 0
    for t in TESTS:
        with tempfile.TemporaryDirectory(prefix="grid_review_test_") as d:
            try:
                t(Path(d))
                print(f"PASS {t.__name__}")
            except Exception:
                fails += 1
                print(f"FAIL {t.__name__}")
                traceback.print_exc()
    print(f"{len(TESTS) - fails}/{len(TESTS)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

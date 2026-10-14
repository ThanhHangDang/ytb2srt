#!/usr/bin/env python3
"""
yt2srt - Tải phụ đề YouTube hàng loạt và lưu thành file .srt

Cài đặt:
    pip install -U yt-dlp
    pip install faster-whisper        # (tuỳ chọn) chỉ cần nếu dùng --whisper

Cách dùng:
    python yt2srt.py links.txt                      # file txt, mỗi dòng 1 link
    python yt2srt.py URL1 URL2 URL3                 # truyền link trực tiếp
    python yt2srt.py links.txt -l vi,en -o srt      # ưu tiên tiếng Việt, rồi tiếng Anh
    python yt2srt.py links.txt --whisper            # video không có phụ đề -> tự nhận dạng giọng nói
    python yt2srt.py links.txt --no-auto            # chỉ lấy phụ đề do người đăng tải lên

Thứ tự ưu tiên cho mỗi video:
    1. Phụ đề thủ công (do người đăng upload) theo thứ tự ngôn ngữ trong --lang
    2. Phụ đề tự động của YouTube (bỏ qua nếu --no-auto)
    3. Whisper (chỉ khi có --whisper)
"""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    sys.exit("Thiếu thư viện yt-dlp. Chạy: pip install -U yt-dlp")


# ----------------------------------------------------------------------------
# Tiện ích
# ----------------------------------------------------------------------------

def ms_to_srt_time(ms: int) -> str:
    ms = max(0, int(ms))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def safe_filename(name: str, max_len: int = 120) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", name).strip(" ._")
    return (name[:max_len].rstrip(" ._")) or "video"


def write_srt(cues, out_path: Path) -> int:
    """cues: list[(start_ms, end_ms, text)]"""
    lines = []
    for i, (start, end, text) in enumerate(cues, 1):
        lines += [str(i), f"{ms_to_srt_time(start)} --> {ms_to_srt_time(end)}", text, ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return len(cues)


def json3_to_cues(data: dict):
    """Chuyển định dạng json3 của YouTube sang danh sách cue cho SRT.
    Xử lý được cả phụ đề tự động (các dòng chạy chồng lên nhau)."""
    raw = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs)
        text = re.sub(r"[ \t]+", " ", text).strip()
        if not text or text == "\n":
            continue
        start = int(ev.get("tStartMs", 0))
        end = start + int(ev.get("dDurationMs", 0) or 2000)
        raw.append([start, end, text])

    raw.sort(key=lambda c: c[0])
    cues = []
    for i, (start, end, text) in enumerate(raw):
        # Cắt thời gian kết thúc để không đè lên câu tiếp theo
        if i + 1 < len(raw):
            end = min(end, raw[i + 1][0])
        if end <= start:
            end = start + 500
        # Gộp câu trùng lặp liền kề
        if cues and cues[-1][2] == text:
            cues[-1][1] = end
            continue
        cues.append([start, end, text])
    return [tuple(c) for c in cues]


# ----------------------------------------------------------------------------
# Lấy phụ đề YouTube
# ----------------------------------------------------------------------------

def pick_tracks(info: dict, langs, allow_auto: bool):
    """Trả về danh sách (lang, kind, url_json3) theo thứ tự ưu tiên.
    Phụ đề gốc của video (vd 'ja-orig') luôn được thêm cuối cùng làm dự phòng,
    vì YouTube hay chặn (HTTP 429) các bản dịch tự động."""
    sources = [("manual", info.get("subtitles") or {})]
    if allow_auto:
        sources.append(("auto", info.get("automatic_captions") or {}))

    orig_lang = (info.get("language") or "").split("-")[0]
    wanted = list(langs) + ([orig_lang] if orig_lang and orig_lang not in langs else [])

    result, seen = [], set()
    for kind, tracks in sources:
        for want in wanted:
            # Khớp chính xác trước, rồi khớp tiền tố (vd 'en' khớp 'en-US', 'en-orig')
            candidates = [k for k in tracks if k == want] + \
                         sorted(k for k in tracks if k.split("-")[0] == want and k != want)
            for lang in candidates:
                for fmt in tracks[lang]:
                    if fmt.get("ext") == "json3" and fmt.get("url") and fmt["url"] not in seen:
                        seen.add(fmt["url"])
                        result.append((lang, kind, fmt["url"]))
                        break
    return result


def fetch_subtitle(ydl, info, langs, allow_auto):
    last_err = None
    for lang, kind, url in pick_tracks(info, langs, allow_auto):
        try:
            data = json.loads(ydl.urlopen(url).read().decode("utf-8"))
        except Exception as e:
            last_err = e
            print(f"    Không tải được phụ đề {lang} ({e}), thử bản khác...")
            continue
        cues = json3_to_cues(data)
        if cues:
            return lang, kind, cues
    if last_err:
        raise last_err
    return None


# ----------------------------------------------------------------------------
# Whisper (tuỳ chọn)
# ----------------------------------------------------------------------------

_whisper_model = None


def transcribe_whisper(url, model_size, lang, device):
    global _whisper_model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("Thiếu faster-whisper. Chạy: pip install faster-whisper")

    if _whisper_model is None:
        compute = "float16" if device == "cuda" else "int8"
        print(f"    Đang tải model Whisper '{model_size}' ({device})...")
        _whisper_model = WhisperModel(model_size, device=device, compute_type=compute)

    with tempfile.TemporaryDirectory() as tmp:
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(Path(tmp) / "audio.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        audio = next(Path(tmp).glob("audio.*"))
        print("    Đang nhận dạng giọng nói (có thể mất vài phút)...")
        segments, info = _whisper_model.transcribe(str(audio), language=lang, vad_filter=True)
        cues = [(int(s.start * 1000), int(s.end * 1000), s.text.strip())
                for s in segments if s.text.strip()]
    return info.language, cues


# ----------------------------------------------------------------------------
# Xử lý chính
# ----------------------------------------------------------------------------

def iter_videos(ydl, url):
    """Mở rộng playlist/kênh thành từng video."""
    info = ydl.extract_info(url, download=False)
    if info.get("_type") in ("playlist", "multi_video") or "entries" in info:
        for entry in info.get("entries") or []:
            if entry:
                yield entry
    else:
        yield info


def process(urls, out_dir: Path, langs, allow_auto, use_whisper,
            whisper_model, device, overwrite):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True,
                "ignoreerrors": True}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for n, url in enumerate(urls, 1):
            print(f"[{n}/{len(urls)}] {url}")
            try:
                videos = list(iter_videos(ydl, url))
            except Exception as e:
                print(f"    LỖI: {e}")
                results.append((url, None, f"lỗi: {e}"))
                continue

            for info in videos:
                vid = info.get("id", "unknown")
                title = info.get("title", vid)
                video_url = info.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}"
                out_path = out_dir / f"{safe_filename(title)} [{vid}].srt"

                if out_path.exists() and not overwrite:
                    print(f"    Đã có, bỏ qua: {out_path.name}")
                    results.append((video_url, out_path, "đã có"))
                    continue

                try:
                    sub = fetch_subtitle(ydl, info, langs, allow_auto)
                    if sub:
                        lang, kind, cues = sub
                        count = write_srt(cues, out_path)
                        label = "thủ công" if kind == "manual" else "tự động"
                        print(f"    OK ({label}, {lang}, {count} dòng) -> {out_path.name}")
                        results.append((video_url, out_path, f"{label}/{lang}"))
                    elif use_whisper:
                        print("    Không có phụ đề, chuyển sang Whisper...")
                        w_lang = langs[0] if langs else None
                        lang, cues = transcribe_whisper(video_url, whisper_model, w_lang, device)
                        count = write_srt(cues, out_path)
                        print(f"    OK (whisper, {lang}, {count} dòng) -> {out_path.name}")
                        results.append((video_url, out_path, f"whisper/{lang}"))
                    else:
                        print("    Không tìm thấy phụ đề (thử thêm --whisper)")
                        results.append((video_url, None, "không có phụ đề"))
                except Exception as e:
                    print(f"    LỖI: {e}")
                    results.append((video_url, None, f"lỗi: {e}"))
    return results


def read_inputs(items):
    urls = []
    for item in items:
        p = Path(item)
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
        else:
            urls.append(item)
    # Bỏ trùng, giữ thứ tự
    return list(dict.fromkeys(urls))


def print_summary(results, out_dir: Path):
    ok = [r for r in results if r[1]]
    print("\n" + "=" * 60)
    print(f"Xong: {len(ok)}/{len(results)} video có file .srt -> {out_dir.resolve()}")
    for url, path, note in results:
        mark = "✔" if path else "✘"
        print(f"  {mark} {note:<18} {path.name if path else url}")


# ----------------------------------------------------------------------------
# Giao diện (chạy khi không truyền tham số)
# ----------------------------------------------------------------------------

def run_gui():
    import queue
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    lang_options = {"Tiếng Nhật (ja)": "ja", "Tiếng Việt (vi)": "vi", "Tiếng Anh (en)": "en"}

    log_queue = queue.Queue()

    class QueueWriter:
        """Chuyển print() từ luồng xử lý sang ô log của giao diện."""
        def write(self, text):
            log_queue.put(text)

        def flush(self):
            pass

    root = tk.Tk()
    root.title("yt2srt - Tải phụ đề YouTube - Pizon16")
    root.geometry("760x500")
    root.minsize(560, 360)

    links_var = tk.StringVar(value=str(Path("links.txt").resolve()) if Path("links.txt").is_file() else "")
    out_var = tk.StringVar(value=str(Path("srt_output").resolve()))
    lang_var = tk.StringVar(value="Tiếng Nhật (ja)")
    overwrite_var = tk.BooleanVar(value=False)

    form = tk.Frame(root, padx=10, pady=10)
    form.pack(fill="x")
    form.columnconfigure(1, weight=1)

    def browse_links():
        path = filedialog.askopenfilename(title="Chọn file danh sách link",
                                          filetypes=[("Text", "*.txt"), ("Tất cả", "*.*")])
        if path:
            links_var.set(path)

    def browse_out():
        path = filedialog.askdirectory(title="Chọn thư mục lưu .srt")
        if path:
            out_var.set(path)

    tk.Label(form, text="File link (.txt):").grid(row=0, column=0, sticky="w", pady=3)
    tk.Entry(form, textvariable=links_var).grid(row=0, column=1, sticky="ew", padx=5)
    tk.Button(form, text="Chọn file...", command=browse_links).grid(row=0, column=2, sticky="ew")

    tk.Label(form, text="Thư mục lưu:").grid(row=1, column=0, sticky="w", pady=3)
    tk.Entry(form, textvariable=out_var).grid(row=1, column=1, sticky="ew", padx=5)
    tk.Button(form, text="Chọn thư mục...", command=browse_out).grid(row=1, column=2, sticky="ew")

    tk.Label(form, text="Ngôn ngữ ưu tiên:").grid(row=2, column=0, sticky="w", pady=3)
    ttk.Combobox(form, textvariable=lang_var, values=list(lang_options), state="readonly",
                 width=18).grid(row=2, column=1, sticky="w", padx=5)
    tk.Checkbutton(form, text="Ghi đè file đã có", variable=overwrite_var).grid(row=2, column=2, sticky="w")

    start_btn = tk.Button(form, text="Lấy sub", width=16, bg="#2563eb", fg="white",
                          font=("Segoe UI", 10, "bold"))
    start_btn.grid(row=3, column=0, columnspan=3, pady=(10, 0))

    log = scrolledtext.ScrolledText(root, font=("Consolas", 9), state="disabled")
    log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def worker(urls, out_dir, langs, overwrite):
        # Bản .exe không có console (stdout/stderr = None) -> đưa cả hai vào ô log
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = QueueWriter()
        try:
            results = process(urls, out_dir, langs, True, False, "small", "cpu", overwrite)
            print_summary(results, out_dir)
        except Exception as e:
            print(f"LỖI: {e}")
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            log_queue.put(None)  # báo đã xong

    def start():
        links_path = Path(links_var.get().strip().strip('"'))
        if not links_path.is_file():
            messagebox.showerror("Lỗi", "Hãy chọn file .txt chứa danh sách link.")
            return
        urls = read_inputs([str(links_path)])
        if not urls:
            messagebox.showerror("Lỗi", "File không có link nào.")
            return
        langs = [lang_options[lang_var.get()]]
        out_dir = Path(out_var.get().strip().strip('"') or "srt_output")

        log.configure(state="normal")
        log.delete("1.0", "end")
        log.configure(state="disabled")
        start_btn.configure(state="disabled", text="Đang lấy sub...")
        threading.Thread(target=worker, args=(urls, out_dir, langs, overwrite_var.get()),
                         daemon=True).start()

    def poll_log():
        try:
            while True:
                text = log_queue.get_nowait()
                if text is None:
                    start_btn.configure(state="normal", text="Lấy sub")
                    continue
                log.configure(state="normal")
                log.insert("end", text)
                log.see("end")
                log.configure(state="disabled")
        except queue.Empty:
            pass
        root.after(100, poll_log)

    start_btn.configure(command=start)
    poll_log()
    root.mainloop()


def main():
    # Tránh UnicodeEncodeError khi in tiếng Việt ra console/pipe dùng mã cp1252
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) == 1:
        run_gui()
        return

    ap = argparse.ArgumentParser(description="Tải phụ đề YouTube hàng loạt thành file .srt")
    ap.add_argument("inputs", nargs="+", help="Link YouTube hoặc file .txt chứa danh sách link")
    ap.add_argument("-o", "--out", default="srt_output", help="Thư mục lưu .srt (mặc định: srt_output)")
    ap.add_argument("-l", "--lang", default="vi,en",
                    help="Ngôn ngữ ưu tiên, cách nhau bởi dấu phẩy (mặc định: vi,en)")
    ap.add_argument("--no-auto", action="store_true", help="Không dùng phụ đề tự động của YouTube")
    ap.add_argument("--whisper", action="store_true",
                    help="Dùng Whisper cho video không có phụ đề (cần faster-whisper)")
    ap.add_argument("--whisper-model", default="small",
                    help="tiny / base / small / medium / large-v3 (mặc định: small)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Thiết bị chạy Whisper")
    ap.add_argument("--overwrite", action="store_true", help="Ghi đè file .srt đã có")
    args = ap.parse_args()

    urls = read_inputs(args.inputs)
    if not urls:
        sys.exit("Không có link nào.")
    langs = [x.strip() for x in args.lang.split(",") if x.strip()]

    results = process(urls, Path(args.out), langs, not args.no_auto, args.whisper,
                      args.whisper_model, args.device, args.overwrite)

    print_summary(results, Path(args.out))


if __name__ == "__main__":
    main()
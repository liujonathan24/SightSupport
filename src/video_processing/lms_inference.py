import asyncio
import base64
import time
import statistics as _stats
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Deque, List, Sequence, Optional

from PIL import Image, ImageDraw, ImageFont
from openai import AsyncOpenAI
from src.video_processing.meeting_window import Window

# -------------------- config --------------------
MODEL_ID = "Qwen2.5-VL-7B-Instruct"    # update to exact id from /v1/models if needed
BASE_URL = "http://localhost:1234/v1"  # LM Studio OpenAI-compatible server
API_KEY  = "lm-studio"                 # any non-empty string

CAPTURE_PERIOD_S = 0.5                # 500 ms capture cadence
QUEUE_MAXSIZE = 64                     # producer backpressure
BATCH_SIZE = 16                        # frames per storyboard
STORY_SIZE = 256                       # 256x256 canvas
TILE_SIZE = STORY_SIZE // 4            # 64x64 tiles for a 4x4 grid
ZOOM = 1.0                             # set 2.0 for 2x zoom before tiling
ANNOTATE = False                       # draw small panel indices

PROMPT = (
    """"First describe whether the person in the picture shows a positive or negative expression for most of the subimages in the grid. Reason through your analysis and output POSITIVE or NEGATIVE in your last sentence."""
)

# -------------------- helpers on PIL images --------------------
def center_crop_square(img: Image.Image, zoom: float = 1.0) -> Image.Image:
    """
    Center-crop to a square. If zoom > 1, zooms in by cropping a smaller
    centered square (side/zoom) and resizing back to the original square size.
    """
    w, h = img.size
    side = min(w, h)

    left = (w - side) // 2
    top  = (h - side) // 2
    sq = img.crop((left, top, left + side, top + side))

    if zoom and zoom > 1.0:
        inner = max(1, int(round(side / zoom)))
        cx = side // 2
        cy = side // 2
        half = inner // 2
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(side, x0 + inner)
        y1 = min(side, y0 + inner)
        sq = sq.crop((x0, y0, x1, y1)).resize((side, side), Image.LANCZOS)

    return sq

def pick_16_evenly(images: Sequence[Image.Image]) -> List[Image.Image]:
    if not images:
        raise ValueError("No images provided")
    n = len(images)
    if n == 16:
        return list(images)
    if n < 16:
        return list(images) + [images[-1]] * (16 - n)
    idxs = [round(i * (n - 1) / 15) for i in range(16)]
    return [images[i] for i in idxs]

def make_storyboard_16(
    images: Sequence[Image.Image],
    out_path: str = "story_256.png",
    annotate: bool = True,
    tile_size: int = TILE_SIZE,
    cols_rows: int = 4,
    zoom: float = ZOOM,
) -> str:
    """
    Build a cols_rows x cols_rows grid (default 4x4) from 16 frames into a STORY_SIZE PNG.
    Accepts a list of PIL Images directly.
    """
    imgs16 = pick_16_evenly(images)

    board = Image.new("RGB", (cols_rows * tile_size, cols_rows * tile_size), (245, 245, 245))
    draw = ImageDraw.Draw(board)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, img in enumerate(imgs16):
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        img = center_crop_square(img, zoom=zoom).resize((tile_size, tile_size), Image.LANCZOS)

        r, c = divmod(i, cols_rows)
        x, y = c * tile_size, r * tile_size
        board.paste(img, (x, y))

        if annotate:
            lab = str(i)
            try:
                bbox = draw.textbbox((0, 0), lab, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                tw, th = draw.textsize(lab, font=font)
            draw.rectangle([x + 4, y + 4, x + 4 + tw + 6, y + 4 + th + 6], fill=(0, 0, 0))
            draw.text((x + 7, y + 7), lab, fill=(255, 255, 255), font=font)

    board.save(out_path)
    return out_path

def to_data_url(fp: str) -> str:
    mime = "image/png" if fp.lower().endswith(".png") else "image/jpeg"
    b64 = base64.b64encode(Path(fp).read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# -------------------- telemetry wrappers --------------------
@dataclass
class Frame:
    img: Image.Image
    t_capture: float  # seconds since epoch

class Stats:
    def __init__(self):
        self.captured = 0
        self.dropped = 0
        self.inferred = 0
        self.max_q = 0
        self.infer_ms_ewma: Optional[float] = None
        self.lag_ms_ewma: Optional[float] = None

    @staticmethod
    def ewma(old: Optional[float], new: float, alpha: float = 0.2) -> float:
        return new if old is None else (alpha * new + (1.0 - alpha) * old)

# -------------------- async pipeline --------------------
class AsyncInference:
    def __init__(self, app: str = "Zoom", model: str = MODEL_ID, q_maxsize: int = QUEUE_MAXSIZE):
        self.window = Window(app)
        self.queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=q_maxsize)
        self.buffer: Deque[Frame] = deque(maxlen=64)  # rolling store
        self.client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.model = model
        self.stats = Stats()

    async def capture_frames(self, period_s: float = CAPTURE_PERIOD_S):
        """Producer: capture every ~period_s without blocking."""
        while True:
            img = await asyncio.to_thread(self.window.get_image)
            if isinstance(img, Image.Image):
                f = Frame(img=img, t_capture=time.time())
                try:
                    self.queue.put_nowait(f)
                    self.stats.captured += 1
                    self.stats.max_q = max(self.stats.max_q, self.queue.qsize())
                except asyncio.QueueFull:
                    # drop oldest to keep recency
                    try:
                        _ = self.queue.get_nowait()
                        self.queue.task_done()
                    except asyncio.QueueEmpty:
                        pass
                    self.stats.dropped += 1
                    await self.queue.put(f)
            await asyncio.sleep(period_s)

    async def analyze_batches(self, prompt: str = PROMPT):
        """Consumer: when ≥16 frames available, build storyboard, call model, print result."""
        while True:
            f = await self.queue.get()
            self.buffer.append(f)
            self.queue.task_done()

            if len(self.buffer) >= BATCH_SIZE:
                # take the most recent BATCH_SIZE frames (deque has no slicing -> convert to list)
                recent = list(self.buffer)[-BATCH_SIZE:]
                frames = [x.img for x in recent]

                story = make_storyboard_16(
                    frames,
                    out_path="story_256.png",
                    annotate=ANNOTATE,
                    tile_size=TILE_SIZE,
                    cols_rows=4,
                    zoom=ZOOM,
                )
                img_url = to_data_url(story)

                t0 = time.time()
                t_cap_avg = _stats.fmean(x.t_capture for x in recent)

                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": img_url}},
                            ],
                        }],
                        temperature=0.1,
                    )
                    answer = resp.choices[0].message.content.strip()
                    t1 = time.time()

                    infer_ms = (t1 - t0) * 1000.0
                    lag_ms   = (t1 - t_cap_avg) * 1000.0

                    self.stats.inferred += 1
                    self.stats.infer_ms_ewma = Stats.ewma(self.stats.infer_ms_ewma, infer_ms)
                    self.stats.lag_ms_ewma   = Stats.ewma(self.stats.lag_ms_ewma,   lag_ms)

                    # ---- The only line you need for the UI handoff ----
                    print("::RESULT:: " + answer, flush=True)
                    # ----------------------------------------------------

                    # optional human-readable log
                    print("[RESULT]", answer, flush=True)

                    # disjoint batches; for sliding window remove next line
                    self.buffer.clear()

                except Exception as e:
                    print("[ERROR]", e, flush=True)

    async def report_stats(self, interval_s: float = 1.0):
        last_captured = last_inferred = last_dropped = 0
        while True:
            await asyncio.sleep(interval_s)
            c, i, d = self.stats.captured, self.stats.inferred, self.stats.dropped
            dc, di, dd = c - last_captured, i - last_inferred, d - last_dropped
            last_captured, last_inferred, last_dropped = c, i, d

            q = self.queue.qsize()
            # if self.stats.infer_ms_ewma is not None:
            #     print(
            #         f"[STATS] q={q} (max {self.stats.max_q}) | "
            #         f"cap {dc}/s, infer {di}/s, drops {dd}/s | "
            #         f"EWMA infer={self.stats.infer_ms_ewma:.0f} ms, "
            #         f"EWMA lag={self.stats.lag_ms_ewma:.0f} ms",
            #         flush=True
            #     )
            # else:
            #     print(
            #         f"[STATS] q={q} (max {self.stats.max_q}) | "
            #         f"cap {dc}/s, infer {di}/s, drops {dd}/s",
            #         flush=True
            #     )

            # simple warnings
            # if self.queue.maxsize and q > 0.8 * self.queue.maxsize:
            #     print("[WARN] Queue near capacity — inference is not keeping up.", flush=True)
            # if dd > 0:
            #     print(f"[WARN] Dropping frames ({dd}/s). Consider slower capture or smaller images.", flush=True)
            # if self.stats.lag_ms_ewma and self.stats.lag_ms_ewma > 1500:
            #     print("[WARN] High end-to-end lag (>1.5s).", flush=True)

    async def run(self):
        producer = asyncio.create_task(self.capture_frames(period_s=CAPTURE_PERIOD_S))
        consumer = asyncio.create_task(self.analyze_batches(prompt=PROMPT))
        reporter = asyncio.create_task(self.report_stats(interval_s=1.0))
        await asyncio.gather(producer, consumer, reporter)

# -------------------- entry point --------------------
if __name__ == "__main__":
    import traceback, sys
    print("::READY:: lms_inference booted", flush=True)
    try:
        asyncio.run(AsyncInference(app="Zoom").run())
    except KeyboardInterrupt:
        print("::EXIT:: keyboard interrupt", flush=True)
    except Exception as e:
        print("[FATAL] Unhandled exception:", repr(e), flush=True)
        traceback.print_exc()
        sys.exit(1)

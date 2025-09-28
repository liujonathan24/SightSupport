import asyncio, base64
from pathlib import Path
from collections import deque
from typing import Deque, List, Sequence
from PIL import Image, ImageDraw, ImageFont
from openai import AsyncOpenAI
from src.video_processing.meeting_window import Window

# ---------- helpers on PIL images ----------
def center_crop_square(img: Image.Image, zoom: float = 1) -> Image.Image:
    """
    Center-crop to a square. If zoom > 1, zooms in by cropping a smaller
    centered square (side/zoom) and resizing back to the original square size.

    Example: zoom=2.0 -> 2x zoom.
    """
    w, h = img.size
    side = min(w, h)

    # 1) center-crop to the largest square
    left = (w - side) // 2
    top  = (h - side) // 2
    sq = img.crop((left, top, left + side, top + side))

    # 2) optional zoom-in: crop a smaller centered square and upscale back
    if zoom and zoom > 1.0:
        inner = max(1, int(round(side / zoom)))
        cx = side // 2
        cy = side // 2
        half = inner // 2
        # clamp to bounds just in case of odd sizes
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
    if n == 16: return list(images)
    if n < 16:  return list(images) + [images[-1]] * (16 - n)
    idxs = [round(i * (n - 1) / 15) for i in range(16)]
    return [images[i] for i in idxs]

def make_storyboard_16(images: Sequence[Image.Image], out_path="story_256.png", annotate=True) -> str:
    imgs16 = pick_16_evenly(images)
    TILE = 64; COLS = ROWS = 4
    board = Image.new("RGB", (COLS*TILE, ROWS*TILE), (245,245,245))
    draw = ImageDraw.Draw(board); 
    try: font = ImageFont.load_default()
    except Exception: font = None
    for i, img in enumerate(imgs16):
        if img.mode not in ("RGB","RGBA"): img = img.convert("RGB")
        img = center_crop_square(img).resize((TILE, TILE), Image.LANCZOS)
        r, c = divmod(i, COLS); x, y = c*TILE, r*TILE
        board.paste(img, (x, y))
        if annotate:
            lab = str(i)
            try: tw, th = (lambda b: (b[2]-b[0], b[3]-b[1]))(draw.textbbox((0,0), lab, font=font))
            except Exception: tw, th = draw.textsize(lab, font=font)
            draw.rectangle([x+4, y+4, x+4+tw+6, y+4+th+6], fill=(0,0,0))
            draw.text((x+7, y+7), lab, fill=(255,255,255), font=font)
    board.save(out_path)
    return out_path

def to_data_url(fp: str) -> str:
    mime = "image/png" if fp.lower().endswith(".png") else "image/jpeg"
    b64 = base64.b64encode(Path(fp).read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# ---------- async pipeline ----------
class AsyncInference:
    def __init__(self, app="Zoom", model="Qwen2.5-VL-7B-Instruct", q_maxsize=64):
        self.window = Window(app)
        self.queue: asyncio.Queue[Image.Image] = asyncio.Queue(maxsize=q_maxsize)
        self.buffer: Deque[Image.Image] = deque(maxlen=64)  # rolling store; ≥16 for a batch
        self.client = AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.model = model

    async def capture_frames(self, period_s=0.1):
        """Producer: capture every ~100 ms without blocking."""
        while True:
            # Window.get_image() is sync -> offload to thread to avoid blocking event loop
            img = await asyncio.to_thread(self.window.get_image)
            if isinstance(img, Image.Image):
                try:
                    self.queue.put_nowait(img)
                except asyncio.QueueFull:
                    # Drop oldest by pulling one and pushing the latest (keeps recency)
                    _ = self.queue.get_nowait()
                    self.queue.task_done()
                    await self.queue.put(img)
            await asyncio.sleep(period_s)

    async def analyze_batches(self, prompt=(
        "Across the panels in chronological order, is the person shaking their head? Respond only True or False."
    )):
        """Consumer: build 16-frame storyboard whenever ≥16 frames are available, send async request."""
        while True:
            img = await self.queue.get()
            self.buffer.append(img)
            self.queue.task_done()

            if len(self.buffer) >= 16:
                # Take the last 16 frames (or evenly sample more)
                frames = list(self.buffer)
                story = make_storyboard_16(frames, out_path="story_256.png", annotate=True)
                img_url = to_data_url(story)

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
                    print("[RESULT]", resp.choices[0].message.content.strip())
                    # Optional: clear buffer to make strictly non-overlapping windows
                    self.buffer.clear()
                except Exception as e:
                    print("[ERROR]", e)

    async def run(self):
        producer = asyncio.create_task(self.capture_frames(period_s=0.05))
        consumer = asyncio.create_task(self.analyze_batches())
        await asyncio.gather(producer, consumer)

# ---------- entry point ----------
if __name__ == "__main__":
    try:
        asyncio.run(AsyncInference(app="Zoom").run())
    except KeyboardInterrupt:
        print("\nStopped.")

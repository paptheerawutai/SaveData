"""
catcam — กล้องดูแมวน้อย ผ่านเว็บมือถือ + คุยสองทาง (WebRTC)
-------------------------------------------------------------
- วิดีโอสด: ดึงภาพจากกล้อง (USB webcam / Pi Camera) ด้วย OpenCV
- ฟังเสียงแมว: ไมค์ฝั่งกล้อง -> ส่งเสียงไปมือถือ
- คุยกับแมว: ไมค์มือถือ -> เสียงออกลำโพงฝั่งกล้อง (กดปุ่ม Push-to-talk)
- เข้าผ่านเว็บได้ทุกอุปกรณ์ในวง LAN เดียวกัน (รีโมตข้ามเน็ตดู README)

รันด้วย:  python3 server.py
แล้วเปิดมือถือไปที่  http://<ip-เครื่องที่รัน>:8080
"""

import argparse
import asyncio
import datetime
import ipaddress
import json
import logging
import os
import socket
import ssl
import threading
import time
from fractions import Fraction

import cv2
import numpy as np
from aiohttp import web
from av import AudioFrame, VideoFrame
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    MediaStreamTrack,
)
from aiortc.rtcrtpsender import RTCRtpSender

try:
    import av  # for AudioResampler
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception as _audio_err:  # pragma: no cover
    AUDIO_AVAILABLE = False
    _AUDIO_IMPORT_ERROR = _audio_err

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("catcam")

ROOT = os.path.dirname(__file__)
pcs = set()  # เก็บ peer connection ที่เปิดอยู่

# ---- ค่าเสียงมาตรฐาน (WebRTC ใช้ 48kHz) ----
SAMPLE_RATE = 48000
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 960 ตัวอย่าง/เฟรม


# ==========================================================================
#  วิดีโอ: ดึงภาพจากกล้องด้วย OpenCV
# ==========================================================================
VIDEO_CLOCK = 90000  # นาฬิกามาตรฐานของวิดีโอใน WebRTC (90kHz)


# ==========================================================================
#  วิดีโอ: ดึงภาพจากกล้องด้วย OpenCV
#  - อ่านภาพในเธรดแยก เก็บเฉพาะ "เฟรมล่าสุด" -> ไม่มีดีเลย์สะสม
#  - ขอภาพแบบ MJPG จากกล้อง -> ลดแบนด์วิดท์ USB และภาระ CPU
#  - คุมเฟรมเรตให้ตรงค่าที่ตั้ง -> ประหยัด CPU ตอน fps ต่ำ (เหมาะกับ Pi)
# ==========================================================================
class CameraVideoTrack(VideoStreamTrack):
    """อ่านเฟรมจากกล้องแล้วส่งออกเป็น track วิดีโอ"""

    def __init__(self, device, width, height, fps, flip, mjpg=True):
        super().__init__()
        self.flip = flip
        self.fps = max(1, fps)
        self.width = width
        self.height = height

        # บน Linux/Pi ใช้ V4L2; ระบบอื่นปล่อยให้ OpenCV เลือกเอง
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(device, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(device)  # ลองแบบ default อีกรอบ
        if not self.cap.isOpened():
            raise RuntimeError(
                f"เปิดกล้อง index {device} ไม่ได้ — ลองเปลี่ยน --camera หรือเช็คสายกล้อง"
            )

        # ขอให้กล้องส่งภาพแบบ MJPG (USB webcam ส่วนใหญ่รองรับ) — ได้ HD/fps สูงโดยไม่กิน CPU มาก
        if mjpg:
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ลดดีเลย์
        except Exception:
            pass

        # อ่านขนาดจริงที่กล้องให้มา (บางตัวไม่ได้ตามที่ขอ)
        real_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        real_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        self.width, self.height = real_w, real_h
        log.info("เปิดกล้องสำเร็จ (index %s, %dx%d @ %dfps)", device, real_w, real_h, self.fps)

        # เธรดอ่านภาพ: เก็บเฉพาะเฟรมล่าสุด
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        # ตัวคุมจังหวะส่งเฟรม
        self._ts = 0
        self._next_t = None

    def _read_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            if self.flip:
                frame = cv2.flip(frame, -1)  # หมุน 180° (กล้องกลับหัว)
            with self._lock:
                self._frame = frame

    async def recv(self):
        # คุมจังหวะให้ตรงกับ fps ที่ตั้งไว้ (ไม่ encode ถี่เกินจำเป็น)
        now = time.monotonic()
        if self._next_t is None:
            self._next_t = now
        delay = self._next_t - now
        if delay > 0:
            await asyncio.sleep(delay)
        self._next_t += 1.0 / self.fps

        with self._lock:
            frame = self._frame
        if frame is None:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self._ts
        video_frame.time_base = Fraction(1, VIDEO_CLOCK)
        self._ts += int(VIDEO_CLOCK / self.fps)
        return video_frame

    def stop(self):
        super().stop()
        self._running = False
        try:
            self._reader.join(timeout=1.0)
        except Exception:
            pass
        if self.cap and self.cap.isOpened():
            self.cap.release()


# ==========================================================================
#  เสียงขาออก: ไมค์ฝั่งกล้อง -> ส่งไปมือถือ (ฟังเสียงแมว)
# ==========================================================================
class MicAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._timestamp = 0
        self._loop = asyncio.get_event_loop()

        def _push(data):
            try:
                self._queue.put_nowait(data)
            except asyncio.QueueFull:
                pass  # คิวเต็ม ทิ้งเฟรมได้ (กันดีเลย์สะสม)

        def callback(indata, frames, time_info, status):
            if status:
                log.debug("mic status: %s", status)
            data = bytes(indata)  # int16 mono
            self._loop.call_soon_threadsafe(_push, data)

        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=SAMPLES_PER_FRAME,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()
        log.info("เปิดไมค์ฝั่งกล้องแล้ว (ส่งเสียงไปมือถือ)")

    async def recv(self):
        data = await self._queue.get()
        samples = np.frombuffer(data, dtype=np.int16)
        frame = AudioFrame(format="s16", layout="mono", samples=len(samples))
        frame.planes[0].update(samples.tobytes())
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, SAMPLE_RATE)
        self._timestamp += len(samples)
        return frame

    def stop(self):
        super().stop()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


# ==========================================================================
#  เสียงขาเข้า: เสียงจากมือถือ -> ลำโพงฝั่งกล้อง (คุยกับแมว)
# ==========================================================================
async def play_incoming_audio(track):
    """รับ track เสียงจากมือถือแล้วเล่นออกลำโพงเครื่องที่รันเซิร์ฟเวอร์"""
    resampler = av.AudioResampler(format="s16", layout="stereo", rate=SAMPLE_RATE)
    out = sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, dtype="int16")
    out.start()
    log.info("พร้อมเล่นเสียงจากมือถือออกลำโพงฝั่งกล้องแล้ว")
    try:
        while True:
            frame = await track.recv()
            resampled = resampler.resample(frame)
            # PyAV รุ่นใหม่คืน list, รุ่นเก่าคืนเฟรมเดียว
            if not isinstance(resampled, list):
                resampled = [resampled]
            for f in resampled:
                arr = f.to_ndarray()  # shape (1, samples*channels) สำหรับ s16 packed
                arr = arr.reshape(-1, 2)  # -> (samples, 2)
                await asyncio.get_event_loop().run_in_executor(None, out.write, arr)
    except Exception as e:
        log.info("หยุดเล่นเสียงขาเข้า: %s", e)
    finally:
        try:
            out.stop()
            out.close()
        except Exception:
            pass


# ==========================================================================
#  WebRTC signaling
# ==========================================================================
async def offer(request):
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    cfg = request.app["config"]

    @pc.on("connectionstatechange")
    async def on_state_change():
        log.info("การเชื่อมต่อ: %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log.info("ได้รับ track จากมือถือ: %s", track.kind)
        if track.kind == "audio" and cfg["speaker"] and AUDIO_AVAILABLE:
            asyncio.ensure_future(play_incoming_audio(track))

    # ส่งวิดีโอกล้องออกไปเสมอ
    video = CameraVideoTrack(
        cfg["camera"], cfg["width"], cfg["height"], cfg["fps"], cfg["flip"], cfg["mjpg"]
    )
    pc.addTrack(video)

    # ส่งเสียงไมค์ฝั่งกล้องออกไป (ถ้าเปิดใช้และมีอุปกรณ์)
    if cfg["mic"] and AUDIO_AVAILABLE:
        try:
            pc.addTrack(MicAudioTrack())
        except Exception as e:
            log.warning("เปิดไมค์ฝั่งกล้องไม่ได้ (ข้ามไป): %s", e)

    await pc.setRemoteDescription(offer_desc)

    # ถ้าเปิด --h264: ดัน H.264 ขึ้นเป็นตัวเลือกแรก (มือถือถอดรหัสด้วยฮาร์ดแวร์ ภาพลื่นกว่า)
    if cfg["h264"]:
        prefer_video_codec(pc, "video/H264")

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


def prefer_video_codec(pc, mime):
    """จัดลำดับ codec ที่ต้องการขึ้นก่อน แต่ยังเก็บตัวอื่นเป็น fallback (เจรจาไม่พังแน่นอน)"""
    try:
        caps = RTCRtpSender.getCapabilities("video")
        if not caps:
            return
        want = [c for c in caps.codecs if c.mimeType.lower() == mime.lower()]
        rest = [c for c in caps.codecs if c.mimeType.lower() != mime.lower()]
        if not want:
            return
        ordered = want + rest
        for t in pc.getTransceivers():
            if t.kind == "video":
                t.setCodecPreferences(ordered)
                log.info("ตั้งให้ใช้ %s เป็นตัวเลือกแรก", mime)
    except Exception as e:
        log.warning("ตั้ง codec %s ไม่ได้ (ใช้ค่าเริ่มต้นแทน): %s", mime, e)


async def index(request):
    with open(os.path.join(ROOT, "static", "index.html"), "r", encoding="utf-8") as f:
        return web.Response(content_type="text/html", text=f.read())


async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in pcs], return_exceptions=True)
    pcs.clear()


def build_app(config):
    app = web.Application()
    app["config"] = config
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_static("/static", os.path.join(ROOT, "static"))
    return app


def get_lan_ip():
    """หา IP ของเครื่องในวง LAN (เพื่อบอกว่ามือถือต้องเปิด URL ไหน)"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # ไม่ได้ส่งข้อมูลจริง แค่ให้ OS เลือก interface
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def ensure_self_signed_cert(cert_path, key_path, lan_ip):
    """สร้างใบรับรอง SSL แบบ self-signed อัตโนมัติ (ถ้ายังไม่มี) เพื่อให้ใช้ไมค์บนมือถือได้"""
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "catcam")])

    san = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        san.append(x509.IPAddress(ipaddress.ip_address(lan_ip)))
    except Exception:
        pass

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256())
    )
    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    log.info("สร้างใบรับรอง SSL ใหม่: %s", cert_path)


def main():
    p = argparse.ArgumentParser(description="catcam — กล้องดูแมวน้อย")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--camera", type=int, default=0, help="index กล้อง (0,1,...)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--flip", action="store_true", help="หมุนภาพ 180° (กล้องกลับหัว)")
    p.add_argument("--no-mjpg", action="store_true", help="ไม่ขอภาพ MJPG จากกล้อง (ใช้ถ้ากล้องมีปัญหา)")
    p.add_argument(
        "--h264",
        action="store_true",
        help="ใช้ codec H.264 (มือถือถอดรหัสด้วยฮาร์ดแวร์ ลื่นขึ้น) — ถ้าภาพไม่ขึ้นให้เอาออก",
    )
    p.add_argument("--no-mic", action="store_true", help="ปิดไมค์ฝั่งกล้อง (ไม่ฟังเสียงแมว)")
    p.add_argument("--no-speaker", action="store_true", help="ปิดลำโพงฝั่งกล้อง (คุยกับแมวไม่ได้)")
    p.add_argument("--cert", help="ไฟล์ใบรับรอง SSL (.pem) สำหรับ https")
    p.add_argument("--key", help="ไฟล์ private key (.pem) สำหรับ https")
    p.add_argument(
        "--https",
        action="store_true",
        help="เปิด HTTPS (สร้างใบรับรองเองอัตโนมัติ) — จำเป็นถ้าจะกดพูดบนมือถือ",
    )
    args = p.parse_args()

    if not AUDIO_AVAILABLE:
        log.warning(
            "โหมดวิดีโออย่างเดียว — ติดตั้ง sounddevice ไม่สำเร็จ (%s). "
            "ดูเสียงแมว/คุยกับแมวจะยังไม่ทำงาน",
            _AUDIO_IMPORT_ERROR,
        )

    config = {
        "camera": args.camera,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "flip": args.flip,
        "mjpg": not args.no_mjpg,
        "h264": args.h264,
        "mic": not args.no_mic,
        "speaker": not args.no_speaker,
    }

    lan_ip = get_lan_ip()

    ssl_ctx = None
    if args.cert and args.key:
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(args.cert, args.key)
    elif args.https:
        # สร้างใบรับรองเองไว้ข้าง ๆ server.py
        cert_path = os.path.join(ROOT, "cert.pem")
        key_path = os.path.join(ROOT, "key.pem")
        ensure_self_signed_cert(cert_path, key_path, lan_ip)
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(cert_path, key_path)

    app = build_app(config)
    scheme = "https" if ssl_ctx else "http"
    bar = "─" * 48
    print("\n" + bar)
    print("  🐾  catcam พร้อมแล้ว")
    print(bar)
    print(f"  เครื่องนี้ (Mac):  {scheme}://localhost:{args.port}")
    print(f"  มือถือ/แท็บเล็ต:   {scheme}://{lan_ip}:{args.port}")
    print(bar)
    if ssl_ctx:
        print("  📱 มือถือต้องอยู่ Wi-Fi วงเดียวกับ Mac")
        print("  ⚠️  ครั้งแรกเบราว์เซอร์จะเตือน 'ไม่ปลอดภัย' (เพราะใบรับรองทำเอง)")
        print("      ให้กด Advanced / รายละเอียด แล้วเลือก 'ไปต่อ' — ปลอดภัย")
        print("      หลังจากนั้นปุ่มกดพูดบนมือถือจะใช้ได้")
    else:
        print("  📱 มือถือ: ดูภาพได้ผ่าน LAN แต่ 'กดพูด' จะยังไม่ทำงาน")
        print("     อยากกดพูดบนมือถือ ให้รันใหม่ใส่  --https")
    print(bar + "\n")
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_ctx)


if __name__ == "__main__":
    main()
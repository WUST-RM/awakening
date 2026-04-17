from flask import Flask, render_template, Response, jsonify
import time
import json
import socket
import os
import logging
import struct
import mmap
import threading
import atexit
import setproctitle

setproctitle.setproctitle("wust_vision_web")

app = Flask(__name__)

STREAM_FPS = 60
FRAME_INTERVAL = 1.0 / STREAM_FPS

shared_memory_path = "/dev/shm/awakening_frame"
shared_size = 2 * 1024 * 1024  # 2MB
data_path = "/dev/shm/awakening_data.json"
log_path = "/dev/shm/awakening_log.json"

port = 8000

fd = None
mapfile = None
use_shared_memory = False


latest_jpg = None
latest_seq = 0
frame_cond = threading.Condition()
stop_event = threading.Event()
reader_thread = None


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def init_shared_memory() -> bool:
    global use_shared_memory, mapfile, fd

    try:
        if mapfile is not None:
            try:
                mapfile.close()
            except Exception:
                pass
            mapfile = None

        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
            fd = None

        fd = os.open(shared_memory_path, os.O_RDONLY)
        mapfile = mmap.mmap(fd, shared_size, access=mmap.ACCESS_READ)
        use_shared_memory = True
        print("✅ 共享内存初始化成功")
        return True

    except Exception as e:
        print(f"[ERROR] 共享内存初始化失败: {e}")
        use_shared_memory = False
        mapfile = None
        fd = None
        return False


def read_jpeg_from_shared_memory():
    if not (use_shared_memory and mapfile):
        return None

    try:
        mapfile.seek(0)
        size_bytes = mapfile.read(4)
        if len(size_bytes) != 4:
            return None

        jpg_size = struct.unpack("<I", size_bytes)[0]
        if not (0 < jpg_size <= shared_size - 4):
            return None

        data = mapfile.read(jpg_size)
        if len(data) != jpg_size:
            return None

        if not data.startswith(b"\xff\xd8"):
            return None

        return data

    except Exception:
        return None


def frame_reader_loop():
    global latest_jpg, latest_seq, use_shared_memory

    last_fix_attempt = 0.0

    while not stop_event.is_set():
        jpg = read_jpeg_from_shared_memory()

        if jpg is not None:
            with frame_cond:
                latest_jpg = jpg
                latest_seq += 1
                frame_cond.notify_all()

            time.sleep(FRAME_INTERVAL)
            continue

        now = time.time()
        if now - last_fix_attempt > 5.0:
            print("尝试重新初始化共享内存...")
            init_shared_memory()
            last_fix_attempt = now

        time.sleep(0.2)


@atexit.register
def cleanup():
    global mapfile, fd, stop_event

    stop_event.set()

    if mapfile is not None:
        try:
            mapfile.close()
        except Exception:
            pass
        mapfile = None

    if fd is not None:
        try:
            os.close(fd)
        except Exception:
            pass
        fd = None


def mjpeg_stream():
    last_seq = -1

    while not stop_event.is_set():
        with frame_cond:
            frame_cond.wait_for(
                lambda: latest_seq != last_seq or stop_event.is_set(),
                timeout=1.0,
            )

            if stop_event.is_set():
                break

            frame = latest_jpg
            last_seq = latest_seq

        if frame is None:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame +
            b"\r\n"
        )


@app.route("/")
def index():
    url = f"http://{get_local_ip()}:{port}"
    return render_template("index.html", server_url=url)


@app.route("/video")
def video_feed():
    return Response(
        mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/data")
def get_data():
    try:
        with open(data_path, "r") as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/log")
def get_log():
    try:
        with open(log_path, "r") as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    init_shared_memory()

    reader_thread = threading.Thread(target=frame_reader_loop, daemon=True)
    reader_thread.start()

    url = f"http://{get_local_ip()}:{port}"
    print(f"✅ Web 调试器已启动: {url}")

    app.run(
        host="0.0.0.0",
        port=port,
        threaded=True,
        use_reloader=False,
    )
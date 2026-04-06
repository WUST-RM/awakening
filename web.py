from flask import Flask, render_template, Response, jsonify
import time, json, socket, os, logging, struct
import mmap
import threading
import atexit
import fcntl
import setproctitle

setproctitle.setproctitle("wust_vision_web")

app = Flask(__name__)

STREAM_FPS = 60
FRAME_INTERVAL = 1.0 / STREAM_FPS

shared_memory_path = "/dev/shm/awaking_frame"
shared_size = 2 * 1024 * 1024  # 2MB


mapfile = None
fd = None

permission_lock = threading.Lock()
port = 8000



def ensure_shared_memory():
    with permission_lock:
        if not os.path.exists(shared_memory_path):
            print(f"创建共享内存文件: {shared_memory_path}")
            with open(shared_memory_path, "wb") as f:
                f.write(b"\0" * shared_size)
        try:
            os.chmod(shared_memory_path, 0o777)
        except PermissionError:
            print("[WARN] 无法修改权限，请手动 chmod 777")


def init_shared_memory():
    global use_shared_memory, mapfile, fd

    ensure_shared_memory()

    try:
        fd = os.open(shared_memory_path, os.O_RDONLY)
        mapfile = mmap.mmap(fd, shared_size, mmap.MAP_SHARED, mmap.PROT_READ)

        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except:
            pass

        use_shared_memory = True
        print("✅ 共享内存初始化成功")
        return True

    except Exception as e:
        print(f"[ERROR] 共享内存初始化失败: {e}")
        use_shared_memory = False
        return False



@atexit.register
def cleanup():
    global mapfile, fd
    if mapfile:
        try:
            mapfile.close()
        except:
            pass
    if fd:
        try:
            os.close(fd)
        except:
            pass



def mjpeg_stream():
    global mapfile
    last_fix_attempt = 0

    while True:
        jpg_bytes = None

        if use_shared_memory and mapfile:
            try:
                mapfile.seek(0)
                size_bytes = mapfile.read(4)

                if len(size_bytes) == 4:
                    jpg_size = struct.unpack("I", size_bytes)[0]

                    if 0 < jpg_size <= shared_size - 4:
                        data = mapfile.read(jpg_size)

                        if len(data) == jpg_size and data.startswith(b"\xff\xd8"):
                            jpg_bytes = data

            except Exception:
                current_time = time.time()
                if current_time - last_fix_attempt > 60:
                    print("尝试重新初始化共享内存...")
                    init_shared_memory()
                    last_fix_attempt = current_time

        if jpg_bytes is None:
            time.sleep(FRAME_INTERVAL)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg_bytes +
            b"\r\n"
        )

        time.sleep(FRAME_INTERVAL)



@app.route("/")
def index():
    def get_local_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        except:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

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
        with open("/dev/shm/awakening_data.json", "r") as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/log")
def get_log():
    try:
        with open("/dev/shm/awakening_log.json", "r") as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500






if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    init_shared_memory()

    def get_local_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        except:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    url = f"http://{get_local_ip()}:{port}"
    print(f"✅ Web 调试器已启动: {url}")

    app.run(host="0.0.0.0", port=port, threaded=True)
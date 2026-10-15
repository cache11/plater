
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import storage
import classifier

BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format_string, *args):
        pass


def start_audio_server():
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(
            QuietHandler,
            directory=str(storage.LIB_DIR)
        )
    )

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True
    )
    thread.start()

    return server, server.server_address[1]


class Api:
    def __init__(self, port):
        self.port = port
        self.window = None

    def list_songs(self):
        return storage.list_songs(self.port)

    def add_songs(self):
        import webview

        selected_files = self.window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=True,
            file_types=(
                "Audio files (*.mp3;*.wav;*.ogg;*.flac;*.m4a;*.aac;*.au)",
            )
        )

        if selected_files:
            storage.add_files(selected_files)

        return self.list_songs()

    def delete_song(self, song_id):
        storage.delete(song_id)
        return self.list_songs()

    def mark_played(self, song_id):
        storage.update(song_id, played=True)
        return True

    def set_duration(self, song_id, seconds):
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return False

        if seconds < 0:
            return False

        storage.update(song_id, duration=seconds)
        return True

    def classify_song(self, song_id):
        song = storage.get(song_id)

        if not song:
            return {"error": "Song not found."}

        audio_path = storage.LIB_DIR / song["file"]

        try:
            result = classifier.classify(str(audio_path))
        except Exception as exc:
            return {"error": str(exc)}

        storage.update(song_id, genre=result)

        return result


def main():
    import webview

    if not UI_DIR.is_dir():
        raise FileNotFoundError(
            f"Application UI folder was not found: {UI_DIR}"
        )

    index_file = UI_DIR / "index.html"

    if not index_file.is_file():
        raise FileNotFoundError(
            f"Application interface was not found: {index_file}"
        )

    server, port = start_audio_server()
    api = Api(port)

    window = webview.create_window(
        "Neon Music Player",
        url=index_file.as_uri(),
        js_api=api,
        width=1280,
        height=820,
        min_size=(420, 620),
        background_color="#10091f"
    )

    api.window = window

    try:
        webview.start()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--train":
        classifier.train(sys.argv[2])
    else:
        main()
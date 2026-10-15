
import json
import shutil
import time
import urllib.parse
from pathlib import Path

APP_DIR = Path.home() / "NeonMusicPlayer"
LIB_DIR = APP_DIR / "library"
DB_FILE = APP_DIR / "songs.json"

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".m4a",
    ".aac",
    ".au"
}

APP_DIR.mkdir(parents=True, exist_ok=True)
LIB_DIR.mkdir(parents=True, exist_ok=True)


def _load():
    if not DB_FILE.exists():
        return []

    try:
        data = json.loads(DB_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        backup = DB_FILE.with_suffix(".corrupted.json")

        try:
            shutil.copy2(DB_FILE, backup)
        except OSError:
            pass

        return []

    if not isinstance(data, list):
        return []

    return [
        song for song in data
        if isinstance(song, dict)
        and song.get("id")
        and song.get("file")
    ]


def _save(data):
    temporary_file = DB_FILE.with_suffix(".tmp")
    temporary_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    temporary_file.replace(DB_FILE)


def _split_title(stem):
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return title.strip(), artist.strip()

    return stem.strip(), "Unknown artist"


def list_songs(port):
    database = _load()
    available = []

    for song in database:
        filename = Path(song["file"]).name

        if filename != song["file"]:
            continue

        file_path = LIB_DIR / filename

        if not file_path.is_file():
            continue

        item = dict(song)
        item["url"] = (
            f"http://127.0.0.1:{port}/"
            f"{urllib.parse.quote(filename)}"
        )

        available.append(item)

    return available


def add_files(paths):
    database = _load()
    imported = 0

    existing_sources = set()

    for path in paths or []:
        source = Path(path).expanduser()

        if not source.is_file():
            continue

        if source.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        try:
            resolved_source = source.resolve()
        except OSError:
            continue

        if resolved_source in existing_sources:
            continue

        existing_sources.add(resolved_source)

        filename = source.name
        destination = LIB_DIR / filename
        counter = 1

        while destination.exists():
            destination = LIB_DIR / (
                f"{source.stem}_{counter}{source.suffix}"
            )
            counter += 1

        try:
            shutil.copy2(source, destination)
        except OSError:
            continue

        title, artist = _split_title(source.stem)

        database.append({
            "id": destination.name,
            "file": destination.name,
            "title": title,
            "artist": artist,
            "added": time.time(),
            "played": None,
            "duration": None,
            "genre": None
        })

        imported += 1

    if imported:
        _save(database)

    return imported


def get(song_id):
    for song in _load():
        if song["id"] == song_id:
            filename = Path(song["file"]).name

            if filename != song["file"]:
                return None

            if (LIB_DIR / filename).is_file():
                return song

            return None

    return None


def update(song_id, **fields):
    database = _load()
    allowed_fields = {"duration", "genre"}
    changed = False

    for song in database:
        if song["id"] != song_id:
            continue

        if fields.pop("played", False):
            song["played"] = time.time()
            changed = True

        for key, value in fields.items():
            if key in allowed_fields:
                song[key] = value
                changed = True

        break

    if changed:
        _save(database)


def delete(song_id):
    database = _load()
    updated_database = [
        song for song in database
        if song["id"] != song_id
    ]

    if len(updated_database) != len(database):
        _save(updated_database)

    return len(updated_database) != len(database)
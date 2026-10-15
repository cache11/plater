
from builtins import ValueError as BuiltinValueError
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "genre_model.joblib"

AUDIO_EXT = (
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".m4a",
    ".aac",
    ".au"
)

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

_model_cache = None
_model_mtime = None


class ValueError(BuiltinValueError):
    def __init__(self, message, *, context=None, details=None):
        self.message = str(message)
        self.context = context
        self.details = dict(details or {})
        super().__init__(self.message)

    def __str__(self):
        parts = [self.message]

        if self.context:
            parts.append(f"context={self.context}")

        if self.details:
            detail_text = "; ".join(
                f"{key}={value}"
                for key, value in sorted(self.details.items())
            )
            parts.append(f"details={{{detail_text}}}")

        return " | ".join(parts)

    @classmethod
    def for_audio(cls, path, reason, *, context=None):
        return cls(
            f"Unable to process audio file '{path}': {reason}",
            context=context or "audio_loading",
            details={
                "path": str(path),
                "reason": str(reason)
            }
        )

    @classmethod
    def for_dataset(
        cls,
        message,
        *,
        genre_count=None,
        sample_count=None
    ):
        details = {}

        if genre_count is not None:
            details["genre_count"] = genre_count

        if sample_count is not None:
            details["sample_count"] = sample_count

        return cls(
            message,
            context="dataset_validation",
            details=details
        )

    def to_dict(self):
        return {
            "message": self.message,
            "context": self.context,
            "details": dict(self.details)
        }


def normalize_genre(name):
    value = str(name).strip().lower()
    value = value.replace("-", "").replace("_", "").replace(" ", "")

    aliases = {
        "hiphop": "hiphop",
        "hip-hop": "hiphop",
        "classical": "classical",
        "country": "country",
        "reggae": "reggae",
        "blues": "blues",
        "disco": "disco",
        "jazz": "jazz",
        "metal": "metal",
        "pop": "pop",
        "rock": "rock"
    }

    return aliases.get(value, value)


def extract_features(path, duration=30):
    import librosa

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        y, sr = librosa.load(
            str(path),
            sr=22050,
            mono=True,
            duration=duration,
            offset=15
        )

        if len(y) < sr * 3:
            y, sr = librosa.load(
                str(path),
                sr=22050,
                mono=True,
                duration=duration,
                offset=0
            )

    except Exception as exc:
        raise ValueError(
            f"Could not read audio file '{path.name}': {exc}",
            context="audio_loading",
            details={"path": str(path)}
        ) from exc

    if len(y) < sr * 3:
        raise ValueError(
            f"Audio is too short for classification: {path.name}",
            context="audio_validation"
        )

    if not np.isfinite(y).all():
        raise ValueError(
            f"Audio contains invalid values: {path.name}",
            context="audio_validation"
        )

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = {
        "tempo": float(np.atleast_1d(tempo)[0]),
        "centroid": float(
            librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        ),
        "rolloff": float(
            librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        ),
        "bandwidth": float(
            librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        ),
        "zcr": float(
            librosa.feature.zero_crossing_rate(y).mean()
        ),
        "rms": float(rms.mean()),
        "rms_var": float(rms.std())
    }

    vector = np.hstack([
        list(features.values()),
        mfcc.mean(axis=1),
        mfcc.std(axis=1)
    ]).astype(np.float32)

    if not np.isfinite(vector).all():
        raise ValueError(
            "Audio feature extraction produced invalid values.",
            context="feature_extraction"
        )

    return features, vector


def _rules(features):
    tempo = features["tempo"]
    centroid = features["centroid"]
    zcr = features["zcr"]
    rms = features["rms"]
    rms_var = features["rms_var"]

    scores = dict.fromkeys(GENRES, 0.0)

    scores["classical"] += (
        (centroid < 1600) * 2
        + (rms < 0.05) * 2
        + (zcr < 0.06)
    )

    scores["jazz"] += (
        (1200 < centroid < 2200) * 1.5
        + (rms < 0.1)
        + (90 < tempo < 140)
    )

    scores["blues"] += (
        (1300 < centroid < 2300)
        + (70 < tempo < 110) * 1.5
        + (rms < 0.12)
    )

    scores["country"] += (
        (1800 < centroid < 2600)
        + (90 < tempo < 130)
        + (0.06 < zcr < 0.1)
    )

    scores["reggae"] += (
        (60 < tempo < 95) * 1.5
        + (1500 < centroid < 2500)
        + (rms_var > 0.03)
    )

    scores["hiphop"] += (
        (80 < tempo < 105) * 1.5
        + (rms > 0.12) * 1.5
        + (centroid < 2600)
    )

    scores["pop"] += (
        (100 < tempo < 130)
        + (2000 < centroid < 3200) * 1.5
        + (rms > 0.1)
    )

    scores["rock"] += (
        (2400 < centroid < 3400) * 1.5
        + (zcr > 0.09)
        + (rms > 0.1)
        + (110 < tempo < 160)
    )

    scores["metal"] += (
        (centroid > 3000) * 2
        + (zcr > 0.12) * 2
        + (rms > 0.13)
    )

    scores["disco"] += (
        (100 < tempo < 140) * 1.5
        + (centroid > 1800)
        + (rms_var < 0.05)
    )

    values = np.asarray(list(scores.values()), dtype=float)
    values = np.exp(values - values.max())
    values /= values.sum()

    return dict(zip(scores.keys(), values.tolist()))


def _load_model():
    global _model_cache, _model_mtime

    if not MODEL_FILE.is_file():
        _model_cache = None
        _model_mtime = None
        return None

    mtime = MODEL_FILE.stat().st_mtime

    if _model_cache is None or _model_mtime != mtime:
        import joblib

        _model_cache = joblib.load(MODEL_FILE)
        _model_mtime = mtime

    return _model_cache


def classify(path):
    features, vector = extract_features(path)
    model = _load_model()

    if model is not None:
        scores_array = model.predict_proba(vector.reshape(1, -1))[0]
        classes = model.classes_

        scores = {
            normalize_genre(label): float(score)
            for label, score in zip(classes, scores_array)
            if normalize_genre(label) in GENRES
        }

        if not scores:
            raise ValueError(
                "The trained model returned no supported genres.",
                context="model_prediction"
            )

        method = "Trained ML model"

    else:
        scores = _rules(features)
        method = "Heuristic audio analysis (fallback)"

    ordered = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top = ordered[:3]

    return {
        "genre": top[0][0],
        "confidence": round(top[0][1] * 100, 1),
        "top": [
            [genre, round(score * 100, 1)]
            for genre, score in top
        ],
        "tempo": round(features["tempo"]),
        "method": method
    }


def train(dataset_dir):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    dataset_dir = Path(dataset_dir)

    if not dataset_dir.is_dir():
        raise NotADirectoryError(
            f"Dataset directory does not exist: {dataset_dir}"
        )

    X = []
    y = []

    for genre_dir in sorted(dataset_dir.iterdir()):
        if not genre_dir.is_dir():
            continue

        genre = normalize_genre(genre_dir.name)

        if genre not in GENRES:
            print(f"Skipping unsupported genre folder: {genre_dir.name}")
            continue

        files = sorted(
            file for file in genre_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in AUDIO_EXT
        )

        for file in files:
            try:
                _, vector = extract_features(file)
                X.append(vector)
                y.append(genre)
                print(f"Loaded: {genre} / {file.name}")
            except Exception as exc:
                print(f"Skipped: {file.name} — {exc}")

    if not X:
        raise ValueError(
            "No usable audio files were found. Check your dataset folders.",
            context="dataset_validation"
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    print(f"\nUsable audio files: {len(X)}")
    print(f"Genres found: {sorted(set(y))}")

    if len(X) < 10 or len(set(y)) < 2:
        raise ValueError(
            "The dataset is too small or contains fewer than two genres. "
            "Add more labeled audio files before training.",
            context="dataset_validation",
            details={
                "sample_count": len(X),
                "genre_count": len(set(y))
            }
        )

    unique, counts = np.unique(y, return_counts=True)
    can_stratify = counts.min() >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if can_stratify else None
    )

    def make_model():
        return make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )

    evaluation_model = make_model()
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)

    print(
        f"\nHeld-out test accuracy: "
        f"{accuracy_score(y_test, predictions):.3f}"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    final_model = make_model()
    final_model.fit(X, y)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_FILE)

    global _model_cache, _model_mtime

    _model_cache = final_model
    _model_mtime = MODEL_FILE.stat().st_mtime

    print(f"Final model saved to: {MODEL_FILE}")
    print("Training complete.")
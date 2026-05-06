"""
core/face_store.py
Persistence layer for face encodings (pickle files) and the KNN classifier
that is rebuilt from them on every detection request.
"""

import os
import pickle

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from config import NAMES_PKL, FACES_PKL, KNN_MAX_K


# ── Low-level pickle I/O ───────────────────────────────────────────────────────

def load_face_data() -> tuple[list[str] | None, np.ndarray | None]:
    """
    Load names and face encodings from disk.
    Returns (None, None) if either file is missing.
    """
    if not os.path.exists(NAMES_PKL) or not os.path.exists(FACES_PKL):
        return None, None
    with open(NAMES_PKL, 'rb') as f:
        names: list[str] = pickle.load(f)
    with open(FACES_PKL, 'rb') as f:
        faces: np.ndarray = pickle.load(f)
    return names, faces


def save_face_data(name: str, face_samples: list[np.ndarray]) -> None:
    """
    Append *face_samples* (each a 1-D encoding) labelled *name* to the
    pickle files, creating them if they don't exist yet.
    """
    new_faces = np.array(face_samples)          # shape: (N, 7500)
    new_names = [name] * len(face_samples)

    existing_names, existing_faces = load_face_data()

    if existing_names is not None:
        existing_names = existing_names + new_names
        existing_faces = np.vstack([existing_faces, new_faces])
    else:
        existing_names = new_names
        existing_faces = new_faces

    with open(NAMES_PKL, 'wb') as f:
        pickle.dump(existing_names, f)
    with open(FACES_PKL, 'wb') as f:
        pickle.dump(existing_faces, f)


def delete_face_data_for(student_name: str) -> None:
    """Remove all encodings belonging to *student_name* from the pickle files."""
    names, faces = load_face_data()
    if names is None:
        return
    mask = [n != student_name for n in names]
    filtered_names = [n for n, keep in zip(names, mask) if keep]
    filtered_faces = faces[mask]
    with open(NAMES_PKL, 'wb') as f:
        pickle.dump(filtered_names, f)
    with open(FACES_PKL, 'wb') as f:
        pickle.dump(filtered_faces, f)


# ── KNN model ─────────────────────────────────────────────────────────────────

def get_knn_model() -> tuple[KNeighborsClassifier | None, list[str] | None]:
    """
    Build and return a fitted KNN classifier from the stored face data.
    Returns (None, None) if no data is available yet.
    """
    names, faces = load_face_data()
    if names is None or len(names) == 0:
        return None, None
    k = min(KNN_MAX_K, len(set(names)))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(faces, names)
    return knn, names

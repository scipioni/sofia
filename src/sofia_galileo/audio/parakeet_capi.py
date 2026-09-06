"""ctypes binding for parakeet.cpp's flat streaming C-API.

The C-API is flat and exception-free by design (see parakeet_capi.h) so it can
be embedded without a C++ ABI dependency — this project already avoids
compiled extensions of its own (qaa/tools.py), and ctypes keeps that posture:
no build step, no wheel, just a shared library loaded at runtime. CDLL calls
also release the GIL for the duration of the call, which is what makes the
`anyio.to_thread.run_sync` offload around `push()` in realtime.py worthwhile.

Symbol presence is validated once, at load, rather than left to fail on first
call: an ABI drift from an unpinned build should surface at startup, not
mid-session (design.md D9 in add-parakeet-streaming-asr).
"""

from __future__ import annotations

import ctypes

import numpy as np

_REQUIRED_SYMBOLS = (
    "parakeet_capi_load",
    "parakeet_capi_free",
    "parakeet_capi_last_error",
    "parakeet_capi_stream_begin_lang",
    "parakeet_capi_stream_feed",
    "parakeet_capi_stream_finalize",
    "parakeet_capi_stream_free",
    "parakeet_capi_free_string",
)


class ParakeetLibraryError(RuntimeError):
    """The library could not be loaded, or is missing a symbol this binding calls."""


class ParakeetModelError(RuntimeError):
    """The C-API reported an error (see parakeet_capi_last_error)."""


def _configure(lib: ctypes.CDLL) -> None:
    lib.parakeet_capi_load.restype = ctypes.c_void_p
    lib.parakeet_capi_load.argtypes = [ctypes.c_char_p]

    lib.parakeet_capi_free.restype = None
    lib.parakeet_capi_free.argtypes = [ctypes.c_void_p]

    lib.parakeet_capi_last_error.restype = ctypes.c_char_p
    lib.parakeet_capi_last_error.argtypes = [ctypes.c_void_p]

    lib.parakeet_capi_stream_begin_lang.restype = ctypes.c_void_p
    lib.parakeet_capi_stream_begin_lang.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    lib.parakeet_capi_stream_feed.restype = ctypes.c_void_p
    lib.parakeet_capi_stream_feed.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]

    lib.parakeet_capi_stream_finalize.restype = ctypes.c_void_p
    lib.parakeet_capi_stream_finalize.argtypes = [ctypes.c_void_p]

    lib.parakeet_capi_stream_free.restype = None
    lib.parakeet_capi_stream_free.argtypes = [ctypes.c_void_p]

    lib.parakeet_capi_free_string.restype = None
    lib.parakeet_capi_free_string.argtypes = [ctypes.c_void_p]


def load_library(path: str) -> ctypes.CDLL:
    """Load libparakeet.so and validate the symbols this binding calls.

    Raises ParakeetLibraryError rather than letting a missing symbol surface as
    an AttributeError deep inside a session, or an ABI drift surface as a
    segfault mid-call.
    """
    try:
        lib = ctypes.CDLL(path)
    except OSError as exc:
        raise ParakeetLibraryError(f"could not load parakeet library {path!r}: {exc}") from exc

    missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(lib, name)]
    if missing:
        raise ParakeetLibraryError(
            f"{path!r} is missing required symbols {missing}; "
            "the pinned parakeet.cpp commit may not match this binding"
        )
    _configure(lib)
    return lib


def _take_string(lib: ctypes.CDLL, ptr: int | None) -> str:
    """Decode and free a malloc'd UTF-8 string returned by the C-API."""
    if not ptr:
        return ""
    text = ctypes.cast(ptr, ctypes.c_char_p).value or b""
    lib.parakeet_capi_free_string(ptr)
    return text.decode("utf-8", errors="replace")


class ParakeetContext:
    """One loaded model. Cheap to spawn many streams from; expensive to load."""

    def __init__(self, lib: ctypes.CDLL, model_path: str) -> None:
        self._lib = lib
        self._ctx: int | None = lib.parakeet_capi_load(model_path.encode("utf-8"))
        if not self._ctx:
            raise ParakeetModelError(f"failed to load model at {model_path!r}")

    def stream(self, lang: str) -> ParakeetStream:
        s = self._lib.parakeet_capi_stream_begin_lang(self._ctx, lang.encode("utf-8"))
        if not s:
            err = self._lib.parakeet_capi_last_error(self._ctx)
            message = err.decode("utf-8", errors="replace") if err else "unknown error"
            raise ParakeetModelError(f"stream_begin_lang failed for lang={lang!r}: {message}")
        return ParakeetStream(self._lib, s)

    def close(self) -> None:
        if self._ctx:
            self._lib.parakeet_capi_free(self._ctx)
            self._ctx = None

    def __del__(self) -> None:
        self.close()


class ParakeetStream:
    """One streaming session's C-API handle. Not safe to share across connections."""

    def __init__(self, lib: ctypes.CDLL, handle: int) -> None:
        self._lib = lib
        self._handle: int | None = handle

    def feed(self, pcm_16k_mono: np.ndarray) -> tuple[str, bool]:
        """Feed 16 kHz mono float32 PCM. Returns (new_text, eou).

        `eou` is populated only by nvidia/parakeet_realtime_eou_120m-v1, not by
        the multilingual Nemotron model this project uses — see design.md D6.
        Callers on the Nemotron path should not rely on it for turn boundaries.
        """
        pcm = np.ascontiguousarray(pcm_16k_mono, dtype=np.float32)
        ptr_pcm = pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        eou = ctypes.c_int(0)
        ptr = self._lib.parakeet_capi_stream_feed(
            self._handle, ptr_pcm, len(pcm), ctypes.byref(eou)
        )
        if ptr is None:
            raise ParakeetModelError("stream_feed failed")
        return _take_string(self._lib, ptr), bool(eou.value)

    def finalize(self) -> str:
        ptr = self._lib.parakeet_capi_stream_finalize(self._handle)
        if ptr is None:
            raise ParakeetModelError("stream_finalize failed")
        return _take_string(self._lib, ptr)

    def close(self) -> None:
        if self._handle:
            self._lib.parakeet_capi_stream_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

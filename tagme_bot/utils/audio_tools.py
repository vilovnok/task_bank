# -*- coding: utf-8 -*-
"""
Audio Quality Analysis Tools (v2)

Анализ дефектов записи/синтеза: шум, клики, клиппинг, эхо, "металл",
музыка, затухание. Возвращает те же ключи, что и v1, плюс сырые признаки
(features) для подбора порогов по голдену.

Что изменено относительно v1 (см. CHANGELOG в конце файла).
Тяжёлые вычисления сделаны на numpy/scipy; librosa нужна только для
загрузки нестандартных форматов и для HPSS (музыка).
"""

import warnings
warnings.filterwarnings("ignore")

import base64
import dataclasses
import json as json_module
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from scipy import ndimage
from scipy import signal as scipy_signal


@dataclass
class AudioDefect:
    type: str
    start: float
    end: float
    severity: float
    description: str


# --------------------------------------------------------------------------- #
# Пороги в одном месте — их и нужно калибровать по голдену
# --------------------------------------------------------------------------- #
DEFAULT_THRESHOLDS = {
    "loud_noise_snr_db": 10.0,      # SNR ниже -> loud_noise
    "click_ratio": 8.0,             # всплеск |dx| во сколько раз выше локальной медианы
    "click_abs_min": 0.05,          # и не ниже этого абсолютного уровня |dx|
    "clip_ratio": 0.001,            # доля сэмплов |x|>=0.99 -> клиппинг
    "metallic_centroid_hz": 3200.0, # средний центроид активных фреймов (было 0.4*Nyquist)
    "echo_peak": 0.3,               # пик автокорреляции в лагах 50-500 мс
    "music_harmonic_ratio": 0.7,    # доля гармонической энергии (HPSS), НЕ откалибровано
    "fading_drop": 0.7,             # хвост/начало по RMS активной речи
}


class AudioQualityAnalyzer:
    def __init__(
            self,
            sr: int = 16000,
            frame_length: int = 2048,
            hop_length: int = 512,
            thresholds: Optional[Dict[str, float]] = None,
            verify_ssl: bool = False,
            enable_music: bool = True,
            penalize_fading: bool = False,
    ):
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.verify_ssl = verify_ssl
        self.enable_music = enable_music
        self.penalize_fading = penalize_fading

    # ------------------------------------------------------------------ I/O
    def _load_path(self, path: str) -> Tuple[np.ndarray, int]:
        try:
            import librosa
            y, sr = librosa.load(path, sr=self.sr, mono=True)
            return y, sr
        except ImportError:
            import soundfile as sf
            y, sr = sf.read(path, dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != self.sr:
                y = scipy_signal.resample_poly(y, self.sr, sr).astype(np.float32)
            return y, self.sr

    def _load_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            return self._load_path(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def load_audio(self, audio_source: str) -> Tuple[np.ndarray, int]:
        if isinstance(audio_source, str) and audio_source.strip().startswith("data:audio/"):
            s = audio_source.strip()
            if ";base64," not in s:
                raise ValueError("Not a base64 data:audio URL")
            return self._load_bytes(base64.b64decode(s.split(";base64,", 1)[1]))

        if isinstance(audio_source, str) and audio_source.startswith(("http://", "https://")):
            r = requests.get(audio_source, timeout=30.0, verify=self.verify_ssl)
            r.raise_for_status()
            return self._load_bytes(r.content)

        return self._load_path(audio_source)

    # -------------------------------------------------------------- helpers
    def _frames(self, y: np.ndarray) -> np.ndarray:
        """Кадры с центрированием (как librosa center=True)."""
        pad = self.frame_length // 2
        yp = np.pad(y, (pad, pad), mode="reflect") if len(y) > pad else np.pad(y, (pad, pad))
        win = np.lib.stride_tricks.sliding_window_view(yp, self.frame_length)
        return win[:: self.hop_length]

    def _rms(self, y: np.ndarray) -> np.ndarray:
        fr = self._frames(y)
        return np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1))

    @staticmethod
    def _db(x, eps=1e-8):
        return 20.0 * np.log10(np.maximum(x, eps))

    # ------------------------------------------------------------------ SNR
    def estimate_snr(self, y: np.ndarray, percentile: float = 10) -> float:
        """
        SNR = средний RMS верхних 90% кадров / средний RMS нижних 10%.
        Fix: при почти цифровой тишине (noise_rms~0) v1 возвращал 100 dB,
        при вырожденных кадрах - 0 dB (=> 100% шума). Теперь пол шума 1e-5
        и результат ограничен [0, 60] dB.
        """
        rms = self._rms(y)
        if len(rms) == 0:
            return 0.0
        thr = np.percentile(rms, percentile)
        noise = rms[rms <= thr]
        sig = rms[rms > thr]
        if len(noise) == 0 or len(sig) == 0:
            return 60.0 if np.max(rms) > 1e-4 else 0.0
        noise_rms = max(float(np.mean(noise)), 1e-5)
        snr = 20 * np.log10(float(np.mean(sig)) / noise_rms)
        return float(np.clip(snr, 0.0, 60.0))

    def detect_background_noise(self, y: np.ndarray) -> Dict[str, Any]:
        snr_db = self.estimate_snr(y, 10)
        if snr_db >= 20:
            noise_percent = 0.0
        elif snr_db >= 10:
            noise_percent = 50.0 * (1 - (snr_db - 10) / 10)
        elif snr_db >= 5:
            noise_percent = 50.0 + 30.0 * (1 - (snr_db - 5) / 5)
        else:
            noise_percent = min(100.0, 80.0 + 20.0 * (1 - snr_db / 5))
        rms_db = self._db(self._rms(y))
        return {
            "snr_db": float(snr_db),
            "noise_percent": float(np.clip(noise_percent, 0, 100)),
            "has_loud_noise": bool(snr_db < self.th["loud_noise_snr_db"]),
            "noise_floor_dbfs": float(np.percentile(rms_db, 10)) if len(rms_db) else -100.0,
        }

    # --------------------------------------------------------------- clicks
    def detect_clicks(self, y: np.ndarray) -> List[AudioDefect]:
        """
        Fix: v1 искал клики по ZCR на окнах 128 мс. ZCR высок на любых
        фрикативных (с, ш, щ, ф, х) -> почти каждый файл получал 2-4 "клика",
        а сам клик длится миллисекунды и в 128-мс окне не виден.
        Теперь: всплески первой разности |dx| на 1-мс блоках относительно
        локальной медианы (~41 мс). Клик = резкий узкий выброс.
        """
        if len(y) < self.sr // 10:
            return []
        blk = max(1, int(0.001 * self.sr))
        d = np.abs(np.diff(y, prepend=y[0]))
        n = (len(d) // blk) * blk
        bmax = d[:n].reshape(-1, blk).max(axis=1)
        local = ndimage.median_filter(bmax, size=41, mode="nearest")
        ratio = bmax / np.maximum(local, 1e-4)
        hit = np.where((ratio > self.th["click_ratio"]) & (bmax > self.th["click_abs_min"]))[0]
        if len(hit) == 0:
            return []

        groups, cur = [], [hit[0]]
        for b in hit[1:]:
            if b - cur[-1] <= 10:      # склеиваем всплески в пределах 10 мс
                cur.append(b)
            else:
                groups.append(cur)
                cur = [b]
        groups.append(cur)

        out = []
        for g in groups:
            sev = min(1.0, float((ratio[g].max() - self.th["click_ratio"]) / self.th["click_ratio"]))
            out.append(AudioDefect(
                type="click",
                start=float(g[0] * blk / self.sr),
                end=float((g[-1] + 1) * blk / self.sr),
                severity=max(0.0, sev),
                description="Клик/щелчок",
            ))
        return out

    # ------------------------------------------------------------- clipping
    def detect_clipping(self, y: np.ndarray) -> Dict[str, Any]:
        """New: клиппинг (перегруз) - частый дефект, в v1 не проверялся."""
        if len(y) == 0:
            return {"clip_ratio": 0.0, "has_clipping": False}
        ratio = float(np.mean(np.abs(y) >= 0.99))
        return {"clip_ratio": ratio, "has_clipping": bool(ratio > self.th["clip_ratio"])}

    # ------------------------------------------------------------- metallic
    def detect_metallic_sound(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Fix: v1 усреднял центроид по ВСЕМ кадрам, включая тишину/шум, где
        центроид случайный. Теперь только активные кадры (> -35 dB от пика).
        Порог в Гц (3200 = 0.4*Nyquist при sr=16k, как в v1), не завязан на sr.
        """
        fr = self._frames(y).astype(np.float64)
        rms_db = self._db(np.sqrt(np.mean(fr ** 2, axis=1)))
        active = rms_db > (rms_db.max() - 35.0)
        if active.sum() == 0:
            return {"centroid_mean_hz": 0.0, "centroid_mean": 0.0, "is_metallic": False, "severity": 0.0,
                    "rolloff95_hz": 0.0}
        spec = np.abs(np.fft.rfft(fr[active] * np.hanning(self.frame_length), axis=1))
        freqs = np.fft.rfftfreq(self.frame_length, 1.0 / self.sr)
        power = spec ** 2 + 1e-12
        centroid = (power * freqs).sum(axis=1) / power.sum(axis=1)
        cum = np.cumsum(power, axis=1) / power.sum(axis=1, keepdims=True)
        rolloff = freqs[np.argmax(cum >= 0.95, axis=1)]
        c = float(np.mean(centroid))
        thr = self.th["metallic_centroid_hz"]
        return {
            "centroid_mean_hz": c,
            "centroid_mean": float(c / (self.sr / 2)),
            "rolloff95_hz": float(np.mean(rolloff)),
            "is_metallic": bool(c > thr),
            "severity": float(max(0.0, (c - thr) / (0.75 * self.sr / 2 - thr))),
        }

    # ----------------------------------------------------------------- echo
    def detect_echo(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Fix: v1 использовал np.correlate(mode='full') - O(N^2), для 10 с при
        16 кГц это ~2.5e10 операций (минуты). Теперь FFT-автокорреляция.
        Также убираем DC и нормируем по энергии.
        """
        y = y - np.mean(y)
        if len(y) < 2 or not np.any(y):
            return {"has_echo": False, "echo_strength": 0.0}
        ac = scipy_signal.fftconvolve(y, y[::-1], mode="full")[len(y) - 1:]
        ac = ac / ac[0]
        lo, hi = int(0.05 * self.sr), min(int(0.5 * self.sr), len(ac) - 1)
        if hi <= lo:
            return {"has_echo": False, "echo_strength": 0.0}
        peak = float(np.max(ac[lo:hi]))
        return {"has_echo": bool(peak > self.th["echo_peak"]), "echo_strength": peak}

    # --------------------------------------------------------------- fading
    def detect_fading(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Fix: v1 сравнивал первые и последние 25% ВСЕХ кадров, включая паузы
        и хвостовую тишину -> нормальные фразы помечались как "затухание"
        (в вашей выгрузке has_fading=True у 19 из 56). К тому же файлы идут
        через s2s_norm_lufs_and_fading, т.е. fade-out там намеренный.
        Теперь сравниваются только активные кадры (> -35 dB от пика).
        """
        rms = self._rms(y)
        if len(rms) < 10:
            return {"has_fading": False, "fade_percent": 0.0}
        db = self._db(rms)
        act = rms[db > db.max() - 35.0]
        if len(act) < 8:
            return {"has_fading": False, "fade_percent": 0.0}
        q = len(act) // 4
        start, end = float(np.mean(act[:q])), float(np.mean(act[-q:]))
        if start <= 0:
            return {"has_fading": False, "fade_percent": 0.0}
        ratio = end / start
        return {"has_fading": bool(ratio < self.th["fading_drop"]),
                "fade_percent": float((1 - ratio) * 100)}

    # ---------------------------------------------------------------- music
    def detect_music(self, y: np.ndarray) -> Dict[str, Any]:
        """
        ВНИМАНИЕ: гармоническая доля HPSS у чистой речи обычно уже высокая
        (вокализованные звуки гармоничны), поэтому порог 0.7 ненадёжен.
        Детектор оставлен для совместимости; нужен калибровочный прогон.
        """
        if not self.enable_music:
            return {"has_music": False, "music_confidence": 0.0}
        import librosa
        yh, _ = librosa.effects.hpss(y)
        total = float(np.sum(y ** 2))
        if total == 0:
            return {"has_music": False, "music_confidence": 0.0}
        hr = float(np.sum(yh ** 2) / total)
        return {"has_music": bool(hr > self.th["music_harmonic_ratio"]), "music_confidence": hr}

    # -------------------------------------------------------------- analyze
    @staticmethod
    def _defect_dict(d: AudioDefect) -> Dict[str, Any]:
        return {"type": d.type, "start_sec": float(d.start), "end_sec": float(d.end),
                "duration_sec": float(d.end - d.start), "severity": float(d.severity),
                "description": d.description}

    def analyze(self, audio_source: str) -> Dict[str, Any]:
        y, sr = self.load_audio(audio_source)
        dur = float(len(y) / sr)

        noise = self.detect_background_noise(y)
        clicks = self.detect_clicks(y)
        clip = self.detect_clipping(y)
        metallic = self.detect_metallic_sound(y)
        echo = self.detect_echo(y)
        fading = self.detect_fading(y)
        music = self.detect_music(y)

        defects = [self._defect_dict(c) for c in clicks]

        def whole(type_, sev, desc, start=0.0):
            defects.append({"type": type_, "start_sec": start, "end_sec": dur,
                            "duration_sec": dur - start,
                            "severity": float(np.clip(sev, 0.0, 1.0)), "description": desc})

        if noise["has_loud_noise"]:
            whole("loud_noise", 1.0 - noise["snr_db"] / 20.0, f"Громкий фоновый шум (SNR={noise['snr_db']:.1f} dB)")
        if music["has_music"]:
            whole("music", music["music_confidence"], "Музыка в фоне")
        if echo["has_echo"]:
            whole("echo", echo["echo_strength"], f"Эхо (strength={echo['echo_strength']:.2f})")
        if metallic["is_metallic"]:
            whole("metallic", metallic["severity"], "Металлический голос")
        if clip["has_clipping"]:
            whole("clipping", clip["clip_ratio"] * 100, f"Клиппинг ({clip['clip_ratio']*100:.2f}% сэмплов)")
        if fading["has_fading"]:
            whole("fading", fading["fade_percent"] / 100.0, f"Звук затихает ({fading['fade_percent']:.1f}% потери)",
                  start=dur * 0.75)

        q = 100.0 - noise["noise_percent"] * 0.5
        q -= min(len(clicks) * 2, 20)                    # v1: без ограничения
        q -= 20 if music["has_music"] else 0
        q -= 15 if echo["has_echo"] else 0
        q -= 10 if metallic["is_metallic"] else 0
        q -= 15 if clip["has_clipping"] else 0
        if self.penalize_fading and fading["has_fading"]:
            q -= 10
        q = float(np.clip(q, 0.0, 100.0))

        return {
            "noise_percent": noise["noise_percent"],
            "quality_score": q,
            "defects": defects,
            "snr_db": noise["snr_db"],
            "has_loud_noise": noise["has_loud_noise"],
            "has_music": music["has_music"],
            "has_echo": echo["has_echo"],
            "has_metallic_sound": metallic["is_metallic"],
            "has_fading": fading["has_fading"],
            "has_clipping": clip["has_clipping"],
            "duration_sec": dur,
            "defects_count": int(len(defects)),
            "clicks_count": int(len(clicks)),
            # сырые признаки для подбора порогов по голдену
            "features": {
                "noise_floor_dbfs": noise["noise_floor_dbfs"],
                "clip_ratio": clip["clip_ratio"],
                "centroid_mean_hz": metallic["centroid_mean_hz"],
                "rolloff95_hz": metallic["rolloff95_hz"],
                "echo_strength": echo["echo_strength"],
                "music_confidence": music["music_confidence"],
                "fade_percent": fading["fade_percent"],
                "clicks_per_sec": float(len(clicks) / dur) if dur > 0 else 0.0,
                "max_click_severity": float(max([c.severity for c in clicks], default=0.0)),
            },
        }


# --------------------------------------------------------------------------- #
# Работа со входным dict
# --------------------------------------------------------------------------- #
def _normalize_input_key(key: str) -> str:
    return key.replace("INPUT:", "") if key.startswith("INPUT:") else key


def _extract_audio_url_from_dict(input_dict: Dict[str, Any]) -> Optional[str]:
    for key in ["INPUT:zaudio_links", "zaudio_links"]:
        if key in input_dict:
            f = input_dict[key]
            if not f:
                continue
            if isinstance(f, str):
                f = f.strip()
                if not f:
                    continue
                if f.startswith("["):
                    try:
                        parsed = json_module.loads(f)
                        if isinstance(parsed, list) and parsed:
                            return str(parsed[0]).strip()
                    except json_module.JSONDecodeError:
                        pass
                return f
            if isinstance(f, list) and f:
                return str(f[0]).strip()
    for key in ["INPUT:audio", "audio", "INPUT:path", "path"]:
        if key in input_dict and input_dict[key]:
            return str(input_dict[key]).strip()
    return None


def analyze_audio_quality(audio_source: str, sr: int = 16000, **kwargs) -> Dict[str, Any]:
    return AudioQualityAnalyzer(sr=sr, **kwargs).analyze(audio_source)


_RESULT_KEYS = ["duration_sec", "noise_percent", "quality_score", "snr_db", "has_loud_noise",
                "has_music", "has_echo", "has_metallic_sound", "has_fading", "has_clipping",
                "defects_count", "clicks_count", "defects", "features"]


def analyze_audio_quality_from_dict(
        input_dict: Dict[str, Any],
        sr: int = 16000,
        include_input_fields: bool = True,
        **kwargs,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if include_input_fields:
        for field in ["chat_id", "file_name", "FILENAME", "context", "dialog", "criteria", "fact",
                      "isCommentAvailable", "isCommentRequired", "score", "tagsList"]:
            for key in (f"INPUT:{field}", field):
                if key in input_dict:
                    result[field] = input_dict[key]
                    break

    audio_url = _extract_audio_url_from_dict(input_dict)
    if not audio_url:
        result.update({"audio_url": None, **{k: None for k in _RESULT_KEYS},
                       "error": "No audio URL found in input dict (checked: INPUT:zaudio_links, zaudio_links, audio, path)"})
        return result

    result["audio_url"] = audio_url
    try:
        analysis = analyze_audio_quality(audio_url, sr=sr, **kwargs)
        result.update({k: analysis.get(k) for k in _RESULT_KEYS})
        result["error"] = None
    except Exception as e:  # noqa: BLE001
        result.update({k: None for k in _RESULT_KEYS})
        result["error"] = f"{type(e).__name__}: {e}"
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_tools.py <audio_file_or_url>")
        sys.exit(1)

    res = analyze_audio_quality(sys.argv[1])
    print(f"Duration: {res['duration_sec']:.2f} s | Noise: {res['noise_percent']:.1f}% | "
          f"Quality: {res['quality_score']:.1f}/100 | SNR: {res['snr_db']:.1f} dB")
    print(f"Features: {json_module.dumps(res['features'], ensure_ascii=False)}")
    for i, d in enumerate(res["defects"], 1):          # v1: d['start'] -> KeyError
        print(f"  {i}. {d['description']}  {d['start_sec']:.2f}-{d['end_sec']:.2f}s  sev={d['severity']:.2f}")

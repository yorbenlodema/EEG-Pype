#!/usr/bin/env python3
"""
Scientific workflow
-------------------
* Recursively inventory EEG-Pype .txt output.
* Keep sensor-level and each source method / atlas / region selection separate.
* Prefer unfiltered data and otherwise use the widest passband that covers the fit.
* Calculate a multitaper PSD for each saved epoch or internal continuous-data window.
* Average PSDs in linear power space before spectral parameterization.
* Fit both a global (channel-averaged) spectrum and every channel / source region.
* Preview fits before running and exporting the full batch.
* Retain failed and warned fits; no fit is silently excluded by fit quality.

Dependencies
------------
numpy, pandas, scipy, mne, matplotlib, openpyxl, FreeSimpleGUI,
specparam==2.0.0rc7

Suggested installation command for the new dependency:
    pip install specparam==2.0.0rc7

Author: Yorben Lodema
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import pickle
import platform
import random
import re
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import mne
import numpy as np
import pandas as pd
from scipy import signal

try:
    import FreeSimpleGUI as sg
except ImportError:  # Allows non-GUI functions and syntax checks without the GUI package.
    sg = None


EEG_PYPE_VERSION = "v4.6.0"
APERIODIC_MODULE_VERSION = "0.1.3"
REQUIRED_SPECPARAM_VERSION = "2.0.0rc7"
DEFAULT_OUTPUT_FOLDER = "aperiodic_analysis"

# Recording-level metadata sidecar written by EEG-Pype.
EEG_PYPE_EXPORT_METADATA_FILENAME = "eeg_pype_export_metadata.json"
SUPPORTED_EEG_PYPE_METADATA_SCHEMA_VERSION = 1

DEFAULT_FIT_RANGE = (1.0, 30.0)
DEFAULT_BANDS = "delta:0.5-4; theta:4-8; alpha:8-13; beta:13-30"
LOGGER = logging.getLogger("eeg_aperiodic_analysis")

SUPPORTED_SOURCE_METHODS = ("beamformer", "sloreta", "eloreta", "dspm", "mne")
SUPPORTED_ATLASES = ("desikan", "bna", "aal2", "aal3")
SUPPORTED_REGION_SELECTIONS = ("cortical", "full")


# =============================================================================
# Typed data records
# =============================================================================


@dataclass(frozen=True)
class InventoryItem:
    path: str
    subject: str
    level: str
    profile_key: str
    profile_label: str
    representation: str
    epoch_number: Optional[int]
    source_method: str = ""
    atlas: str = ""
    region_selection: str = ""
    passband_low: Optional[float] = None
    passband_high: Optional[float] = None
    spectral_variant: str = "unfiltered"
    file_size: int = 0
    modified_time: float = 0.0

    # Recording-level EEG-Pype JSON sidecar.
    metadata_path: str = ""
    metadata_status: str = "missing"
    metadata_schema_version: Optional[int] = None
    metadata_eeg_pype_version: str = ""
    metadata_sampling_frequency_hz: Optional[float] = None
    metadata_epoch_length_seconds: Optional[float] = None
    metadata_output_representation: str = ""
    metadata_profile_available: bool = False
    metadata_base_highpass_hz: Optional[float] = None
    metadata_base_lowpass_hz: Optional[float] = None
    metadata_export_scaling_factor: Optional[float] = None
    metadata_output_unit: str = ""
    metadata_reference_scheme: str = ""
    metadata_reference_implementation: str = ""
    metadata_channel_names: tuple[str, ...] = ()


@dataclass
class SubjectBundle:
    subject: str
    profile_key: str
    profile_label: str
    level: str
    representation: str
    source_method: str
    atlas: str
    region_selection: str
    files: list[str]
    spectral_variant: str
    passband_low: Optional[float]
    passband_high: Optional[float]

    # Recording-level EEG-Pype JSON metadata.
    metadata_path: str = ""
    metadata_status: str = "missing"
    metadata_schema_version: Optional[int] = None
    metadata_eeg_pype_version: str = ""
    metadata_sampling_frequency_hz: Optional[float] = None
    metadata_epoch_length_seconds: Optional[float] = None
    metadata_output_representation: str = ""
    metadata_profile_available: bool = False
    metadata_trusted: bool = False
    metadata_base_highpass_hz: Optional[float] = None
    metadata_base_lowpass_hz: Optional[float] = None
    metadata_export_scaling_factor: Optional[float] = None
    metadata_output_unit: str = ""
    metadata_reference_scheme: str = ""
    metadata_reference_implementation: str = ""
    metadata_channel_names: tuple[str, ...] = ()

    warnings: list[str] = field(default_factory=list)


@dataclass
class PSDRecord:
    subject: str
    profile_key: str
    profile_label: str
    level: str
    representation: str
    source_method: str
    atlas: str
    region_selection: str
    spectral_variant: str
    passband_low: Optional[float]
    passband_high: Optional[float]
    frequencies: np.ndarray
    mean_psd: np.ndarray  # frequencies x channels/regions
    global_psd: np.ndarray
    channel_names: list[str]
    n_segments: int
    n_available_segments: int
    segment_length_seconds: float
    sfreq: float
    cache_path: str
    input_files: list[str]

    sampling_frequency_source: str = "manual setting"
    metadata_path: str = ""
    metadata_status: str = "missing"
    metadata_schema_version: Optional[int] = None
    metadata_eeg_pype_version: str = ""
    metadata_epoch_length_seconds: Optional[float] = None
    metadata_export_scaling_factor: Optional[float] = None
    metadata_output_unit: str = ""
    metadata_reference_scheme: str = ""
    metadata_reference_implementation: str = ""
    metadata_base_highpass_hz: Optional[float] = None
    metadata_base_lowpass_hz: Optional[float] = None

    warnings: list[str] = field(default_factory=list)
    global_channel_names: list[str] = field(default_factory=list)
    invalid_channel_reasons: dict[str, str] = field(default_factory=dict)


# =============================================================================
# General helpers
# =============================================================================


def safe_slug(value: str, max_length: int = 160) -> str:
    """Create a filesystem-safe but still readable identifier."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return (value or "unnamed")[:max_length]


def jsonable(value: Any) -> Any:
    """Recursively convert common scientific Python objects for JSON output."""
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json_dump(data: Any, path: str | Path) -> None:
    """Write JSON via a temporary file to avoid half-written manifests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(jsonable(data), file, indent=2, ensure_ascii=False)
    temp.replace(path)


def package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def collect_package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "EEG-Pype": EEG_PYPE_VERSION,
        "aperiodic_module": APERIODIC_MODULE_VERSION,
        "specparam": package_version("specparam"),
        "mne": package_version("mne"),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scipy": package_version("scipy"),
        "matplotlib": package_version("matplotlib"),
        "FreeSimpleGUI": package_version("FreeSimpleGUI"),
        "openpyxl": package_version("openpyxl"),
    }


def setup_logging(output_folder: str | Path) -> str:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    log_path = output_folder / "aperiodic_analysis.log"

    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    LOGGER.propagate = False

    LOGGER.info("EEG-Pype aperiodic analysis logging initialized")
    return str(log_path)


def gui_log(window: Any, message: str, level: str = "info") -> None:
    getattr(LOGGER, level, LOGGER.info)(message)
    if window is not None and "-LOG-" in window.AllKeysDict:
        window["-LOG-"].update(message + "\n", append=True)
        window.refresh()


@functools.lru_cache(maxsize=1)
def ensure_specparam_rc7() -> tuple[Any, str]:
    """Import and strictly version-check Specparam once per Python process."""
    try:
        version = importlib.metadata.version("specparam")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Specparam is not installed. Install the tested version with:\n"
            "pip install specparam==2.0.0rc7"
        ) from exc

    if version != REQUIRED_SPECPARAM_VERSION:
        raise RuntimeError(
            f"This script targets specparam=={REQUIRED_SPECPARAM_VERSION}, but {version} is installed.\n"
            f"Install the tested version with:\npip install specparam=={REQUIRED_SPECPARAM_VERSION}"
        )

    from specparam import SpectralModel

    return SpectralModel, version


# =============================================================================
# EEG-Pype filename parsing and inventory
# =============================================================================


def _strip_epoch_suffix(stem: str) -> tuple[str, Optional[int]]:
    match = re.search(r"_Epoch_?(\d+)$", stem, flags=re.IGNORECASE)
    if not match:
        return stem, None
    return stem[: match.start()], int(match.group(1))


def _strip_passband_suffix(stem: str) -> tuple[str, Optional[float], Optional[float]]:
    """Remove EEG-Pype numerical passband suffix from a filename stem."""
    pattern = re.compile(
        r"_(?P<low>\d+(?:\.\d+)?)-(?P<high>\d+(?:\.\d+)?)\s*_?Hz$",
        flags=re.IGNORECASE,
    )
    match = pattern.search(stem)
    if not match:
        return stem, None, None
    return stem[: match.start()], float(match.group("low")), float(match.group("high"))


def _positive_finite_float(value: Any) -> Optional[float]:
    """Convert a value to a positive finite float, otherwise return None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _nonnegative_finite_float(value: Any) -> Optional[float]:
    """Convert a value to a non-negative finite float, otherwise None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _metadata_is_trusted(
    status: str,
    schema_version: Optional[int],
    profile_available: bool,
) -> bool:
    """Return whether sidecar values may configure the analysis."""
    return (
        status == "complete"
        and schema_version == SUPPORTED_EEG_PYPE_METADATA_SCHEMA_VERSION
        and profile_available
    )


def _profile_reference(profile: dict[str, Any]) -> tuple[str, str]:
    """Extract sensor-reference metadata from a sensor or source profile."""
    reference = profile.get("reference")
    if not isinstance(reference, dict):
        reference = profile.get("sensor_reference_before_source_reconstruction")
    if not isinstance(reference, dict):
        return "", ""
    return (
        str(reference.get("scheme", "") or ""),
        str(reference.get("implementation", "") or ""),
    )


@functools.lru_cache(maxsize=None)
def load_eeg_pype_export_metadata(
    folder: str | Path,
) -> tuple[dict[str, Any], str, str]:
    """Load one recording-level EEG-Pype metadata sidecar.

    Parameters
    ----------
    folder
        Directory containing the EEG-Pype TXT exports and, when available,
        ``eeg_pype_export_metadata.json``.

    Returns
    -------
    metadata
        Parsed JSON dictionary. Empty when the sidecar is missing or invalid.
    metadata_path
        Absolute sidecar path, or an empty string when no sidecar exists.
    metadata_status
        One of: missing, complete, in_progress, unknown, invalid_json,
        invalid_structure or unreadable.

    Notes
    -----
    The function is cached because one recording can contain many epoch TXT
    files that all refer to the same JSON sidecar.
    """
    folder = Path(folder)
    metadata_path = folder / EEG_PYPE_EXPORT_METADATA_FILENAME

    if not metadata_path.is_file():
        return {}, "", "missing"

    resolved_path = str(metadata_path.resolve())

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError:
        return {}, resolved_path, "invalid_json"
    except OSError:
        return {}, resolved_path, "unreadable"

    if not isinstance(metadata, dict):
        return {}, resolved_path, "invalid_structure"

    status = str(
        metadata.get("status", "unknown")
    ).strip().lower()

    if not status:
        status = "unknown"

    return metadata, resolved_path, status


def parse_eeg_pype_txt(path: str | Path) -> Optional[InventoryItem]:
    """Parse current and legacy EEG-Pype output names without assuming simple subject IDs."""
    path = Path(path)
    if path.suffix.lower() != ".txt" or path.name.startswith((".", "._")):
        return None

    stem, epoch_number = _strip_epoch_suffix(path.stem)
    stem, passband_low, passband_high = _strip_passband_suffix(stem)
    representation = "epoched" if epoch_number is not None else "continuous"

    subject = ""
    level = ""
    source_method = ""
    atlas = ""
    region_selection = ""

    sensor_match = re.match(r"^(?P<subject>.+)_Sensor_level$", stem, flags=re.IGNORECASE)
    legacy_source_match = re.match(r"^(?P<subject>.+)_Source_level$", stem, flags=re.IGNORECASE)
    source_match = re.match(
        r"^(?P<subject>.+)_Source_(?P<method>beamformer|sLORETA|eLORETA|dSPM|MNE)_"
        r"(?P<atlas>desikan|bna|aal2|aal3)_(?P<region>cortical|full)$",
        stem,
        flags=re.IGNORECASE,
    )
    source_atlas_match = re.match(
        r"^(?P<subject>.+)_Source_(?P<atlas>desikan|bna|aal2|aal3)_"
        r"(?P<region>cortical|full)$",
        stem,
        flags=re.IGNORECASE,
    )

    if sensor_match:
        subject = sensor_match.group("subject")
        level = "sensor"
        profile_key = "sensor"
        profile_label = "Sensor level"
    elif source_match:
        subject = source_match.group("subject")
        level = "source"
        source_method = source_match.group("method")
        atlas = source_match.group("atlas").lower()
        region_selection = source_match.group("region").lower()
        profile_key = f"source_{source_method}_{atlas}_{region_selection}"
        profile_label = f"Source: {source_method} – {atlas} – {region_selection}"
    elif source_atlas_match:
        subject = source_atlas_match.group("subject")
        level = "source"
        atlas = source_atlas_match.group("atlas").lower()
        region_selection = source_atlas_match.group("region").lower()
        profile_key = f"source_{atlas}_{region_selection}"
        profile_label = f"Source: {atlas} – {region_selection}"
    elif legacy_source_match:
        subject = legacy_source_match.group("subject")
        level = "source"
        profile_key = "source_legacy"
        profile_label = "Source level (legacy naming)"
    else:
        return None

    if passband_low is None:
        spectral_variant = "unfiltered"
    else:
        spectral_variant = f"{passband_low:g}-{passband_high:g} Hz"

    # Look for the recording-level JSON sidecar in the same directory
    # as the TXT export.
    metadata, metadata_path, metadata_status = (
        load_eeg_pype_export_metadata(path.parent)
    )

    metadata_schema_version = metadata.get("schema_version")
    try:
        metadata_schema_version = (
            int(metadata_schema_version)
            if metadata_schema_version is not None
            else None
        )
    except (TypeError, ValueError):
        metadata_schema_version = None

    metadata_eeg_pype_version = str(
        metadata.get("eeg_pype_version", "") or ""
    )

    metadata_output = metadata.get("output", {})
    if not isinstance(metadata_output, dict):
        metadata_output = {}

    metadata_sampling_frequency_hz = _positive_finite_float(
        metadata_output.get("sampling_frequency_hz")
    )
    metadata_epoch_length_seconds = _positive_finite_float(
        metadata_output.get("epoch_length_seconds")
    )
    metadata_output_representation = str(
        metadata_output.get("representation", "") or ""
    ).strip().lower()

    metadata_profiles = metadata.get("profiles", {})
    if not isinstance(metadata_profiles, dict):
        metadata_profiles = {}

    metadata_profile = metadata_profiles.get(profile_key, {})
    if not isinstance(metadata_profile, dict):
        metadata_profile = {}

    metadata_profile_available = bool(metadata_profile)

    metadata_base_highpass_hz = _nonnegative_finite_float(
        metadata_profile.get(
            "base_highpass_hz",
            metadata_output.get("base_highpass_hz"),
        )
    )
    metadata_base_lowpass_hz = _positive_finite_float(
        metadata_profile.get(
            "base_lowpass_hz",
            metadata_output.get("base_lowpass_hz"),
        )
    )

    export_scaling = metadata_profile.get("export_scaling", {})
    if not isinstance(export_scaling, dict):
        export_scaling = {}

    metadata_export_scaling_factor = _positive_finite_float(
        export_scaling.get("factor")
    )
    metadata_output_unit = str(
        export_scaling.get("output_unit", "") or ""
    )

    (
        metadata_reference_scheme,
        metadata_reference_implementation,
    ) = _profile_reference(metadata_profile)

    raw_metadata_channel_names = metadata_profile.get(
        "channel_names",
        metadata_profile.get("region_names", []),
    )
    if isinstance(raw_metadata_channel_names, (list, tuple)):
        metadata_channel_names = tuple(
            str(name) for name in raw_metadata_channel_names
        )
    else:
        metadata_channel_names = ()

    stat = path.stat()

    return InventoryItem(
        path=str(path.resolve()),
        subject=subject,
        level=level,
        profile_key=profile_key,
        profile_label=profile_label,
        representation=representation,
        epoch_number=epoch_number,
        source_method=source_method,
        atlas=atlas,
        region_selection=region_selection,
        passband_low=passband_low,
        passband_high=passband_high,
        spectral_variant=spectral_variant,
        file_size=stat.st_size,
        modified_time=stat.st_mtime,
        metadata_path=metadata_path,
        metadata_status=metadata_status,
        metadata_schema_version=metadata_schema_version,
        metadata_eeg_pype_version=metadata_eeg_pype_version,
        metadata_sampling_frequency_hz=metadata_sampling_frequency_hz,
        metadata_epoch_length_seconds=metadata_epoch_length_seconds,
        metadata_output_representation=metadata_output_representation,
        metadata_profile_available=metadata_profile_available,
        metadata_base_highpass_hz=metadata_base_highpass_hz,
        metadata_base_lowpass_hz=metadata_base_lowpass_hz,
        metadata_export_scaling_factor=metadata_export_scaling_factor,
        metadata_output_unit=metadata_output_unit,
        metadata_reference_scheme=metadata_reference_scheme,
        metadata_reference_implementation=(
            metadata_reference_implementation
        ),
        metadata_channel_names=metadata_channel_names,
    )


def scan_eeg_pype_outputs(root_folder: str | Path) -> list[InventoryItem]:
    # A sidecar may have changed from in_progress to complete since an
    # earlier scan in the same GUI session.
    load_eeg_pype_export_metadata.cache_clear()

    root = Path(root_folder)
    if not root.is_dir():
        raise ValueError(f"Input folder does not exist: {root}")

    output_folder_name = DEFAULT_OUTPUT_FOLDER.lower()
    items: list[InventoryItem] = []
    for path in root.rglob("*.txt"):
        lower_parts = {part.lower() for part in path.parts}
        if output_folder_name in lower_parts:
            continue
        item = parse_eeg_pype_txt(path)
        if item is not None:
            items.append(item)

    items.sort(
        key=lambda item: (
            item.profile_key.lower(),
            item.representation,
            item.subject.lower(),
            item.spectral_variant,
            item.epoch_number or 0,
            item.path.lower(),
        )
    )
    return items


def find_representation_overlaps(
    items: list[InventoryItem],
) -> list[dict[str, str]]:
    """Find subject/profile/variant combinations saved both ways."""
    grouped: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for item in items:
        grouped[
            (item.subject, item.profile_key, item.spectral_variant)
        ].add(item.representation)

    overlaps: list[dict[str, str]] = []
    for (subject, profile_key, spectral_variant), representations in sorted(
        grouped.items(),
        key=lambda entry: tuple(str(value).lower() for value in entry[0]),
    ):
        if {"epoched", "continuous"}.issubset(representations):
            overlaps.append(
                {
                    "subject": subject,
                    "profile_key": profile_key,
                    "spectral_variant": spectral_variant,
                }
            )
    return overlaps


def trusted_inventory_sampling_frequencies(
    items: list[InventoryItem],
) -> list[float]:
    """Return unique trusted sidecar sampling frequencies."""
    return sorted(
        {
            float(item.metadata_sampling_frequency_hz)
            for item in items
            if _metadata_is_trusted(
                item.metadata_status,
                item.metadata_schema_version,
                item.metadata_profile_available,
            )
            and item.metadata_sampling_frequency_hz is not None
        }
    )


def trusted_inventory_epoch_lengths(
    items: list[InventoryItem],
) -> list[float]:
    """Return unique trusted sidecar epoch lengths."""
    return sorted(
        {
            float(item.metadata_epoch_length_seconds)
            for item in items
            if _metadata_is_trusted(
                item.metadata_status,
                item.metadata_schema_version,
                item.metadata_profile_available,
            )
            and item.metadata_epoch_length_seconds is not None
        }
    )


def profile_id(item: InventoryItem) -> str:
    return f"{item.profile_key}||{item.representation}"


def profile_display(item: InventoryItem) -> str:
    return f"{item.profile_label} | {item.representation}"


def summarize_profiles(items: list[InventoryItem]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for item in items:
        key = profile_id(item)
        if key not in profiles:
            profiles[key] = {
                "id": key,
                "label": profile_display(item),
                "profile_key": item.profile_key,
                "profile_label": item.profile_label,
                "level": item.level,
                "representation": item.representation,
                "source_method": item.source_method,
                "atlas": item.atlas,
                "region_selection": item.region_selection,
                "subjects": set(),
                "variants": defaultdict(set),
                "n_files": 0,
            }
        profiles[key]["subjects"].add(item.subject)
        profiles[key]["variants"][item.spectral_variant].add(item.subject)
        profiles[key]["n_files"] += 1
    return profiles


def inventory_summary_text(
    items: list[InventoryItem],
    config_info: dict[str, Any],
) -> str:
    if not items:
        return "No valid EEG-Pype sensor/source .txt outputs found."

    profiles = summarize_profiles(items)
    lines = [
        f"Found {len(items)} parsed .txt files across "
        f"{len(profiles)} analysis profiles."
    ]
    for profile in sorted(
        profiles.values(),
        key=lambda value: (
            value["representation"] != "epoched",
            value["label"].lower(),
        ),
    ):
        variants = ", ".join(
            f"{name} ({len(subjects)} subjects)"
            for name, subjects in sorted(profile["variants"].items())
        )
        lines.append(
            f"• {profile['label']}: {len(profile['subjects'])} subjects, "
            f"{profile['n_files']} files; {variants}"
        )

    metadata_complete = sum(
        item.metadata_status == "complete" for item in items
    )
    metadata_missing = sum(
        item.metadata_status == "missing" for item in items
    )
    metadata_other = len(items) - metadata_complete - metadata_missing
    lines.append(
        "JSON sidecars across parsed files: "
        f"{metadata_complete} complete, {metadata_missing} missing, "
        f"{metadata_other} other/invalid."
    )

    sfreqs = trusted_inventory_sampling_frequencies(items)
    epoch_lengths = trusted_inventory_epoch_lengths(items)
    if sfreqs:
        lines.append(
            "Trusted JSON sampling frequency/frequencies: "
            + ", ".join(f"{value:g} Hz" for value in sfreqs)
        )
    if epoch_lengths:
        lines.append(
            "Trusted JSON epoch length(s): "
            + ", ".join(f"{value:g} s" for value in epoch_lengths)
        )

    overlaps = find_representation_overlaps(items)
    if overlaps:
        lines.append(
            f"WARNING: {len(overlaps)} subject/profile/variant combination(s) "
            "exist as both epoched and continuous output. They remain separate "
            "profiles; epoched profiles are listed first."
        )
        for overlap in overlaps[:10]:
            lines.append(
                "  - "
                f"{overlap['subject']} | {overlap['profile_key']} | "
                f"{overlap['spectral_variant']}"
            )
        if len(overlaps) > 10:
            lines.append(f"  - ... and {len(overlaps) - 10} more")

    if config_info.get("config_path"):
        lines.append(f"Configuration: {config_info['config_path']}")
        lines.append(
            f"Detected sampling frequency: "
            f"{config_info.get('sfreq', 'unknown')} Hz; "
            f"epoch length: {config_info.get('epoch_length', 'unknown')} s; "
            f"EEG-Pype: {config_info.get('eeg_version', 'unknown')}"
        )
    else:
        lines.append(
            "No preprocessing .pkl configuration was detected; "
            "trusted JSON metadata or manual fallback settings will be used."
        )
    return "\n".join(lines)


# =============================================================================
# Preprocessing metadata
# =============================================================================


def find_preprocessing_config(root_folder: str | Path) -> dict[str, Any]:
    root = Path(root_folder)
    candidates = [path for path in root.rglob("*.pkl") if DEFAULT_OUTPUT_FOLDER.lower() not in {p.lower() for p in path.parts}]
    if not candidates:
        return {}

    # Prefer a config in the selected root, then the most recently modified candidate.
    candidates.sort(key=lambda path: (path.parent != root, -path.stat().st_mtime))
    errors: list[str] = []
    for candidate in candidates:
        try:
            with candidate.open("rb") as file:
                cfg = pickle.load(file)
            if not isinstance(cfg, dict):
                continue
            sfreq = cfg.get("downsampled_sample_frequency", cfg.get("sample_frequency"))
            epoch_length = cfg.get("epoch_length", cfg.get("default_epoch_length"))
            eeg_version = cfg.get("EEG_version", cfg.get("eeg_version", "unknown"))
            return {
                "config_path": str(candidate.resolve()),
                "sfreq": float(sfreq) if sfreq not in (None, "", " ") else None,
                "epoch_length": float(epoch_length) if epoch_length not in (None, "", " ") else None,
                "eeg_version": eeg_version,
                "raw_config": cfg,
            }
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    return {"errors": errors}


# =============================================================================
# Profile selection and validation
# =============================================================================


def group_items_for_profile(items: list[InventoryItem], selected_profile_id: str) -> dict[str, dict[str, list[InventoryItem]]]:
    grouped: dict[str, dict[str, list[InventoryItem]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        if profile_id(item) == selected_profile_id:
            grouped[item.subject][item.spectral_variant].append(item)
    return grouped


def _effective_item_passband(
    item: InventoryItem,
) -> tuple[Optional[float], Optional[float]]:
    """Return filename passband, or trusted base passband from JSON."""
    if item.passband_low is not None and item.passband_high is not None:
        return item.passband_low, item.passband_high

    if _metadata_is_trusted(
        item.metadata_status,
        item.metadata_schema_version,
        item.metadata_profile_available,
    ):
        return (
            item.metadata_base_highpass_hz,
            item.metadata_base_lowpass_hz,
        )

    # Legacy files without sidecars retain historical behaviour: a filename
    # without a passband suffix is treated as unfiltered.
    return None, None


def _variant_covers_fit(
    items: list[InventoryItem],
    fit_min: float,
    fit_max: float,
) -> bool:
    low, high = _effective_item_passband(items[0])
    if low is None or high is None:
        return True
    return low <= fit_min and high >= fit_max


def _single_value(
    values: list[Any],
    default: Any = None,
) -> tuple[Any, bool]:
    """Return the one unique value and whether the values were consistent."""
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    if len(unique) == 1:
        return unique[0], True
    return default, len(unique) <= 1


def summarize_selected_metadata(
    selected_items: list[InventoryItem],
    profile_key: str,
    representation: str,
) -> dict[str, Any]:
    """Summarize and validate metadata for selected TXT files."""
    warnings: list[str] = []

    metadata_paths = sorted(
        {
            item.metadata_path
            for item in selected_items
            if item.metadata_path
        }
    )
    statuses = [item.metadata_status for item in selected_items]
    schema_versions = [
        item.metadata_schema_version for item in selected_items
    ]
    profile_flags = [
        item.metadata_profile_available for item in selected_items
    ]

    metadata_status, status_consistent = _single_value(
        statuses,
        "mixed",
    )
    metadata_schema_version, schema_consistent = _single_value(
        schema_versions,
        None,
    )
    metadata_profile_available = bool(
        profile_flags and all(profile_flags)
    )
    metadata_path = (
        metadata_paths[0] if len(metadata_paths) == 1 else ""
    )

    if not metadata_paths:
        warnings.append(
            "No EEG-Pype JSON metadata sidecar was found. "
            "Manual/legacy settings will be used."
        )
    elif len(metadata_paths) > 1:
        warnings.append(
            "Selected TXT files refer to multiple EEG-Pype metadata "
            "sidecars. Metadata will not configure this subject."
        )

    if not status_consistent:
        warnings.append(
            "Selected TXT files have inconsistent metadata statuses."
        )
    elif metadata_status == "in_progress":
        warnings.append(
            "The EEG-Pype metadata sidecar is marked 'in_progress'; "
            "the preprocessing export may be incomplete."
        )
    elif metadata_status not in {"complete", "missing"}:
        warnings.append(
            "The EEG-Pype metadata sidecar could not be fully validated "
            f"(status: {metadata_status})."
        )

    if not schema_consistent:
        warnings.append(
            "Selected TXT files have inconsistent metadata schema versions."
        )
    elif (
        metadata_schema_version is not None
        and metadata_schema_version
        != SUPPORTED_EEG_PYPE_METADATA_SCHEMA_VERSION
    ):
        warnings.append(
            "EEG-Pype metadata schema version "
            f"{metadata_schema_version} was found, but this module "
            f"supports version "
            f"{SUPPORTED_EEG_PYPE_METADATA_SCHEMA_VERSION}."
        )

    if metadata_status == "complete" and not metadata_profile_available:
        warnings.append(
            "The sidecar does not contain the selected profile "
            f"'{profile_key}'. Filename-derived information will be used."
        )

    trusted = (
        len(metadata_paths) == 1
        and status_consistent
        and schema_consistent
        and _metadata_is_trusted(
            str(metadata_status),
            metadata_schema_version,
            metadata_profile_available,
        )
    )

    fields = {
        "metadata_eeg_pype_version": "",
        "metadata_sampling_frequency_hz": None,
        "metadata_epoch_length_seconds": None,
        "metadata_output_representation": "",
        "metadata_base_highpass_hz": None,
        "metadata_base_lowpass_hz": None,
        "metadata_export_scaling_factor": None,
        "metadata_output_unit": "",
        "metadata_reference_scheme": "",
        "metadata_reference_implementation": "",
        "metadata_channel_names": (),
    }

    for field_name in fields:
        values = [getattr(item, field_name) for item in selected_items]
        value, consistent = _single_value(values, fields[field_name])
        fields[field_name] = value
        if not consistent:
            warnings.append(
                f"Selected TXT files have inconsistent {field_name} values."
            )
            trusted = False

    metadata_representation = fields["metadata_output_representation"]
    if (
        trusted
        and metadata_representation
        and metadata_representation != representation
    ):
        warnings.append(
            "Representation mismatch: filenames indicate "
            f"'{representation}', while the JSON reports "
            f"'{metadata_representation}'. Metadata will not configure "
            "this subject."
        )
        trusted = False

    if trusted and fields["metadata_sampling_frequency_hz"] is None:
        warnings.append(
            "The completed JSON sidecar has no valid output sampling "
            "frequency. Manual fallback will be used."
        )
        trusted = False

    return {
        "metadata_path": metadata_path,
        "metadata_status": str(metadata_status),
        "metadata_schema_version": metadata_schema_version,
        "metadata_profile_available": metadata_profile_available,
        "metadata_trusted": trusted,
        **fields,
        "warnings": warnings,
    }


def choose_subject_bundle(
    subject: str,
    variants: dict[str, list[InventoryItem]],
    fit_min: float,
    fit_max: float,
) -> SubjectBundle:
    """Prefer unfiltered input; otherwise use the widest valid passband."""
    candidates: list[
        tuple[tuple[int, float], str, list[InventoryItem]]
    ] = []
    all_variant_descriptions: list[str] = []

    for variant, variant_items in variants.items():
        first = variant_items[0]
        all_variant_descriptions.append(variant)
        if not _variant_covers_fit(
            variant_items,
            fit_min,
            fit_max,
        ):
            continue

        low, high = _effective_item_passband(first)
        if first.passband_low is None:
            priority = (0, -math.inf)
        else:
            width = (
                float(high - low)
                if low is not None and high is not None
                else 0.0
            )
            priority = (1, -width)
        candidates.append((priority, variant, variant_items))

    if not candidates:
        raise ValueError(
            f"{subject}: no spectral variant covers "
            f"{fit_min:g}-{fit_max:g} Hz. Available: "
            f"{', '.join(sorted(all_variant_descriptions))}"
        )

    candidates.sort(key=lambda entry: entry[0])
    _, selected_variant, selected_items = candidates[0]
    selected_items = sorted(
        selected_items,
        key=lambda item: (item.epoch_number or 0, item.path),
    )
    first = selected_items[0]
    warnings: list[str] = []

    if first.representation == "epoched":
        epoch_paths: dict[int, list[str]] = defaultdict(list)
        for item in selected_items:
            if item.epoch_number is not None:
                epoch_paths[item.epoch_number].append(item.path)
        duplicate_epochs = {
            number: paths
            for number, paths in epoch_paths.items()
            if len(paths) > 1
        }
        if duplicate_epochs:
            duplicate_text = "; ".join(
                f"Epoch {number}: {len(paths)} files"
                for number, paths in sorted(duplicate_epochs.items())
            )
            raise ValueError(
                f"{subject}: duplicate epoch numbers were found for "
                f"{selected_variant} ({duplicate_text}). Keep only one "
                "preprocessing export per recording."
            )

    if (
        first.representation == "continuous"
        and len(selected_items) > 1
    ):
        warnings.append(
            "Multiple continuous files found for "
            f"{selected_variant}; using {selected_items[0].path}."
        )
        selected_items = [selected_items[0]]

    metadata_summary = summarize_selected_metadata(
        selected_items,
        first.profile_key,
        first.representation,
    )
    warnings.extend(metadata_summary.pop("warnings"))

    effective_low, effective_high = _effective_item_passband(first)
    if effective_low is not None and effective_high is not None:
        edge_distance = min(
            fit_min - effective_low,
            effective_high - fit_max,
        )
        if edge_distance < 1.0:
            warnings.append(
                f"Fit range is within {edge_distance:.2f} Hz of the "
                f"effective filter edge ({effective_low:g}-"
                f"{effective_high:g} Hz)."
            )

    return SubjectBundle(
        subject=subject,
        profile_key=first.profile_key,
        profile_label=first.profile_label,
        level=first.level,
        representation=first.representation,
        source_method=first.source_method,
        atlas=first.atlas,
        region_selection=first.region_selection,
        files=[item.path for item in selected_items],
        spectral_variant=selected_variant,
        passband_low=effective_low,
        passband_high=effective_high,
        warnings=warnings,
        **metadata_summary,
    )


def build_subject_bundles(
    items: list[InventoryItem], selected_profile_id: str, fit_min: float, fit_max: float
) -> tuple[list[SubjectBundle], list[str]]:
    grouped = group_items_for_profile(items, selected_profile_id)
    bundles: list[SubjectBundle] = []
    errors: list[str] = []
    for subject in sorted(grouped, key=str.lower):
        try:
            bundles.append(choose_subject_bundle(subject, grouped[subject], fit_min, fit_max))
        except Exception as exc:
            errors.append(str(exc))
    return bundles, errors


# =============================================================================
# TXT reading, continuous windowing and multitaper PSD
# =============================================================================


def _row_looks_numeric(row: pd.DataFrame) -> bool:
    try:
        pd.to_numeric(row.iloc[0], errors="raise")
        return True
    except Exception:
        return False


def _read_csv_tab_first(
    path: Path, *, header: Optional[int] | str, nrows: Optional[int] = None
) -> tuple[pd.DataFrame, bool]:
    """Read EEG-Pype TXT deterministically, falling back only when no tab is present.

    EEG-Pype writes tab-separated files. Legacy or externally edited files may use
    another delimiter, in which case pandas' delimiter sniffing is used as a
    compatibility fallback.
    """
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        sample = handle.read(8192)

    # EEG-Pype output is tab-separated, including legitimate one-channel files
    # that contain no literal tab. Only invoke delimiter sniffing when the file
    # contains clear evidence of another delimiter.
    alternate_delimiter = any(delimiter in sample for delimiter in (",", ";", "|"))
    if not alternate_delimiter and "\t" not in sample:
        for line in (line.strip() for line in sample.splitlines()[:10] if line.strip()):
            tokens = line.split()
            if len(tokens) > 1:
                numeric_tokens = 0
                for token in tokens:
                    try:
                        float(token)
                        numeric_tokens += 1
                    except ValueError:
                        pass
                if numeric_tokens >= 2:
                    alternate_delimiter = True
                    break

    if alternate_delimiter and "\t" not in sample:
        frame = pd.read_csv(path, sep=None, engine="python", header=header, nrows=nrows)
        return frame, True

    frame = pd.read_csv(path, sep="\t", engine="c", header=header, nrows=nrows)
    return frame, False


def read_eeg_txt(path: str | Path, header_mode: str = "Auto") -> tuple[np.ndarray, list[str], bool]:
    """Read an EEG-Pype table as samples x channels and validate finite numeric data."""
    path = Path(path)
    mode = header_mode.strip().lower()
    if mode not in {"auto", "yes", "no"}:
        raise ValueError(f"Unknown header mode: {header_mode}")

    if mode == "auto":
        first_row, fallback_used = _read_csv_tab_first(path, header=None, nrows=1)
        has_header = not _row_looks_numeric(first_row)
    else:
        has_header = mode == "yes"
        fallback_used = False

    if has_header:
        frame, full_fallback = _read_csv_tab_first(path, header="infer")
        channel_names = [str(column).strip() for column in frame.columns]
    else:
        frame, full_fallback = _read_csv_tab_first(path, header=None)
        channel_names = [f"Channel_{index + 1}" for index in range(frame.shape[1])]

    fallback_used = fallback_used or full_fallback
    if fallback_used:
        LOGGER.warning("No tab delimiter found in %s; used delimiter sniffing fallback", path)

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    data = frame.to_numpy(dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1:
        raise ValueError(f"Invalid data dimensions {data.shape} in {path}")
    if not np.all(np.isfinite(data)):
        bad_count = int(np.size(data) - np.count_nonzero(np.isfinite(data)))
        raise ValueError(f"Found {bad_count} nonfinite/non-numeric values in {path}")
    if len(set(channel_names)) != len(channel_names):
        raise ValueError(f"Duplicate channel/region names in {path}")
    return data, channel_names, has_header


def select_indices(n_available: int, limit: bool, maximum: int, seed: int) -> list[int]:
    if n_available < 1:
        return []
    if not limit or maximum >= n_available:
        return list(range(n_available))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_available), maximum))


def load_segments(
    bundle: SubjectBundle,
    settings: dict[str, Any],
) -> tuple[list[np.ndarray], list[str], dict[str, Any]]:
    header_mode = settings["header_mode"]
    limit_segments = settings["limit_segments"]
    max_segments = settings["max_segments"]
    seed = settings["segment_seed"]
    sfreq = float(settings["sfreq"])
    sfreq_source = str(
        settings.get("sfreq_source", "manual setting")
    )

    warnings = list(bundle.warnings)
    manual_sfreq = float(settings.get("manual_sfreq", sfreq))
    if (
        sfreq_source == "EEG-Pype JSON"
        and not math.isclose(
            sfreq,
            manual_sfreq,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
    ):
        warnings.append(
            "Using EEG-Pype JSON sampling frequency "
            f"({sfreq:g} Hz) instead of the manual fallback "
            f"({manual_sfreq:g} Hz)."
        )

    def validate_channel_metadata(
        channel_names: list[str],
        has_header: bool,
    ) -> None:
        expected = list(bundle.metadata_channel_names)
        if not bundle.metadata_trusted or not expected:
            return
        if len(channel_names) != len(expected):
            warnings.append(
                "TXT channel/region count does not match the JSON profile "
                f"({len(channel_names)} versus {len(expected)})."
            )
            return
        if has_header and channel_names != expected:
            if set(channel_names) == set(expected):
                warnings.append(
                    "TXT and JSON contain the same channel/region names "
                    "in a different order; the TXT header order is used."
                )
            else:
                warnings.append(
                    "TXT channel/region names do not match the JSON profile."
                )

    if bundle.representation == "epoched":
        selected_indices = select_indices(
            len(bundle.files),
            limit_segments,
            max_segments,
            seed,
        )
        selected_files = [
            bundle.files[index] for index in selected_indices
        ]
        segments: list[np.ndarray] = []
        channel_names: Optional[list[str]] = None
        n_samples: Optional[int] = None
        detected_headers: set[bool] = set()

        for path in selected_files:
            data, names, has_header = read_eeg_txt(path, header_mode)
            detected_headers.add(has_header)
            if channel_names is None:
                channel_names = names
                n_samples = data.shape[0]
                validate_channel_metadata(names, has_header)
            else:
                if names != channel_names:
                    raise ValueError(
                        f"Inconsistent channel/region order in {path}"
                    )
                if data.shape[0] != n_samples:
                    raise ValueError(
                        f"Variable epoch length in {path}: "
                        f"{data.shape[0]} vs {n_samples}"
                    )
            segments.append(
                signal.detrend(data, axis=0, type="linear")
            )

        if segments and bundle.metadata_trusted:
            actual_length = segments[0].shape[0] / sfreq
            expected_length = bundle.metadata_epoch_length_seconds
            if (
                expected_length is not None
                and not math.isclose(
                    actual_length,
                    expected_length,
                    rel_tol=0,
                    abs_tol=max(1.0 / sfreq, 1e-9),
                )
            ):
                warnings.append(
                    "Epoch-length mismatch: TXT samples imply "
                    f"{actual_length:g} s, while the JSON reports "
                    f"{expected_length:g} s."
                )

        metadata = {
            "n_available_segments": len(bundle.files),
            "n_selected_segments": len(segments),
            "selected_files": selected_files,
            "detected_headers": sorted(detected_headers),
            "incomplete_samples_excluded": 0,
            "analysis_sampling_frequency_hz": sfreq,
            "sampling_frequency_source": sfreq_source,
            "manual_fallback_sampling_frequency_hz": manual_sfreq,
            "eeg_pype_metadata_sampling_frequency_hz": (
                bundle.metadata_sampling_frequency_hz
            ),
        }
        return (
            segments,
            channel_names or [],
            {**metadata, "warnings": warnings},
        )

    # Continuous input: load once, then create fixed-length overlapping windows.
    data, channel_names, has_header = read_eeg_txt(
        bundle.files[0],
        header_mode,
    )
    validate_channel_metadata(channel_names, has_header)

    window_samples = int(
        round(settings["window_length"] * sfreq)
    )
    if window_samples < 2:
        raise ValueError(
            "Continuous window length corresponds to fewer than 2 samples"
        )
    if window_samples > data.shape[0]:
        raise ValueError(
            f"Continuous file for {bundle.subject} is shorter than one "
            f"{settings['window_length']:g} s window"
        )

    step = int(
        round(
            window_samples
            * (1.0 - settings["window_overlap"] / 100.0)
        )
    )
    step = max(step, 1)
    starts = list(
        range(0, data.shape[0] - window_samples + 1, step)
    )
    selected_indices = select_indices(
        len(starts),
        limit_segments,
        max_segments,
        seed,
    )
    selected_starts = [
        starts[index] for index in selected_indices
    ]
    segments = [
        signal.detrend(
            data[start : start + window_samples, :],
            axis=0,
            type="linear",
        )
        for start in selected_starts
    ]

    last_complete_end = (
        starts[-1] + window_samples if starts else 0
    )
    incomplete = max(0, data.shape[0] - last_complete_end)
    if incomplete:
        warnings.append(
            f"Excluded {incomplete} trailing samples that did not "
            "form a complete window."
        )

    metadata = {
        "n_available_segments": len(starts),
        "n_selected_segments": len(segments),
        "selected_files": [bundle.files[0]],
        "selected_window_starts": selected_starts,
        "detected_headers": [has_header],
        "incomplete_samples_excluded": incomplete,
        "analysis_sampling_frequency_hz": sfreq,
        "sampling_frequency_source": sfreq_source,
        "manual_fallback_sampling_frequency_hz": manual_sfreq,
        "eeg_pype_metadata_sampling_frequency_hz": (
            bundle.metadata_sampling_frequency_hz
        ),
        "warnings": warnings,
    }
    return segments, channel_names, metadata


def multitaper_characteristics(
    segment_length: float,
    bandwidth: Optional[float],
    low_bias: bool = True,
) -> dict[str, float | int | str]:
    """Return MNE-compatible multitaper smoothing characteristics."""
    if segment_length <= 0:
        raise ValueError("Segment length must be positive")

    bin_spacing = 1.0 / segment_length
    effective_bandwidth = (
        8.0 / segment_length
        if bandwidth is None
        else float(bandwidth)
    )
    minimum_bandwidth = bin_spacing

    half_nbw = effective_bandwidth * segment_length / 2.0
    maximum_tapers = max(1, int(2.0 * half_nbw))
    approximate_usable_tapers = (
        max(1, maximum_tapers - 1)
        if low_bias
        else maximum_tapers
    )

    return {
        "label": "Auto" if bandwidth is None else f"{bandwidth:g} Hz",
        "bin_spacing": bin_spacing,
        "effective_bandwidth": effective_bandwidth,
        "half_smoothing": effective_bandwidth / 2.0,
        "minimum_bandwidth": minimum_bandwidth,
        "maximum_tapers": maximum_tapers,
        "approximate_usable_tapers": approximate_usable_tapers,
    }


def compute_resolution_summary(
    sfreq: float,
    segment_length: float,
    bandwidth: Optional[float],
) -> str:
    if sfreq <= 0 or segment_length <= 0:
        return (
            "Enter a valid sampling frequency and epoch/window length."
        )

    n_samples = int(round(sfreq * segment_length))
    if n_samples < 2:
        return "Epoch/window contains fewer than two samples."

    characteristics = multitaper_characteristics(
        segment_length,
        bandwidth,
        low_bias=True,
    )
    return (
        f"{n_samples} samples; bin spacing "
        f"{characteristics['bin_spacing']:.3f} Hz; bandwidth "
        f"{characteristics['label']} "
        f"({characteristics['effective_bandwidth']:.3f} Hz full "
        f"smoothing, ±{characteristics['half_smoothing']:.3f} Hz); "
        f"~{characteristics['approximate_usable_tapers']} low-bias "
        "taper(s)."
    )


def multitaper_psd(data: np.ndarray, settings: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    sfreq = settings["sfreq"]
    fmax = min(float(settings["psd_fmax"]), sfreq / 2.0)
    psds, freqs = mne.time_frequency.psd_array_multitaper(
        data.T,
        sfreq=sfreq,
        fmin=0.0,
        fmax=fmax,
        bandwidth=settings["mt_bandwidth"],
        adaptive=settings["mt_adaptive"],
        low_bias=settings["mt_low_bias"],
        normalization=settings["mt_normalization"],
        remove_dc=settings["mt_remove_dc"],
        output="power",
        n_jobs=1,
        verbose=False,
    )
    return freqs, psds.T  # frequencies x channels/regions


def cache_signature(bundle: SubjectBundle, settings: dict[str, Any]) -> str:
    file_meta = []
    for path in bundle.files:
        stat = Path(path).stat()
        file_meta.append({"path": str(Path(path).resolve()), "size": stat.st_size, "mtime": stat.st_mtime_ns})
    payload = {
        "files": file_meta,
        "bundle": asdict(bundle),
        "aperiodic_module_version": APERIODIC_MODULE_VERSION,
        "fit_range": settings["fit_range"],
        "sfreq": settings["sfreq"],
        "header_mode": settings["header_mode"],
        "window_length": settings["window_length"],
        "window_overlap": settings["window_overlap"],
        "limit_segments": settings["limit_segments"],
        "max_segments": settings["max_segments"],
        "segment_seed": settings["segment_seed"],
        "psd_fmax": settings["psd_fmax"],
        "mt_bandwidth": settings["mt_bandwidth"],
        "mt_adaptive": settings["mt_adaptive"],
        "mt_low_bias": settings["mt_low_bias"],
        "mt_normalization": settings["mt_normalization"],
        "mt_remove_dc": settings["mt_remove_dc"],
    }
    return hashlib.sha256(json.dumps(jsonable(payload), sort_keys=True).encode("utf-8")).hexdigest()


def save_psd_cache(record: PSDRecord, signature: str, metadata: dict[str, Any]) -> None:
    cache_path = Path(record.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        frequencies=record.frequencies,
        mean_psd=record.mean_psd,
        global_psd=record.global_psd,
        channel_names=np.asarray(record.channel_names, dtype=str),
        metadata=json.dumps(
            jsonable(
                {
                    **metadata,
                    "signature": signature,
                    "record": {
                        key: value
                        for key, value in asdict(record).items()
                        if key not in {"frequencies", "mean_psd", "global_psd", "channel_names"}
                    },
                }
            )
        ),
    )


def load_psd_cache(cache_path: str | Path, expected_signature: str) -> Optional[PSDRecord]:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("signature") != expected_signature:
                return None
            record_meta = dict(metadata["record"])
            record_meta["warnings"] = list(record_meta.get("warnings", []))
            return PSDRecord(
                **record_meta,
                frequencies=np.asarray(data["frequencies"], dtype=float),
                mean_psd=np.asarray(data["mean_psd"], dtype=float),
                global_psd=np.asarray(data["global_psd"], dtype=float),
                channel_names=[str(value) for value in data["channel_names"].tolist()],
            )
    except Exception:
        LOGGER.warning("Could not read cache %s; recalculating it", cache_path, exc_info=True)
        return None



class SpectrumValidationError(ValueError):
    """Expected per-spectrum validation failure that should be retained in outputs."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def _fit_frequency_mask(frequencies: np.ndarray, fit_range: tuple[float, float]) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=float)
    fit_min, fit_max = fit_range
    return np.isfinite(frequencies) & (frequencies >= fit_min) & (frequencies <= fit_max)


def _validate_spectrum_in_fit_range(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    fit_range: tuple[float, float],
    *,
    minimum_bins: int = 3,
) -> np.ndarray:
    """Validate only the frequencies that will actually be fitted.

    Returns the fitting mask when valid. Invalid individual channels are expected
    and are handled as retained failed fits rather than subject-level failures.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    if frequencies.ndim != 1 or spectrum.ndim != 1 or spectrum.size != frequencies.size:
        raise SpectrumValidationError(
            "frequency_spectrum_shape_mismatch", "Frequency and spectrum vectors do not align"
        )

    fit_mask = _fit_frequency_mask(frequencies, fit_range)
    if int(np.count_nonzero(fit_mask)) < minimum_bins:
        raise SpectrumValidationError(
            "insufficient_frequency_points",
            f"Fewer than {minimum_bins} PSD bins are available in the requested fit range",
        )

    fit_power = spectrum[fit_mask]
    if not np.all(np.isfinite(fit_power)):
        raise SpectrumValidationError(
            "nonfinite_psd_in_fit_range", "Spectrum contains nonfinite power values in the fit range"
        )
    if np.any(fit_power <= 0):
        raise SpectrumValidationError(
            "nonpositive_psd_in_fit_range", "Spectrum contains non-positive power values in the fit range"
        )
    if float(np.std(np.log10(fit_power))) < 1e-8:
        raise SpectrumValidationError(
            "flat_or_near_flat_spectrum", "Spectrum is flat or near-flat in the fit range"
        )
    return fit_mask

def resolve_bundle_sampling_frequency(
    bundle: SubjectBundle,
    manual_sfreq: float,
) -> tuple[float, str]:
    """Use trusted JSON sampling frequency, otherwise manual fallback."""
    if (
        bundle.metadata_trusted
        and bundle.metadata_sampling_frequency_hz is not None
    ):
        return (
            float(bundle.metadata_sampling_frequency_hz),
            "EEG-Pype JSON",
        )
    return float(manual_sfreq), "manual setting"


def prepare_psd_record(
    bundle: SubjectBundle,
    settings: dict[str, Any],
    output_root: str | Path,
) -> PSDRecord:
    manual_sfreq = float(settings["sfreq"])
    effective_sfreq, sfreq_source = (
        resolve_bundle_sampling_frequency(
            bundle,
            manual_sfreq,
        )
    )

    fit_min, fit_max = settings["fit_range"]
    if fit_max > effective_sfreq / 2.0:
        raise ValueError(
            f"Fit range for {bundle.subject} extends to "
            f"{fit_max:g} Hz, above Nyquist for the effective "
            f"sampling frequency {effective_sfreq:g} Hz."
        )

    subject_settings = dict(settings)
    subject_settings["manual_sfreq"] = manual_sfreq
    subject_settings["sfreq"] = effective_sfreq
    subject_settings["sfreq_source"] = sfreq_source
    subject_settings["psd_fmax"] = min(
        float(settings["psd_fmax"]),
        effective_sfreq / 2.0,
    )

    cache_dir = (
        Path(output_root)
        / "cache"
        / safe_slug(
            bundle.profile_key + "_" + bundle.representation
        )
    )
    cache_path = cache_dir / f"{safe_slug(bundle.subject)}.npz"
    signature = cache_signature(bundle, subject_settings)
    cached = load_psd_cache(cache_path, signature)
    if cached is not None:
        return cached

    segments, channel_names, segment_meta = load_segments(
        bundle,
        subject_settings,
    )
    if not segments:
        raise ValueError(
            f"No usable epochs/windows for {bundle.subject}"
        )

    psd_sum: Optional[np.ndarray] = None
    frequencies: Optional[np.ndarray] = None
    for segment in segments:
        current_freqs, current_psd = multitaper_psd(
            segment,
            subject_settings,
        )
        if frequencies is None:
            frequencies = current_freqs
            psd_sum = np.zeros_like(current_psd, dtype=float)
        elif not np.allclose(
            current_freqs,
            frequencies,
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Inconsistent PSD frequency vectors for "
                f"{bundle.subject}"
            )
        psd_sum += current_psd

    assert frequencies is not None and psd_sum is not None
    mean_psd = psd_sum / len(segments)

    valid_indices: list[int] = []
    invalid_channel_reasons: dict[str, str] = {}
    for channel_index, channel_name in enumerate(channel_names):
        try:
            _validate_spectrum_in_fit_range(
                frequencies,
                mean_psd[:, channel_index],
                tuple(subject_settings["fit_range"]),
            )
            valid_indices.append(channel_index)
        except SpectrumValidationError as exc:
            invalid_channel_reasons[channel_name] = exc.category

    if not valid_indices:
        raise ValueError(
            "No channel/region has a valid PSD in the requested fit "
            f"range for {bundle.subject}"
        )

    global_psd = np.mean(
        mean_psd[:, valid_indices],
        axis=1,
    )
    global_channel_names = [
        channel_names[index] for index in valid_indices
    ]

    segment_length_seconds = (
        segments[0].shape[0] / effective_sfreq
    )

    record = PSDRecord(
        subject=bundle.subject,
        profile_key=bundle.profile_key,
        profile_label=bundle.profile_label,
        level=bundle.level,
        representation=bundle.representation,
        source_method=bundle.source_method,
        atlas=bundle.atlas,
        region_selection=bundle.region_selection,
        spectral_variant=bundle.spectral_variant,
        passband_low=bundle.passband_low,
        passband_high=bundle.passband_high,
        frequencies=frequencies,
        mean_psd=mean_psd,
        global_psd=global_psd,
        channel_names=channel_names,
        n_segments=len(segments),
        n_available_segments=int(
            segment_meta["n_available_segments"]
        ),
        segment_length_seconds=segment_length_seconds,
        sfreq=effective_sfreq,
        cache_path=str(cache_path),
        input_files=list(segment_meta["selected_files"]),
        sampling_frequency_source=sfreq_source,
        metadata_path=bundle.metadata_path,
        metadata_status=bundle.metadata_status,
        metadata_schema_version=bundle.metadata_schema_version,
        metadata_eeg_pype_version=bundle.metadata_eeg_pype_version,
        metadata_epoch_length_seconds=(
            bundle.metadata_epoch_length_seconds
        ),
        metadata_export_scaling_factor=(
            bundle.metadata_export_scaling_factor
        ),
        metadata_output_unit=bundle.metadata_output_unit,
        metadata_reference_scheme=(
            bundle.metadata_reference_scheme
        ),
        metadata_reference_implementation=(
            bundle.metadata_reference_implementation
        ),
        metadata_base_highpass_hz=(
            bundle.metadata_base_highpass_hz
        ),
        metadata_base_lowpass_hz=(
            bundle.metadata_base_lowpass_hz
        ),
        warnings=list(segment_meta.get("warnings", [])),
        global_channel_names=global_channel_names,
        invalid_channel_reasons=invalid_channel_reasons,
    )
    save_psd_cache(record, signature, segment_meta)
    return record


# =============================================================================
# Specparam adapter and fit-quality calculations
# =============================================================================


def parse_frequency_intervals(value: str) -> list[tuple[float, float]]:
    if not value.strip():
        return []
    intervals: list[tuple[float, float]] = []
    for part in re.split(r"[;,]", value):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", part)
        if not match:
            raise ValueError(f"Invalid frequency interval: {part}")
        low, high = map(float, match.groups())
        if low >= high:
            raise ValueError(f"Invalid frequency interval: {part}")
        intervals.append((low, high))
    return intervals


def parse_peak_bands(value: str) -> dict[str, tuple[float, float]]:
    bands: dict[str, tuple[float, float]] = {}
    if not value.strip():
        return bands
    for part in value.split(";"):
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError(f"Invalid peak band definition: {part}")
        name, interval = part.split(":", 1)
        parsed = parse_frequency_intervals(interval.strip())
        if len(parsed) != 1:
            raise ValueError(f"Invalid peak band definition: {part}")
        bands[name.strip()] = parsed[0]
    return bands


def interpolate_frequency_intervals(
    freqs: np.ndarray, spectrum: np.ndarray, intervals: list[tuple[float, float]]
) -> tuple[np.ndarray, list[str]]:
    """Interpolate selected intervals in log10 power while retaining a regular frequency grid."""
    output = np.asarray(spectrum, dtype=float).copy()
    warnings: list[str] = []
    for low, high in intervals:
        inside = np.where((freqs >= low) & (freqs <= high))[0]
        if inside.size == 0:
            warnings.append(f"Line-noise interval {low:g}-{high:g} Hz contains no PSD bins.")
            continue
        left_candidates = np.where(freqs < low)[0]
        right_candidates = np.where(freqs > high)[0]
        if left_candidates.size == 0 or right_candidates.size == 0:
            warnings.append(f"Could not interpolate boundary interval {low:g}-{high:g} Hz.")
            continue
        left = left_candidates[-1]
        right = right_candidates[0]
        log_values = np.log10(np.maximum(output, np.finfo(float).tiny))
        log_values[inside] = np.interp(freqs[inside], [freqs[left], freqs[right]], [log_values[left], log_values[right]])
        output = np.power(10.0, log_values)
    return output, warnings


def _extract_specparam_components(model: Any) -> dict[str, np.ndarray]:
    model_component = model.results.model
    full = np.asarray(model_component.get_component("full"), dtype=float)
    aperiodic = np.asarray(model_component.get_component("aperiodic"), dtype=float)
    periodic = np.asarray(model_component.get_component("peak"), dtype=float)
    return {"full": full, "aperiodic": aperiodic, "periodic": periodic}


def _extract_specparam_metrics(model: Any) -> dict[str, float]:
    try:
        metrics = dict(model.results.metrics.results)
    except Exception:
        metrics = model.get_metrics("all")
        if not isinstance(metrics, dict):
            metrics = {}
    return {str(key): float(value) for key, value in metrics.items() if np.isscalar(value)}


def _squared_pearson_r_squared(observed: np.ndarray, modeled: np.ndarray) -> float:
    """Match specparam 2.0.0rc7's gof_rsquared definition exactly."""
    observed = np.asarray(observed, dtype=float)
    modeled = np.asarray(modeled, dtype=float)
    if observed.size < 2 or modeled.size != observed.size:
        return np.nan
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(modeled)):
        return np.nan
    if np.std(observed) == 0 or np.std(modeled) == 0:
        return np.nan
    correlation = float(np.corrcoef(observed, modeled)[0, 1])
    return correlation**2 if np.isfinite(correlation) else np.nan


def _adjusted_r_squared(r_squared: float, n_points: int, n_params: int) -> float:
    """Adjusted form of the same Pearson-correlation-squared metric.

    This is retained as a descriptive fit summary only; neighbouring multitaper
    frequency bins are correlated and should not be interpreted as independent
    regression observations.
    """
    if not np.isfinite(r_squared) or n_points <= n_params + 1:
        return np.nan
    return float(1.0 - (1.0 - r_squared) * (n_points - 1) / (n_points - n_params - 1))


def fit_spectrum_specparam(
    frequencies: np.ndarray,
    power_spectrum: np.ndarray,
    fit_settings: dict[str, Any],
    fit_identity: dict[str, Any],
    include_display: bool = False,
    native_output: Optional[tuple[str, str]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Optional[dict[str, Any]]]:
    """Fit one spectrum and return tabular EEG-Pype records plus optional plot data."""
    SpectralModel, specparam_version = ensure_specparam_rc7()
    fit_min, fit_max = fit_settings["fit_range"]
    row = {
        **fit_identity,
        "aperiodic_mode": fit_settings["aperiodic_mode"],
        "periodic_mode": fit_settings["periodic_mode"],
        "offset": np.nan,
        "knee": np.nan,
        "knee_frequency": np.nan,
        "exponent": np.nan,
        "n_peaks": 0,
        "r_squared": np.nan,
        "adjusted_r_squared": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "median_abs_error": np.nan,
        "maximum_abs_error": np.nan,
        "residual_bias": np.nan,
        "fit_status": "failed",
        "warning_flags": "",
        "n_frequency_bins": 0,
        "frequency_resolution": np.nan,
        "fit_min": fit_min,
        "fit_max": fit_max,
        "specparam_version": specparam_version,
        "EEG_Pype_version": EEG_PYPE_VERSION,
        "included": True,
        "exclusion_reason": "",
    }
    peak_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    display: Optional[dict[str, Any]] = None
    warnings: list[str] = list(fit_identity.get("pre_fit_warnings", []))

    try:
        frequencies = np.asarray(frequencies, dtype=float)
        spectrum = np.asarray(power_spectrum, dtype=float).copy()
        fit_mask = _validate_spectrum_in_fit_range(
            frequencies, spectrum, (fit_min, fit_max)
        )

        # Interpolate line-noise intervals on the complete available PSD before
        # restricting it to the fitting range. This preserves anchor bins just
        # outside the fit boundaries (for example, a 58-62 Hz interval when the
        # requested fit ends at 62 Hz). Intervals wholly outside the fit range
        # cannot affect the model and are skipped without producing misleading
        # "contains no PSD bins" warnings.
        active_intervals = [
            (low, high)
            for low, high in fit_settings["line_noise_intervals"]
            if high >= fit_min and low <= fit_max
        ]
        interpolated_spectrum, interpolation_warnings = interpolate_frequency_intervals(
            frequencies, spectrum, active_intervals
        )
        warnings.extend(interpolation_warnings)

        fit_input_freqs = frequencies[fit_mask]
        fit_input_spectrum = interpolated_spectrum[fit_mask]

        model = SpectralModel(
            aperiodic_mode=fit_settings["aperiodic_mode"],
            periodic_mode=fit_settings["periodic_mode"],
            peak_width_limits=tuple(fit_settings["peak_width_limits"]),
            max_n_peaks=fit_settings["max_n_peaks"],
            min_peak_height=fit_settings["min_peak_height"],
            peak_threshold=fit_settings["peak_threshold"],
            debug=False,
            verbose=False,
        )
        model.fit(fit_input_freqs, fit_input_spectrum, [fit_min, fit_max])
        if not model.results.has_model:
            raise RuntimeError("Specparam returned no fitted model")

        aperiodic_fit = np.asarray(model.results.get_params("aperiodic", version="fit"), dtype=float).ravel()
        try:
            aperiodic_converted = np.asarray(
                model.results.get_params("aperiodic", version="converted"), dtype=float
            ).ravel()
        except Exception:
            aperiodic_converted = aperiodic_fit.copy()

        try:
            periodic_converted = np.asarray(
                model.results.get_params("periodic", version="converted"), dtype=float
            )
            if periodic_converted.ndim == 1 and np.all(np.isnan(periodic_converted)):
                periodic_converted = np.empty((0, 3), dtype=float)
            elif periodic_converted.ndim == 1:
                periodic_converted = periodic_converted.reshape(1, -1)
        except Exception:
            periodic_converted = np.empty((0, 3), dtype=float)

        components = _extract_specparam_components(model)
        observed_log = np.asarray(model.data.power_spectrum, dtype=float)
        fit_freqs = np.asarray(model.data.freqs, dtype=float)
        modeled_log = components["full"]
        residual = observed_log - modeled_log
        n_bins = int(residual.size)
        n_params = int(getattr(model.results, "n_params", len(aperiodic_fit) + periodic_converted.size))
        metrics = _extract_specparam_metrics(model)
        fallback_r_squared = _squared_pearson_r_squared(observed_log, modeled_log)
        specparam_r_squared = float(metrics.get("gof_rsquared", fallback_r_squared))
        adjusted = float(
            metrics.get(
                "gof_adjrsquared",
                metrics.get(
                    "gof_adj_r_squared",
                    _adjusted_r_squared(specparam_r_squared, n_bins, n_params),
                ),
            )
        )

        row.update(
            {
                "offset": float(aperiodic_fit[0]),
                "exponent": float(aperiodic_fit[-1]),
                "n_peaks": int(periodic_converted.shape[0]),
                "r_squared": specparam_r_squared,
                "adjusted_r_squared": adjusted,
                "mae": float(metrics.get("error_mae", np.mean(np.abs(residual)))),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "median_abs_error": float(np.median(np.abs(residual))),
                "maximum_abs_error": float(np.max(np.abs(residual))),
                "residual_bias": float(np.mean(residual)),
                "fit_status": "success",
                "n_frequency_bins": n_bins,
                "frequency_resolution": float(np.median(np.diff(fit_freqs))) if n_bins > 1 else np.nan,
            }
        )
        if fit_settings["aperiodic_mode"] == "knee" and len(aperiodic_fit) >= 3:
            row["knee"] = float(aperiodic_fit[1])
            if len(aperiodic_converted) >= 3 and np.isfinite(aperiodic_converted[1]):
                row["knee_frequency"] = float(aperiodic_converted[1])
            elif row["knee"] > 0 and row["exponent"] != 0:
                row["knee_frequency"] = float(row["knee"] ** (1.0 / row["exponent"]))
            else:
                warnings.append("knee_frequency_not_interpretable")

        peak_width_min, peak_width_max = fit_settings["peak_width_limits"]
        boundary_distance = max(row["frequency_resolution"], 0.5)
        for peak_index, peak in enumerate(periodic_converted, start=1):
            center = float(peak[0])
            power = float(peak[1]) if peak.size > 1 else np.nan
            bandwidth = float(peak[2]) if peak.size > 2 else np.nan
            assigned_band = ""
            for band_name, (band_low, band_high) in fit_settings["peak_bands"].items():
                if band_low <= center < band_high or (center == band_high == fit_max):
                    assigned_band = band_name
                    break
            boundary_warning = center - fit_min <= boundary_distance or fit_max - center <= boundary_distance
            if boundary_warning:
                warnings.append(f"peak_{peak_index}_near_fit_boundary")
            if np.isfinite(bandwidth) and abs(bandwidth - peak_width_min) <= row["frequency_resolution"]:
                warnings.append(f"peak_{peak_index}_bandwidth_at_lower_limit")
            if np.isfinite(bandwidth) and abs(bandwidth - peak_width_max) <= row["frequency_resolution"]:
                warnings.append(f"peak_{peak_index}_bandwidth_at_upper_limit")

            peak_rows.append(
                {
                    **{key: fit_identity.get(key, "") for key in (
                        "run_id", "subject", "analysis_profile", "level", "source_method", "atlas",
                        "region_selection", "representation", "fit_type", "channel_or_region"
                    )},
                    "peak_number": peak_index,
                    "center_frequency": center,
                    "peak_power": power,
                    "bandwidth": bandwidth,
                    "additional_periodic_parameters": json.dumps(peak[3:].tolist()) if peak.size > 3 else "",
                    "assigned_band": assigned_band,
                    "boundary_warning": boundary_warning,
                    "parent_fit_status": "success",
                    "parent_included": True,
                }
            )

        if row["n_peaks"] >= fit_settings["max_n_peaks"] and fit_settings["max_n_peaks"] > 0:
            warnings.append("maximum_peak_count_reached")
        if np.isfinite(row["r_squared"]) and row["r_squared"] < fit_settings["warn_r2_below"]:
            warnings.append("r_squared_below_warning_threshold")
        if np.isfinite(row["mae"]) and row["mae"] > fit_settings["warn_mae_above"]:
            warnings.append("mae_above_warning_threshold")
        if np.isfinite(row["exponent"]) and not (
            fit_settings["warn_exponent_low"] <= row["exponent"] <= fit_settings["warn_exponent_high"]
        ):
            warnings.append("exponent_outside_warning_range")

        warnings = list(dict.fromkeys(str(warning) for warning in warnings if warning))
        row["warning_flags"] = "; ".join(warnings)
        for warning in warnings:
            qc_rows.append(
                {
                    "run_id": fit_identity.get("run_id", ""),
                    "subject": fit_identity.get("subject", ""),
                    "analysis_profile": fit_identity.get("analysis_profile", ""),
                    "fit_type": fit_identity.get("fit_type", ""),
                    "channel_or_region": fit_identity.get("channel_or_region", ""),
                    "warning_category": warning,
                    "relevant_value": "",
                    "user_action": "retained; visual warning only",
                    "exclusion_reason": "",
                }
            )

        if native_output is not None:
            native_dir, native_stem = native_output
            Path(native_dir).mkdir(parents=True, exist_ok=True)
            model.save(
                native_stem,
                file_path=native_dir,
                save_results=True,
                save_settings=True,
                save_data=True,
            )

        if include_display:
            display = {
                "frequencies": fit_freqs,
                "observed_log": observed_log,
                "modeled_log": modeled_log,
                "aperiodic_log": components["aperiodic"],
                "periodic_log": components["periodic"],
                "residual": residual,
                "peaks": periodic_converted,
                "row": row.copy(),
            }

    except Exception as exc:
        failure_category = exc.category if isinstance(exc, SpectrumValidationError) else "fit_failed"
        warning = f"{failure_category}: {type(exc).__name__}: {exc}"
        warnings.append(warning)
        row["warning_flags"] = "; ".join(dict.fromkeys(warnings))
        qc_rows.append(
            {
                "run_id": fit_identity.get("run_id", ""),
                "subject": fit_identity.get("subject", ""),
                "analysis_profile": fit_identity.get("analysis_profile", ""),
                "fit_type": fit_identity.get("fit_type", ""),
                "channel_or_region": fit_identity.get("channel_or_region", ""),
                "warning_category": failure_category,
                "relevant_value": str(exc),
                "user_action": "retained as failed fit",
                "exclusion_reason": "",
            }
        )
        if include_display:
            display_freqs = np.asarray(frequencies, dtype=float)
            display_power = np.asarray(power_spectrum, dtype=float)
            if display_freqs.ndim == 1 and display_power.ndim == 1 and display_freqs.size == display_power.size:
                display_mask = _fit_frequency_mask(display_freqs, (fit_min, fit_max))
                display_freqs = display_freqs[display_mask]
                display_power = display_power[display_mask]
            safe_observed = np.full(display_power.shape, np.nan, dtype=float)
            valid_power = np.isfinite(display_power) & (display_power > 0)
            safe_observed[valid_power] = np.log10(display_power[valid_power])
            display = {
                "frequencies": display_freqs,
                "observed_log": safe_observed,
                "modeled_log": np.full(display_freqs.shape, np.nan, dtype=float),
                "aperiodic_log": np.full(display_freqs.shape, np.nan, dtype=float),
                "periodic_log": np.full(display_freqs.shape, np.nan, dtype=float),
                "residual": np.full(display_freqs.shape, np.nan, dtype=float),
                "peaks": np.empty((0, 3)),
                "row": row.copy(),
            }
    return row, peak_rows, qc_rows, display


def save_fit_plot(display: dict[str, Any], path: str | Path, dpi: int, title: str) -> None:
    """Save an EEG-Pype-controlled publication-sized model plot."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(8, 6), constrained_layout=True)
    axis = figure.add_subplot(2, 1, 1)
    residual_axis = figure.add_subplot(2, 1, 2, sharex=axis)

    freqs = display["frequencies"]
    axis.plot(freqs, display["observed_log"], label="Observed PSD", linewidth=2.0)
    if np.any(np.isfinite(display["modeled_log"])):
        axis.plot(freqs, display["modeled_log"], label="Full model", linewidth=2.0)
        axis.plot(freqs, display["aperiodic_log"], label="Aperiodic component", linewidth=2.0, linestyle="--")
    for peak in np.asarray(display.get("peaks", [])):
        if len(peak) >= 3:
            center, _, bandwidth = peak[:3]
            axis.axvline(center, linewidth=1.0, alpha=0.6)
            axis.axvspan(center - bandwidth / 2.0, center + bandwidth / 2.0, alpha=0.08)

    axis.set_ylabel("log10 power", fontsize=18)
    axis.set_title(title, fontsize=17)
    axis.tick_params(labelsize=15)
    axis.legend(fontsize=13)

    residual_axis.axhline(0.0, linewidth=1.0)
    residual_axis.plot(freqs, display["residual"], linewidth=1.5)
    residual_axis.set_xlabel("Frequency (Hz)", fontsize=18)
    residual_axis.set_ylabel("Residual", fontsize=18)
    residual_axis.tick_params(labelsize=15)

    row = display["row"]
    annotation = (
        f"Exponent={row.get('exponent', np.nan):.3f}; "
        f"R²={row.get('r_squared', np.nan):.3f}; MAE={row.get('mae', np.nan):.3f}; "
        f"peaks={row.get('n_peaks', 0)}"
    )
    axis.text(0.01, 0.02, annotation, transform=axis.transAxes, fontsize=11, va="bottom")

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    figure.savefig(path, dpi=dpi)
    figure.clear()


# =============================================================================
# Subject fitting, multiprocessing and summaries
# =============================================================================


def load_record_from_cache(cache_path: str | Path) -> PSDRecord:
    with np.load(cache_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        record_meta = dict(metadata["record"])
        record_meta["warnings"] = list(record_meta.get("warnings", []))
        return PSDRecord(
            **record_meta,
            frequencies=np.asarray(data["frequencies"], dtype=float),
            mean_psd=np.asarray(data["mean_psd"], dtype=float),
            global_psd=np.asarray(data["global_psd"], dtype=float),
            channel_names=[str(value) for value in data["channel_names"].tolist()],
        )


def make_fit_identity(record: PSDRecord, run_id: str, fit_type: str, name: str) -> dict[str, Any]:
    pre_fit_warnings = list(record.warnings)
    if fit_type == "global" and record.invalid_channel_reasons:
        pre_fit_warnings.append(
            f"global_mean_excluded_{len(record.invalid_channel_reasons)}_invalid_channels_or_regions"
        )
    return {
        "run_id": run_id,
        "subject": record.subject,
        "analysis_profile": record.profile_key,
        "analysis_profile_label": record.profile_label,
        "level": record.level,
        "source_method": record.source_method,
        "atlas": record.atlas,
        "region_selection": record.region_selection,
        "representation": record.representation,
        "spectral_variant": record.spectral_variant,
        "passband_low": record.passband_low,
        "passband_high": record.passband_high,
        "sampling_frequency": record.sfreq,
        "sampling_frequency_source": record.sampling_frequency_source,
        "export_scaling_factor": record.metadata_export_scaling_factor,
        "output_unit": record.metadata_output_unit,
        "reference_scheme": record.metadata_reference_scheme,
        "reference_implementation": (
            record.metadata_reference_implementation
        ),
        "fit_type": fit_type,
        "channel_or_region": name,
        "pre_fit_warnings": pre_fit_warnings,
    }


def summarize_subject_fits(record: PSDRecord, fit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    global_rows = [row for row in fit_rows if row["fit_type"] == "global"]
    channel_rows = [row for row in fit_rows if row["fit_type"] != "global"]
    successful_channels = [row for row in channel_rows if row["fit_status"] == "success"]
    global_row = global_rows[0] if global_rows else {}

    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in successful_channels if np.isfinite(row.get(key, np.nan))], dtype=float)

    exponents = values("exponent")
    offsets = values("offset")
    r2s = values("r_squared")
    maes = values("mae")
    warned = [row for row in fit_rows if row.get("warning_flags")]
    failed = [row for row in fit_rows if row.get("fit_status") != "success"]

    return {
        "subject": record.subject,
        "analysis_profile": record.profile_key,
        "analysis_profile_label": record.profile_label,
        "level": record.level,
        "source_method": record.source_method,
        "atlas": record.atlas,
        "region_selection": record.region_selection,
        "representation": record.representation,
        "spectral_variant": record.spectral_variant,
        "n_input_epochs_or_windows": record.n_segments,
        "n_available_epochs_or_windows": record.n_available_segments,
        "sampling_frequency": record.sfreq,
        "epoch_or_window_length": record.segment_length_seconds,
        "n_channels_or_regions_total": len(record.channel_names),
        "n_channels_or_regions_in_global_mean": len(record.global_channel_names),
        "global_mean_excluded_channels_or_regions": "; ".join(record.invalid_channel_reasons),
        "global_offset": global_row.get("offset", np.nan),
        "global_exponent": global_row.get("exponent", np.nan),
        "global_knee": global_row.get("knee", np.nan),
        "global_knee_frequency": global_row.get("knee_frequency", np.nan),
        "global_r_squared": global_row.get("r_squared", np.nan),
        "global_mae": global_row.get("mae", np.nan),
        "global_n_peaks": global_row.get("n_peaks", 0),
        "mean_channel_or_region_exponent": float(np.mean(exponents)) if exponents.size else np.nan,
        "median_channel_or_region_exponent": float(np.median(exponents)) if exponents.size else np.nan,
        "sd_channel_or_region_exponent": float(np.std(exponents, ddof=1)) if exponents.size > 1 else np.nan,
        "iqr_channel_or_region_exponent": float(np.subtract(*np.percentile(exponents, [75, 25]))) if exponents.size else np.nan,
        "mean_channel_or_region_offset": float(np.mean(offsets)) if offsets.size else np.nan,
        "median_channel_or_region_offset": float(np.median(offsets)) if offsets.size else np.nan,
        "mean_channel_or_region_r_squared": float(np.mean(r2s)) if r2s.size else np.nan,
        "lowest_channel_or_region_r_squared": float(np.min(r2s)) if r2s.size else np.nan,
        "mean_channel_or_region_mae": float(np.mean(maes)) if maes.size else np.nan,
        "maximum_channel_or_region_mae": float(np.max(maes)) if maes.size else np.nan,
        "n_fits": len(fit_rows),
        "n_successful_fits": len(fit_rows) - len(failed),
        "n_warned_fits": len(warned),
        "n_failed_fits": len(failed),
        "proportion_successful": (len(fit_rows) - len(failed)) / len(fit_rows) if fit_rows else np.nan,
        "proportion_warned": len(warned) / len(fit_rows) if fit_rows else np.nan,
        "proportion_failed": len(failed) / len(fit_rows) if fit_rows else np.nan,
    }


def make_band_peak_rows(peak_rows: list[dict[str, Any]], peak_bands: dict[str, tuple[float, float]]) -> list[dict[str, Any]]:
    if not peak_bands:
        return []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    identity_keys = (
        "run_id", "subject", "analysis_profile", "level", "source_method", "atlas",
        "region_selection", "representation", "fit_type", "channel_or_region"
    )
    for peak in peak_rows:
        grouped[tuple(peak.get(key, "") for key in identity_keys)].append(peak)

    rows: list[dict[str, Any]] = []
    for identity, peaks in grouped.items():
        identity_dict = dict(zip(identity_keys, identity))
        for band_name, (low, high) in peak_bands.items():
            candidates = [peak for peak in peaks if low <= peak["center_frequency"] < high]
            strongest = max(candidates, key=lambda peak: peak["peak_power"]) if candidates else None
            rows.append(
                {
                    **identity_dict,
                    "band": band_name,
                    "band_low": low,
                    "band_high": high,
                    "n_peaks_in_band": len(candidates),
                    "strongest_peak_center_frequency": strongest["center_frequency"] if strongest else np.nan,
                    "strongest_peak_power": strongest["peak_power"] if strongest else np.nan,
                    "strongest_peak_bandwidth": strongest["bandwidth"] if strongest else np.nan,
                }
            )
    return rows


def fit_subject_task(task: dict[str, Any]) -> dict[str, Any]:
    """ProcessPool-safe worker that loads one cached PSD and fits all spectra."""
    record = load_record_from_cache(task["cache_path"])
    fit_settings = task["fit_settings"]
    run_id = task["run_id"]
    output_root = Path(task["output_root"])
    plot_scope = task["plot_scope"]
    dpi = task["dpi"]
    save_native_global = task["save_native_global"]

    fit_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    identities_and_spectra: list[tuple[str, str, np.ndarray]] = [("global", "Global spectrum", record.global_psd)]
    fit_type = "channel" if record.level == "sensor" else "region"
    identities_and_spectra.extend(
        (fit_type, name, record.mean_psd[:, index]) for index, name in enumerate(record.channel_names)
    )

    for current_fit_type, name, spectrum in identities_and_spectra:
        identity = make_fit_identity(record, run_id, current_fit_type, name)
        native = None
        if save_native_global and current_fit_type == "global":
            native_dir = output_root / "native_specparam" / safe_slug(record.profile_key)
            native = (str(native_dir), safe_slug(f"{record.subject}_global"))
        row, peaks, qc, display = fit_spectrum_specparam(
            record.frequencies,
            spectrum,
            fit_settings,
            identity,
            include_display=True,
            native_output=native,
        )
        fit_rows.append(row)
        peak_rows.extend(peaks)
        qc_rows.extend(qc)

        warned = bool(row.get("warning_flags"))
        save_plot = (
            plot_scope == "all fits"
            or (plot_scope == "global fits only" and current_fit_type == "global")
            or (plot_scope == "global + warned/failed" and (current_fit_type == "global" or warned or row["fit_status"] != "success"))
        )
        if save_plot and display is not None:
            plot_dir = output_root / "plots" / safe_slug(record.profile_key) / safe_slug(record.subject)
            plot_name = safe_slug(f"{record.subject}_{record.profile_key}_{name}_specparam_fit") + ".png"
            save_fit_plot(display, plot_dir / plot_name, dpi, f"{record.subject} – {name}")

    return {
        "subject": record.subject,
        "fits": fit_rows,
        "peaks": peak_rows,
        "band_peaks": make_band_peak_rows(peak_rows, fit_settings["peak_bands"]),
        "qc": qc_rows,
        "summary": summarize_subject_fits(record, fit_rows),
    }


# =============================================================================
# Export
# =============================================================================


def inventory_to_rows(items: list[InventoryItem], selected_profile_id: str, bundles: list[SubjectBundle]) -> list[dict[str, Any]]:
    selected_paths = {path for bundle in bundles for path in bundle.files}
    rows = []
    for item in items:
        row = asdict(item)
        row["selected_profile"] = profile_id(item) == selected_profile_id
        row["selected_for_analysis"] = item.path in selected_paths
        rows.append(row)
    return rows


def psd_metadata_rows(
    records: list[PSDRecord],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        bandwidth = settings["mt_bandwidth"]
        characteristics = multitaper_characteristics(
            record.segment_length_seconds,
            bandwidth,
            low_bias=bool(settings["mt_low_bias"]),
        )
        rows.append(
            {
                "subject": record.subject,
                "analysis_profile": record.profile_key,
                "input_representation": record.representation,
                "input_files": " | ".join(record.input_files),
                "sampling_frequency": record.sfreq,
                "sampling_frequency_source": (
                    record.sampling_frequency_source
                ),
                "epoch_or_window_length": (
                    record.segment_length_seconds
                ),
                "json_epoch_length_seconds": (
                    record.metadata_epoch_length_seconds
                ),
                "overlap_percent": (
                    settings["window_overlap"]
                    if record.representation == "continuous"
                    else 0
                ),
                "n_epochs_or_windows": record.n_segments,
                "n_available_epochs_or_windows": (
                    record.n_available_segments
                ),
                "multitaper_bandwidth": (
                    "Auto" if bandwidth is None else bandwidth
                ),
                "effective_multitaper_bandwidth": (
                    characteristics["effective_bandwidth"]
                ),
                "multitaper_half_smoothing": (
                    characteristics["half_smoothing"]
                ),
                "approximate_taper_count": (
                    characteristics["approximate_usable_tapers"]
                ),
                "adaptive": settings["mt_adaptive"],
                "low_bias": settings["mt_low_bias"],
                "normalization": settings["mt_normalization"],
                "remove_dc": settings["mt_remove_dc"],
                "frequency_bin_spacing": float(
                    np.median(np.diff(record.frequencies))
                ),
                "available_frequency_min": float(
                    record.frequencies.min()
                ),
                "available_frequency_max": float(
                    record.frequencies.max()
                ),
                "fit_range_min": settings["fit_range"][0],
                "fit_range_max": settings["fit_range"][1],
                "global_aggregation": (
                    "arithmetic mean across valid channels/regions"
                ),
                "n_channels_or_regions_total": len(
                    record.channel_names
                ),
                "n_channels_or_regions_in_global_mean": len(
                    record.global_channel_names
                ),
                "global_mean_included_channels_or_regions": "; ".join(
                    record.global_channel_names
                ),
                "invalid_channel_or_region_reasons": json.dumps(
                    record.invalid_channel_reasons,
                    sort_keys=True,
                ),
                "spectral_variant": record.spectral_variant,
                "effective_passband_low": record.passband_low,
                "effective_passband_high": record.passband_high,
                "export_scaling_factor": (
                    record.metadata_export_scaling_factor
                ),
                "output_unit": record.metadata_output_unit,
                "reference_scheme": (
                    record.metadata_reference_scheme
                ),
                "reference_implementation": (
                    record.metadata_reference_implementation
                ),
                "json_base_highpass_hz": (
                    record.metadata_base_highpass_hz
                ),
                "json_base_lowpass_hz": (
                    record.metadata_base_lowpass_hz
                ),
                "metadata_path": record.metadata_path,
                "metadata_status": record.metadata_status,
                "metadata_schema_version": (
                    record.metadata_schema_version
                ),
                "metadata_eeg_pype_version": (
                    record.metadata_eeg_pype_version
                ),
                "warnings": "; ".join(record.warnings),
                "cache_path": record.cache_path,
            }
        )
    return rows


def write_dataframe_csv(rows: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def export_results(
    output_root: str | Path,
    run_id: str,
    settings: dict[str, Any],
    items: list[InventoryItem],
    selected_profile_id: str,
    bundles: list[SubjectBundle],
    records: list[PSDRecord],
    worker_results: list[dict[str, Any]],
    save_csv: bool,
    save_excel: bool,
) -> dict[str, str]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    fits = [row for result in worker_results for row in result.get("fits", [])]
    peaks = [row for result in worker_results for row in result.get("peaks", [])]
    band_peaks = [row for result in worker_results for row in result.get("band_peaks", [])]
    qc = [row for result in worker_results for row in result.get("qc", [])]
    summaries = [result["summary"] for result in worker_results if result.get("summary")]
    inventory = inventory_to_rows(items, selected_profile_id, bundles)
    psd_metadata = psd_metadata_rows(records, settings)
    setting_rows = [{"setting": key, "value": json.dumps(jsonable(value)) if isinstance(value, (dict, list, tuple)) else value} for key, value in settings.items()]

    tables = {
        "subject_summary": pd.DataFrame(summaries),
        "fits_long": pd.DataFrame(fits),
        "peaks_long": pd.DataFrame(peaks),
        "band_peaks": pd.DataFrame(band_peaks),
        "fit_qc": pd.DataFrame(qc),
        "data_inventory": pd.DataFrame(inventory),
        "psd_metadata": pd.DataFrame(psd_metadata),
        "run_settings": pd.DataFrame(setting_rows),
    }

    outputs: dict[str, str] = {}
    if save_csv:
        for name, frame in tables.items():
            path = output_root / f"{name}.csv"
            frame.to_csv(path, index=False)
            outputs[name] = str(path)

    if save_excel:
        excel_path = output_root / f"EEG_aperiodic_analysis_{run_id}.xlsx"
        readme = pd.DataFrame(
            {
                "Item": [
                    "Module", "Module version", "EEG-Pype version", "Specparam version",
                    "Analysis profile", "Run ID", "Important distinction"
                ],
                "Value": [
                    "EEG-Pype aperiodic / Specparam analysis",
                    APERIODIC_MODULE_VERSION,
                    EEG_PYPE_VERSION,
                    REQUIRED_SPECPARAM_VERSION,
                    selected_profile_id,
                    run_id,
                    "Global-spectrum parameters and averages of separately fitted channel/region parameters are not interchangeable.",
                ],
            }
        )
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            readme.to_excel(writer, sheet_name="README", index=False)
            sheet_names = {
                "subject_summary": "Subject Summary",
                "fits_long": "Fits",
                "peaks_long": "Peaks",
                "band_peaks": "Band Peaks",
                "fit_qc": "Fit QC",
                "data_inventory": "Data Inventory",
                "psd_metadata": "PSD Metadata",
                "run_settings": "Run Settings",
            }
            for name, frame in tables.items():
                frame.to_excel(writer, sheet_name=sheet_names[name], index=False)
        outputs["excel"] = str(excel_path)

    versions = collect_package_versions()
    versions_path = output_root / "package_versions.txt"
    versions_path.write_text("\n".join(f"{key}: {value}" for key, value in versions.items()) + "\n", encoding="utf-8")
    outputs["package_versions"] = str(versions_path)

    specparam_settings = {
        key: settings[key]
        for key in (
            "fit_range", "aperiodic_mode", "periodic_mode", "peak_width_limits",
            "max_n_peaks", "min_peak_height", "peak_threshold", "line_noise_intervals", "peak_bands"
        )
    }
    settings_path = output_root / "specparam_settings.json"
    atomic_json_dump(specparam_settings, settings_path)
    outputs["specparam_settings"] = str(settings_path)

    manifest = {
        "run_id": run_id,
        "created": datetime.now().isoformat(),
        "selected_profile": selected_profile_id,
        "settings": settings,
        "package_versions": versions,
        "n_subject_bundles": len(bundles),
        "n_psd_records": len(records),
        "n_fit_rows": len(fits),
        "n_peak_rows": len(peaks),
        "n_qc_rows": len(qc),
        "subjects": [record.subject for record in records],
        "outputs": outputs,
    }
    manifest_path = output_root / "analysis_manifest.json"
    atomic_json_dump(manifest, manifest_path)
    outputs["manifest"] = str(manifest_path)
    return outputs


# =============================================================================
# Preview review GUI
# =============================================================================


def create_preview_displays(records: list[PSDRecord], settings: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    displays: list[dict[str, Any]] = []
    for record in records:
        spectra: list[tuple[str, str, np.ndarray]] = [("global", "Global spectrum", record.global_psd)]
        fit_type = "channel" if record.level == "sensor" else "region"
        spectra.extend((fit_type, name, record.mean_psd[:, index]) for index, name in enumerate(record.channel_names))
        for current_fit_type, name, spectrum in spectra:
            identity = make_fit_identity(record, run_id, current_fit_type, name)
            row, peaks, qc, display = fit_spectrum_specparam(
                record.frequencies,
                spectrum,
                settings,
                identity,
                include_display=True,
            )
            if display is not None:
                display["subject"] = record.subject
                display["label"] = f"{current_fit_type.title()}: {name}"
                display["fit_type"] = current_fit_type
                displays.append(display)
    return displays


def review_preview(displays: list[dict[str, Any]]) -> None:
    if sg is None:
        raise RuntimeError("FreeSimpleGUI is required for the preview window")
    if not displays:
        sg.popup_error("No preview fits were generated.")
        return

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    subjects = sorted({display["subject"] for display in displays}, key=str.lower)
    current_subject = subjects[0]

    def labels_for_subject(subject: str) -> list[str]:
        return [display["label"] for display in displays if display["subject"] == subject]

    layout = [
        [
            sg.Text("Preview subject:"),
            sg.Combo(subjects, default_value=current_subject, key="-PREVIEW_SUBJECT-", readonly=True, enable_events=True, size=(30, 1)),
            sg.Text("Fit:"),
            sg.Combo(labels_for_subject(current_subject), default_value=labels_for_subject(current_subject)[0], key="-PREVIEW_FIT-", readonly=True, enable_events=True, size=(35, 1)),
        ],
        [
            sg.Button("Previous"), sg.Button("Next"), sg.Button("Worst R²"), sg.Button("Highest MAE"),
            sg.Button("Close review")
        ],
        [sg.Canvas(key="-FIGURE_CANVAS-", size=(800, 550)), sg.Multiline("", key="-FIT_INFO-", size=(48, 30), disabled=True)],
    ]
    window = sg.Window("EEG-Pype Aperiodic Preview Review", layout, finalize=True, resizable=True)

    figure = Figure(figsize=(8, 6), constrained_layout=True)
    axis = figure.add_subplot(2, 1, 1)
    residual_axis = figure.add_subplot(2, 1, 2, sharex=axis)
    canvas_agg = FigureCanvasTkAgg(figure, window["-FIGURE_CANVAS-"].TKCanvas)
    canvas_agg.draw()
    canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)

    def selected_index() -> int:
        subject = window["-PREVIEW_SUBJECT-"].get()
        label = window["-PREVIEW_FIT-"].get()
        for index, display in enumerate(displays):
            if display["subject"] == subject and display["label"] == label:
                return index
        return 0

    def show(index: int) -> None:
        display = displays[index]
        axis.clear()
        residual_axis.clear()
        freqs = display["frequencies"]
        axis.plot(freqs, display["observed_log"], label="Observed PSD", linewidth=2.0)
        if np.any(np.isfinite(display["modeled_log"])):
            axis.plot(freqs, display["modeled_log"], label="Full model", linewidth=2.0)
            axis.plot(freqs, display["aperiodic_log"], label="Aperiodic", linewidth=2.0, linestyle="--")
        for peak in np.asarray(display.get("peaks", [])):
            if len(peak) >= 3:
                center, _, bandwidth = peak[:3]
                axis.axvline(center, linewidth=1.0, alpha=0.6)
                axis.axvspan(center - bandwidth / 2.0, center + bandwidth / 2.0, alpha=0.08)
        axis.set_ylabel("log10 power")
        axis.set_title(f"{display['subject']} – {display['label']}")
        axis.legend()
        residual_axis.axhline(0.0, linewidth=1.0)
        residual_axis.plot(freqs, display["residual"])
        residual_axis.set_xlabel("Frequency (Hz)")
        residual_axis.set_ylabel("Residual")
        canvas_agg.draw()

        row = display["row"]
        info_lines = [
            f"Subject: {row.get('subject', '')}",
            f"Profile: {row.get('analysis_profile_label', row.get('analysis_profile', ''))}",
            f"Fit: {row.get('fit_type', '')} – {row.get('channel_or_region', '')}",
            f"Status: {row.get('fit_status', '')}",
            f"Aperiodic mode: {row.get('aperiodic_mode', '')}",
            f"Offset: {row.get('offset', np.nan):.5f}",
            f"Knee: {row.get('knee', np.nan):.5f}",
            f"Knee frequency: {row.get('knee_frequency', np.nan):.5f}",
            f"Exponent: {row.get('exponent', np.nan):.5f}",
            f"R²: {row.get('r_squared', np.nan):.5f}",
            f"Adjusted R²: {row.get('adjusted_r_squared', np.nan):.5f}",
            f"MAE: {row.get('mae', np.nan):.5f}",
            f"RMSE: {row.get('rmse', np.nan):.5f}",
            f"Number of peaks: {row.get('n_peaks', 0)}",
            "",
            "Warnings:",
            row.get("warning_flags", "") or "None",
        ]
        window["-FIT_INFO-"].update("\n".join(info_lines))
        window["-PREVIEW_SUBJECT-"].update(value=display["subject"])
        window["-PREVIEW_FIT-"].update(values=labels_for_subject(display["subject"]), value=display["label"])

    show(0)
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Close review"):
            break
        if event == "-PREVIEW_SUBJECT-":
            labels = labels_for_subject(values["-PREVIEW_SUBJECT-"])
            window["-PREVIEW_FIT-"].update(values=labels, value=labels[0])
            show(selected_index())
        elif event == "-PREVIEW_FIT-":
            show(selected_index())
        elif event in ("Previous", "Next"):
            current = selected_index()
            delta = -1 if event == "Previous" else 1
            show((current + delta) % len(displays))
        elif event == "Worst R²":
            valid = [
                (index, display["row"].get("r_squared", np.nan))
                for index, display in enumerate(displays)
                if np.isfinite(display["row"].get("r_squared", np.nan))
            ]
            if valid:
                show(min(valid, key=lambda entry: entry[1])[0])
        elif event == "Highest MAE":
            valid = [
                (index, display["row"].get("mae", np.nan))
                for index, display in enumerate(displays)
                if np.isfinite(display["row"].get("mae", np.nan))
            ]
            if valid:
                show(max(valid, key=lambda entry: entry[1])[0])
    window.close()


# =============================================================================
# Main GUI
# =============================================================================


def _parse_optional_float(value: str, label: str) -> Optional[float]:
    if value.strip().lower() in {"", "auto", "none"}:
        return None
    number = float(value)
    if number <= 0:
        raise ValueError(f"{label} must be positive or Auto")
    return number


def collect_gui_settings(values: dict[str, Any]) -> dict[str, Any]:
    sfreq = float(values["-SFREQ-"])
    window_length = float(values["-WINDOW_LENGTH-"])
    overlap = float(values["-WINDOW_OVERLAP-"])
    if sfreq <= 0 or window_length <= 0:
        raise ValueError("Sampling frequency and epoch/window length must be positive")
    if not 0 <= overlap < 100:
        raise ValueError("Window overlap must be between 0 and <100%")

    fit_min = float(values["-FIT_MIN-"])
    fit_max = float(values["-FIT_MAX-"])
    if not 0 < fit_min < fit_max <= sfreq / 2.0:
        raise ValueError(f"Fit range must be >0 and no higher than Nyquist ({sfreq / 2:g} Hz)")

    peak_width_min = float(values["-PEAK_WIDTH_MIN-"])
    peak_width_max = float(values["-PEAK_WIDTH_MAX-"])
    if not 0 < peak_width_min < peak_width_max:
        raise ValueError("Peak-width limits are invalid")

    max_n_peaks = int(values["-MAX_PEAKS-"])
    if max_n_peaks < 0:
        raise ValueError("Maximum number of peaks cannot be negative")

    max_segments = int(values["-MAX_SEGMENTS-"])
    segment_seed = int(values["-SEGMENT_SEED-"])
    if max_segments < 1:
        raise ValueError("Maximum epochs/windows must be at least 1")

    preview_n = int(values["-PREVIEW_N-"])
    preview_seed = int(values["-PREVIEW_SEED-"])
    if preview_n < 1:
        raise ValueError("Preview subject count must be at least 1")

    n_workers = int(values["-WORKERS-"])
    if not 1 <= n_workers <= max(1, os.cpu_count() or 1):
        raise ValueError(f"Workers must be between 1 and {os.cpu_count() or 1}")

    dpi = int(values["-DPI-"])
    if not 72 <= dpi <= 1200:
        raise ValueError("DPI must be between 72 and 1200")

    mt_bandwidth = _parse_optional_float(values["-MT_BANDWIDTH-"], "Multitaper bandwidth")
    psd_fmax = min(sfreq / 2.0, max(60.0, fit_max))

    line_intervals = parse_frequency_intervals(values["-LINE_NOISE-"])
    peak_bands = parse_peak_bands(values["-PEAK_BANDS-"])

    settings = {
        "sfreq": sfreq,
        "header_mode": values["-HEADER_MODE-"],
        "window_length": window_length,
        "window_overlap": overlap,
        "limit_segments": bool(values["-LIMIT_SEGMENTS-"]),
        "max_segments": max_segments,
        "segment_seed": segment_seed,
        "psd_fmax": psd_fmax,
        "mt_bandwidth": mt_bandwidth,
        "mt_adaptive": bool(values["-MT_ADAPTIVE-"]),
        "mt_low_bias": bool(values["-MT_LOW_BIAS-"]),
        "mt_normalization": values["-MT_NORMALIZATION-"],
        "mt_remove_dc": bool(values["-MT_REMOVE_DC-"]),
        "fit_range": (fit_min, fit_max),
        "aperiodic_mode": values["-APERIODIC_MODE-"].lower(),
        "periodic_mode": values["-PERIODIC_MODE-"].lower(),
        "peak_width_limits": (peak_width_min, peak_width_max),
        "max_n_peaks": max_n_peaks,
        "min_peak_height": float(values["-MIN_PEAK_HEIGHT-"]),
        "peak_threshold": float(values["-PEAK_THRESHOLD-"]),
        "line_noise_intervals": line_intervals,
        "peak_bands": peak_bands,
        "warn_r2_below": float(values["-WARN_R2-"]),
        "warn_mae_above": float(values["-WARN_MAE-"]),
        "warn_exponent_low": float(values["-WARN_EXP_LOW-"]),
        "warn_exponent_high": float(values["-WARN_EXP_HIGH-"]),
        "preview_method": values["-PREVIEW_METHOD-"].lower(),
        "preview_n": preview_n,
        "preview_seed": preview_seed,
        "n_workers": n_workers,
        "plot_scope": values["-PLOT_SCOPE-"].lower(),
        "dpi": dpi,
        "save_csv": bool(values["-SAVE_CSV-"]),
        "save_excel": bool(values["-SAVE_EXCEL-"]),
        "save_native_global": bool(values["-SAVE_NATIVE-"]),
    }
    if settings["warn_exponent_low"] >= settings["warn_exponent_high"]:
        raise ValueError("Exponent warning range is invalid")

    mt_characteristics = multitaper_characteristics(
        window_length,
        mt_bandwidth,
        low_bias=bool(settings["mt_low_bias"]),
    )
    if (
        mt_bandwidth is not None
        and mt_bandwidth
        < float(mt_characteristics["minimum_bandwidth"])
    ):
        raise ValueError(
            "Multitaper bandwidth is too small for the selected "
            "epoch/window length. Use at least "
            f"{mt_characteristics['minimum_bandwidth']:.3f} Hz."
        )

    if peak_width_min <= 2 * float(
        mt_characteristics["bin_spacing"]
    ):
        settings.setdefault("global_warnings", []).append(
            "Minimum peak width is <= twice the PSD bin spacing."
        )

    if peak_width_min < float(
        mt_characteristics["effective_bandwidth"]
    ):
        settings.setdefault("global_warnings", []).append(
            "Minimum peak width is narrower than the multitaper full "
            "smoothing bandwidth "
            f"({mt_characteristics['effective_bandwidth']:.3f} Hz; "
            f"±{mt_characteristics['half_smoothing']:.3f} Hz). "
            "Such narrow peaks may not be independently resolvable."
        )

    if settings["aperiodic_mode"] == "knee" and fit_max - fit_min < 20:
        settings.setdefault("global_warnings", []).append(
            "Knee mode was selected for a relatively restricted fitting range."
        )
    return settings


def log_global_settings_warnings(
    window: Any,
    settings: dict[str, Any],
) -> None:
    """Show non-fatal PSD/Specparam settings warnings in the GUI and log."""
    for warning in settings.get("global_warnings", []):
        gui_log(
            window,
            f"Settings warning: {warning}",
            "warning",
        )


def preview_subjects(bundles: list[SubjectBundle], settings: dict[str, Any]) -> list[SubjectBundle]:
    method = settings["preview_method"]
    if method == "all subjects":
        return bundles
    n = min(settings["preview_n"], len(bundles))
    if method == "random n subjects":
        rng = random.Random(settings["preview_seed"])
        return sorted(rng.sample(bundles, n), key=lambda bundle: bundle.subject.lower())
    return bundles[:n]


def create_main_window() -> Any:
    if sg is None:
        raise RuntimeError(
            "FreeSimpleGUI is not installed. Install the GUI dependency used by EEG-Pype before running this script."
        )

    sg.theme("Default1")
    sg.set_options(font=("Helvetica", 10))
    header_bg = "#2C5784"
    main_bg = "#F0F2F6"
    button_color = ("#FFFFFF", "#2C5784")
    
    output_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output_folder = f"{DEFAULT_OUTPUT_FOLDER}_{output_timestamp}"

    header = [
        [sg.Text("EEG-Pype Aperiodic / Specparam Analysis", font=("Helvetica", 20, "bold"), text_color="white", background_color=header_bg)],
        [sg.Text(f"Standalone module v{APERIODIC_MODULE_VERSION} | Specparam {REQUIRED_SPECPARAM_VERSION}", text_color="white", background_color=header_bg)],
    ]

    setup_column = [
        [sg.Text("Step 1: Input and data level", font=("Helvetica", 12, "bold"), background_color=main_bg)],
        [sg.Text("Preprocessing batch folder:", background_color=main_bg)],
        [sg.Input(key="-FOLDER-", size=(34, 1)), sg.FolderBrowse(button_color=button_color), sg.Button("Scan data", button_color=button_color)],
        [sg.Text("Analysis profile:", background_color=main_bg)],
        [sg.Combo([], key="-PROFILE-", readonly=True, size=(42, 1), enable_events=True)],
        [sg.Text("Sampling frequency fallback (Hz):", background_color=main_bg), sg.Input("256", key="-SFREQ-", size=(8, 1), enable_events=True)],
        [sg.Text("Header handling:", background_color=main_bg), sg.Combo(["Auto", "Yes", "No"], default_value="Auto", key="-HEADER_MODE-", readonly=True)],
        [sg.HorizontalSeparator()],
        [sg.Text("Continuous-window / epoch settings", font=("Helvetica", 11, "bold"), background_color=main_bg)],
        [sg.Text("Window/epoch length (s):", background_color=main_bg), sg.Input("8", key="-WINDOW_LENGTH-", size=(8, 1), enable_events=True)],
        [sg.Text("Continuous overlap (%):", background_color=main_bg), sg.Input("0", key="-WINDOW_OVERLAP-", size=(8, 1))],
        [sg.Checkbox("Limit epochs/windows", key="-LIMIT_SEGMENTS-", default=False, background_color=main_bg)],
        [sg.Text("Maximum:", background_color=main_bg), sg.Input("20", key="-MAX_SEGMENTS-", size=(7, 1)), sg.Text("Seed:", background_color=main_bg), sg.Input("111", key="-SEGMENT_SEED-", size=(7, 1))],
        [sg.Text("Data inventory", font=("Helvetica", 11, "bold"), background_color=main_bg)],
        [sg.Multiline("Select a batch folder and click Scan data.", key="-INVENTORY-", size=(48, 16), disabled=True)],
    ]

    analysis_column = [
        [sg.Text("Step 2: PSD and Specparam settings", font=("Helvetica", 12, "bold"), background_color=main_bg)],
        [sg.Frame("Multitaper PSD", [
            [sg.Text("Bandwidth (Hz):", background_color=main_bg), sg.Input("Auto", key="-MT_BANDWIDTH-", size=(9, 1), enable_events=True)],
            [sg.Checkbox("Adaptive weighting", key="-MT_ADAPTIVE-", default=False, background_color=main_bg)],
            [sg.Checkbox("Low-bias tapers", key="-MT_LOW_BIAS-", default=True, background_color=main_bg)],
            [sg.Checkbox("Remove DC", key="-MT_REMOVE_DC-", default=True, background_color=main_bg)],
            [sg.Text("Normalization:", background_color=main_bg), sg.Combo(["full", "length"], default_value="full", key="-MT_NORMALIZATION-", readonly=True)],
            [sg.Text("", key="-RESOLUTION-", size=(48, 4), font=("Helvetica", 9, "italic"), background_color=main_bg)],
        ], background_color=main_bg, expand_x=True)],
        [sg.Frame("Specparam fit", [
            [sg.Text("Fit range (Hz):", background_color=main_bg), sg.Input("1", key="-FIT_MIN-", size=(6, 1)), sg.Text("to", background_color=main_bg), sg.Input("30", key="-FIT_MAX-", size=(6, 1))],
            [sg.Text("Aperiodic mode:", background_color=main_bg), sg.Combo(["Fixed", "Knee"], default_value="Fixed", key="-APERIODIC_MODE-", readonly=True)],
            [sg.Text("Periodic mode:", background_color=main_bg), sg.Combo(["Gaussian", "Skewed_Gaussian", "Cauchy"], default_value="Gaussian", key="-PERIODIC_MODE-", readonly=True)],
            [sg.Text("Peak width limits:", background_color=main_bg), sg.Input("1", key="-PEAK_WIDTH_MIN-", size=(6, 1)), sg.Text("to", background_color=main_bg), sg.Input("8", key="-PEAK_WIDTH_MAX-", size=(6, 1))],
            [sg.Text("Maximum peaks:", background_color=main_bg), sg.Input("6", key="-MAX_PEAKS-", size=(6, 1))],
            [sg.Text("Minimum peak height:", background_color=main_bg), sg.Input("0.10", key="-MIN_PEAK_HEIGHT-", size=(7, 1))],
            [sg.Text("Peak threshold (SD):", background_color=main_bg), sg.Input("2.0", key="-PEAK_THRESHOLD-", size=(7, 1))],
            [sg.Text("Line-noise interpolation:", background_color=main_bg), sg.Input("", key="-LINE_NOISE-", size=(24, 1)), sg.Text("e.g. 49-51", background_color=main_bg)],
            [sg.Text("Post-hoc peak bands:", background_color=main_bg)],
            [sg.Input(DEFAULT_BANDS, key="-PEAK_BANDS-", size=(48, 1))],
        ], background_color=main_bg, expand_x=True)],
        [sg.Frame("Visual QC warnings only", [
            [sg.Text("Warn if R² <", background_color=main_bg), sg.Input("0.90", key="-WARN_R2-", size=(6, 1)), sg.Text("or MAE >", background_color=main_bg), sg.Input("0.10", key="-WARN_MAE-", size=(6, 1))],
            [sg.Text("Warn exponent outside", background_color=main_bg), sg.Input("0", key="-WARN_EXP_LOW-", size=(6, 1)), sg.Text("to", background_color=main_bg), sg.Input("4", key="-WARN_EXP_HIGH-", size=(6, 1))],
        ], background_color=main_bg, expand_x=True)],
    ]

    output_column = [
        [sg.Text("Step 3: Preview, output and execution", font=("Helvetica", 12, "bold"), background_color=main_bg)],
        [sg.Frame("Preview", [
            [sg.Text("Selection:", background_color=main_bg), sg.Combo(["First N subjects", "Random N subjects", "All subjects"], default_value="First N subjects", key="-PREVIEW_METHOD-", readonly=True)],
            [sg.Text("N:", background_color=main_bg), sg.Input("3", key="-PREVIEW_N-", size=(6, 1)), sg.Text("Seed:", background_color=main_bg), sg.Input("111", key="-PREVIEW_SEED-", size=(7, 1))],
            [sg.Button("Prepare & review preview", button_color=button_color)],
        ], background_color=main_bg, expand_x=True)],
        [sg.Frame("Output", [
            [sg.Text("Output folder name:", background_color=main_bg), sg.Input(default_output_folder, key="-OUTPUT_NAME-", size=(32, 1))],
            [sg.Checkbox("Save detailed CSV files", key="-SAVE_CSV-", default=True, background_color=main_bg)],
            [sg.Checkbox("Save Excel workbook", key="-SAVE_EXCEL-", default=True, background_color=main_bg)],
            [sg.Checkbox("Save native Specparam JSON for global fits", key="-SAVE_NATIVE-", default=False, background_color=main_bg)],
            [sg.Text("PNG scope:", background_color=main_bg), sg.Combo(["No PNG output", "Global fits only", "Global + warned/failed", "All fits"], default_value="Global fits only", key="-PLOT_SCOPE-", readonly=True)],
            [sg.Text("PNG DPI:", background_color=main_bg), sg.Input("300", key="-DPI-", size=(7, 1))],
            [sg.Text("Processing workers:", background_color=main_bg), sg.Input(str(max(1, int((os.cpu_count() or 2) * 0.7))), key="-WORKERS-", size=(7, 1))],
        ], background_color=main_bg, expand_x=True)],
        [sg.Button("Run full batch & export", button_color=button_color), sg.Button("Exit")],
        [sg.ProgressBar(100, orientation="h", size=(42, 15), key="-PROGRESS-")],
        [sg.Text("Run log", font=("Helvetica", 11, "bold"), background_color=main_bg)],
        [sg.Multiline("", key="-LOG-", size=(52, 20), autoscroll=True, disabled=True)],
    ]

    layout = [
        [sg.Column(header, background_color=header_bg, expand_x=True)],
        [
            sg.Column(setup_column, background_color=main_bg, vertical_alignment="top", expand_y=True),
            sg.VerticalSeparator(),
            sg.Column(analysis_column, background_color=main_bg, vertical_alignment="top", expand_y=True),
            sg.VerticalSeparator(),
            sg.Column(output_column, background_color=main_bg, vertical_alignment="top", expand_y=True, expand_x=True),
        ],
    ]
    return sg.Window("EEG-Pype Aperiodic Analysis", layout, background_color=main_bg, finalize=True, resizable=True, margins=(0, 0))


def run_full_batch(
    window: Any,
    values: dict[str, Any],
    items: list[InventoryItem],
    profile_lookup: dict[str, str],
    settings: dict[str, Any],
) -> dict[str, str]:
    selected_label = values["-PROFILE-"]
    selected_profile_id = profile_lookup[selected_label]
    fit_min, fit_max = settings["fit_range"]
    bundles, bundle_errors = build_subject_bundles(items, selected_profile_id, fit_min, fit_max)
    if not bundles:
        raise RuntimeError("No subjects have a suitable spectral variant for the selected fit range")

    batch_folder = Path(values["-FOLDER-"])
    output_name = values["-OUTPUT_NAME-"].strip() or DEFAULT_OUTPUT_FOLDER
    output_root = batch_folder / output_name / safe_slug(selected_profile_id)
    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging(output_root)
    for error in bundle_errors:
        gui_log(window, error, "warning")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    gui_log(window, f"Preparing PSDs for {len(bundles)} subjects...")
    records: list[PSDRecord] = []
    psd_failures: list[str] = []
    for index, bundle in enumerate(bundles, start=1):
        try:
            record = prepare_psd_record(bundle, settings, output_root)
            records.append(record)
            gui_log(window, f"PSD ready: {bundle.subject} ({record.n_segments} epochs/windows)")
        except Exception as exc:
            message = f"PSD failed for {bundle.subject}: {exc}"
            psd_failures.append(message)
            gui_log(window, message, "error")
        window["-PROGRESS-"].update(index / max(1, len(bundles)) * 40)
        window.refresh()

    if not records:
        raise RuntimeError("PSD preparation failed for all subjects")

    tasks = [
        {
            "cache_path": record.cache_path,
            "fit_settings": settings,
            "run_id": run_id,
            "output_root": str(output_root),
            "plot_scope": settings["plot_scope"],
            "dpi": settings["dpi"],
            "save_native_global": settings["save_native_global"],
        }
        for record in records
    ]

    gui_log(window, f"Fitting {len(tasks)} subjects with {settings['n_workers']} worker(s)...")
    worker_results: list[dict[str, Any]] = []
    if settings["n_workers"] == 1:
        for index, task in enumerate(tasks, start=1):
            worker_results.append(fit_subject_task(task))
            gui_log(window, f"Fitted: {worker_results[-1]['subject']}")
            window["-PROGRESS-"].update(40 + index / len(tasks) * 55)
            window.refresh()
    else:
        with ProcessPoolExecutor(max_workers=settings["n_workers"]) as executor:
            future_map = {executor.submit(fit_subject_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_map), start=1):
                try:
                    result = future.result()
                    worker_results.append(result)
                    gui_log(window, f"Fitted: {result['subject']}")
                except Exception as exc:
                    gui_log(window, f"Subject fitting worker failed: {exc}", "error")
                    LOGGER.error(traceback.format_exc())
                window["-PROGRESS-"].update(40 + index / len(tasks) * 55)
                window.refresh()

    worker_results.sort(key=lambda result: result.get("subject", "").lower())
    if not worker_results:
        raise RuntimeError("No subject fitting results were generated")

    outputs = export_results(
        output_root=output_root,
        run_id=run_id,
        settings=settings,
        items=items,
        selected_profile_id=selected_profile_id,
        bundles=bundles,
        records=records,
        worker_results=worker_results,
        save_csv=settings["save_csv"],
        save_excel=settings["save_excel"],
    )
    if psd_failures:
        (output_root / "psd_failures.txt").write_text("\n".join(psd_failures) + "\n", encoding="utf-8")
    window["-PROGRESS-"].update(100)
    gui_log(window, f"Analysis complete. Outputs: {output_root}")
    return outputs


def main() -> None:
    if sg is None:
        raise SystemExit(
            "FreeSimpleGUI is not installed. Install the same GUI dependency used by EEG-Pype, then rerun."
        )

    window = create_main_window()
    items: list[InventoryItem] = []
    profile_lookup: dict[str, str] = {}
    preview_signature: Optional[str] = None

    def update_resolution(values: dict[str, Any]) -> None:
        try:
            sfreq = float(values["-SFREQ-"])
            length = float(values["-WINDOW_LENGTH-"])
            bandwidth = _parse_optional_float(values["-MT_BANDWIDTH-"], "bandwidth")
            window["-RESOLUTION-"].update(compute_resolution_summary(sfreq, length, bandwidth))
        except Exception:
            window["-RESOLUTION-"].update("Enter valid PSD settings to calculate resolution.")

    _, initial_values = window.read(timeout=0)
    if initial_values:
        update_resolution(initial_values)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Exit"):
            break

        if event in ("-SFREQ-", "-WINDOW_LENGTH-", "-MT_BANDWIDTH-"):
            update_resolution(values)

        if event == "Scan data":
            folder = values["-FOLDER-"].strip()
            if not folder:
                sg.popup_error("Select the preprocessing batch folder first.")
                continue
            try:
                items = scan_eeg_pype_outputs(folder)
                config_info = find_preprocessing_config(folder)
                profiles = summarize_profiles(items)
                ordered_profiles = sorted(
                    profiles.values(),
                    key=lambda profile: (
                        profile["representation"] != "epoched",
                        profile["label"].lower(),
                    ),
                )
                labels = [
                    profile["label"] for profile in ordered_profiles
                ]
                profile_lookup = {
                    profile["label"]: profile["id"]
                    for profile in profiles.values()
                }
                window["-PROFILE-"].update(
                    values=labels,
                    value=labels[0] if labels else "",
                )

                metadata_sfreqs = (
                    trusted_inventory_sampling_frequencies(items)
                )
                metadata_epoch_lengths = (
                    trusted_inventory_epoch_lengths(items)
                )

                if len(metadata_sfreqs) == 1:
                    window["-SFREQ-"].update(
                        f"{metadata_sfreqs[0]:g}"
                    )
                elif config_info.get("sfreq"):
                    window["-SFREQ-"].update(
                        f"{config_info['sfreq']:g}"
                    )

                if len(metadata_epoch_lengths) == 1:
                    window["-WINDOW_LENGTH-"].update(
                        f"{metadata_epoch_lengths[0]:g}"
                    )
                elif (
                    config_info.get("epoch_length")
                    and config_info["epoch_length"] > 0
                ):
                    window["-WINDOW_LENGTH-"].update(
                        f"{config_info['epoch_length']:g}"
                    )
                window["-INVENTORY-"].update(inventory_summary_text(items, config_info))
                update_resolution(values | {"-SFREQ-": window["-SFREQ-"].get(), "-WINDOW_LENGTH-": window["-WINDOW_LENGTH-"].get()})
                preview_signature = None
            except Exception as exc:
                sg.popup_error(f"Could not scan the batch:\n{exc}")

        if event == "Prepare & review preview":
            try:
                ensure_specparam_rc7()
                if not items or not values["-PROFILE-"]:
                    raise ValueError("Scan the data and select an analysis profile first")
                settings = collect_gui_settings(values)
                log_global_settings_warnings(window, settings)
                selected_profile_id = profile_lookup[values["-PROFILE-"]]
                bundles, errors = build_subject_bundles(items, selected_profile_id, *settings["fit_range"])
                for error in errors:
                    window["-LOG-"].update(error + "\n", append=True)
                if not bundles:
                    raise ValueError("No subjects have a suitable spectral variant for this fit range")
                selected = preview_subjects(bundles, settings)
                output_root = Path(values["-FOLDER-"]) / (values["-OUTPUT_NAME-"].strip() or DEFAULT_OUTPUT_FOLDER) / safe_slug(selected_profile_id)
                setup_logging(output_root)
                gui_log(window, f"Preparing preview for {len(selected)} subject(s)...")
                records = []
                for index, bundle in enumerate(selected, start=1):
                    records.append(prepare_psd_record(bundle, settings, output_root))
                    window["-PROGRESS-"].update(index / len(selected) * 50)
                    window.refresh()
                displays = create_preview_displays(records, settings, datetime.now().strftime("preview_%Y%m%d_%H%M%S"))
                window["-PROGRESS-"].update(100)
                review_preview(displays)
                signature_payload = {"profile": selected_profile_id, "settings": settings}
                preview_signature = hashlib.sha256(json.dumps(jsonable(signature_payload), sort_keys=True).encode()).hexdigest()
                gui_log(window, "Preview completed. Change settings and repeat, or run the full batch.")
            except Exception as exc:
                LOGGER.error(traceback.format_exc())
                sg.popup_error(f"Preview could not be completed:\n{exc}")

        if event == "Run full batch & export":
            try:
                ensure_specparam_rc7()
                if not items or not values["-PROFILE-"]:
                    raise ValueError("Scan the data and select an analysis profile first")
                settings = collect_gui_settings(values)
                log_global_settings_warnings(window, settings)
                selected_profile_id = profile_lookup[values["-PROFILE-"]]
                signature_payload = {"profile": selected_profile_id, "settings": settings}
                current_signature = hashlib.sha256(json.dumps(jsonable(signature_payload), sort_keys=True).encode()).hexdigest()
                if current_signature != preview_signature:
                    answer = sg.popup_yes_no(
                        "These exact profile/settings have not been previewed since the last change.\n"
                        "Run the full batch anyway?",
                        title="Preview not current",
                    )
                    if answer != "Yes":
                        continue
                outputs = run_full_batch(window, values, items, profile_lookup, settings)
                sg.popup(
                    "Aperiodic analysis complete.\n\n"
                    + "\n".join(f"{key}: {path}" for key, path in outputs.items()),
                    title="Success",
                )
            except Exception as exc:
                LOGGER.error(traceback.format_exc())
                sg.popup_error(f"Full analysis could not be completed:\n{exc}")

    window.close()


if __name__ == "__main__":
    main()

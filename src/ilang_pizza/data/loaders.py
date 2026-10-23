"""Read an organizer's exports into a canonical, privacy-safe :class:`~ilang_pizza.schema.Dataset`.

Folder layout (see docs/architecture.md)::

    data/<name>/
      events.csv                 one row per event; event_id and start are required
      rsvps/<event_id>.csv       one export per event: Meetup, Luma, ti.to, Google Forms or canonical
      rsvps/<event_id>.xlsx      the same as an Excel workbook (first sheet); Meetup's ".xls"
                                 download, which is really tab-separated text, works too
      attendance/<event_id>.csv  optional check-in list (emails or names)
      weather.csv                optional daily rain: date, precipitation_mm

Privacy: emails and names are hashed with the folder's salt as soon as a file is read, and the
raw columns (event-question answers included) never leave this module. Errors and log lines
mention files, columns and counts, never people.

``attended`` is 1.0 (came), 0.0 (no-show) or NaN (unknown), and unknown is not a no-show. An
event gets zeros only when its export has check-in data or it has an attendance list, and only
if somebody is marked present: a check-in column that is blank (or all "No") means attendance
wasn't taken or the event hasn't happened yet, so it stays NaN. A mark counts as present only
when it is recognizable (yes, ✓, "/", attended, a check-in time or count); anything else is
unknown. Waitlisted RSVPs who checked in count as 1.0; the others stay NaN, because they may
never have been let in.

Timestamps without an offset are read as Asia/Manila local time. An RSVP whose time is missing
gets ``rsvp_at = NaT``, so its lead time is NaN downstream.
"""

from __future__ import annotations

import difflib
import hashlib
import io
import logging
import os
import re
import secrets
import unicodedata
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from ilang_pizza.schema import EVENT_COLUMNS, MANILA_TZ, RSVP_COLUMNS, SERIES, Dataset

log = logging.getLogger(__name__)

SALT_FILE = ".ilang-pizza-salt"
SYNTHETIC_MARKER = "SYNTHETIC"
FORMATS = ("meetup", "luma", "tito", "google_forms", "canonical")

# Resolution of every canonical timestamp column (pandas 3 parses strings to "us" by default).
TIME_UNIT = "us"
_TIMESTAMP = pd.DatetimeTZDtype(unit=TIME_UNIT, tz=MANILA_TZ)

# ".xls" is included because Meetup's classic download is tab-separated text with that name.
_TABLE_SUFFIXES = (".csv", ".tsv", ".xls", ".xlsx", ".xlsm")


class DataError(ValueError):
    """A problem with the organizer's files, worded so they can fix it."""


# --- identity ---------------------------------------------------------------------------------


def _normalize_identity(value: str) -> str:
    """NFKC, casefold and collapse whitespace: " Juan  DELA Cruz" -> "juan dela cruz"."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def hash_identity(value: str, salt: str) -> str:
    """Return the ``attendee_id`` for an email or name: 16 hex chars of a salted SHA-256.

    The value is normalized first, so "Juan@Example.com " and "juan@example.com" match.
    """
    normalized = _normalize_identity(value)
    if not normalized:
        raise ValueError("cannot hash an empty identity")
    return hashlib.sha256(f"{salt}:{normalized}".encode()).hexdigest()[:16]


def get_or_create_salt(data_dir: Path) -> str:
    """Return the folder's salt, creating ``.ilang-pizza-salt`` (owner-only) on first use.

    The salt keeps attendee ids stable across loads of one folder, and makes them useless for
    matching people across folders. Never commit it.
    """
    path = Path(data_dir) / SALT_FILE
    if path.exists():
        salt = path.read_text(encoding="utf-8").strip()
        if salt:
            return salt
    path.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_hex(16)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(salt + "\n")
    return salt


def _hash_column(values: pd.Series, salt: str) -> pd.Series:
    """Hash every non-blank value; blanks become NaN."""
    return values.map(lambda v: hash_identity(v, salt) if v.strip() else np.nan).astype("str")


# --- reading tables ---------------------------------------------------------------------------


def _column_key(name: object) -> str:
    """Normalize a header for matching: "RSVP'd On " -> "rsvpd on", "Email_Address" -> "email address"."""
    text = unicodedata.normalize("NFKC", str(name)).casefold()
    text = re.sub(r"['’`]", "", text)
    return " ".join(re.sub(r"[\W_]+", " ", text).split())


def _value_key(value: str) -> str:
    """Normalize a cell for vocabulary lookups, keeping symbols such as "/" and "✓"."""
    text = unicodedata.normalize("NFKC", value).casefold().replace("-", " ").replace("_", " ")
    return " ".join(text.split())


def _decode(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:  # Excel on Windows saves CSVs as cp1252 (think "Peña")
        return data.decode("cp1252", errors="replace")


def _read_table(path: Path, where: str) -> pd.DataFrame:
    """Read a CSV/TSV/XLSX file with every cell as a stripped string ("" when blank).

    The format comes from the content, not the extension: Meetup's classic ".xls" download is
    really tab-separated text.
    """
    data = path.read_bytes()
    if data.startswith(b"\xd0\xcf\x11\xe0"):  # a genuine pre-2007 binary workbook
        raise DataError(
            f"{where} is an old binary Excel file. Open it in Excel or Google Sheets and save "
            "it as .xlsx or .csv."
        )
    try:
        if data.startswith(b"PK"):  # xlsx is a zip archive
            table = pd.read_excel(
                io.BytesIO(data), dtype=str, keep_default_na=False, engine="openpyxl"
            )
        else:
            text = _decode(data)
            head = text.splitlines()[:5]
            sep = max((",", "\t", ";"), key=lambda s: sum(line.count(s) for line in head))
            table = pd.read_csv(
                io.StringIO(text), sep=sep, dtype=str, keep_default_na=False, index_col=False
            )
    except pd.errors.EmptyDataError:
        raise DataError(f"{where} is empty.") from None
    except Exception as err:  # pandas and openpyxl errors are cryptic; say which file failed
        raise DataError(f"Couldn't read {where} as a table: {err}") from err
    table.columns = _unique_headers(table.columns)
    table = table.apply(lambda col: col.astype("str").str.strip())
    return table[(table != "").any(axis=1)].reset_index(drop=True)


def _unique_headers(columns: Iterable[object]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for col in columns:
        name = str(col).strip()
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return out


def _find(table: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first column whose normalized name is in ``candidates`` (in candidate order)."""
    by_key: dict[str, str] = {}
    for col in table.columns:
        by_key.setdefault(_column_key(col), col)
    return next((by_key[key] for key in candidates if key in by_key), None)


# --- parsing values ---------------------------------------------------------------------------

_BLANK_VALUES = {"", "nan", "nat", "none", "null", "n/a", "na", "-", "--"}

# Google writes "GMT+8"; generic parsers read that POSIX-style, as UTC-8.
_GMT_OFFSET = re.compile(r"\s*\b(?:gmt|utc)\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?\s*$", re.IGNORECASE)
_LOCAL_ZONE = re.compile(r"\s*\b(?:pht|asia/manila)\s*$", re.IGNORECASE)
_HAS_OFFSET = re.compile(
    r"\d:\d{2}(?::\d{2}(?:[.,]\d+)?)?\s*(?:[ap]\.?m\.?)?\s*(?:z|utc|gmt|[+-]\d{2}(?::?\d{2})?)$",
    re.IGNORECASE,
)
_SLASH_DATE = re.compile(r"^\s*(\d{1,2})[/.](\d{1,2})[/.]\d{2,4}\b")


def _blank(values: pd.Series) -> pd.Series:
    return values.str.strip().str.casefold().isin(_BLANK_VALUES)


def _parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse export timestamps into tz-aware Asia/Manila; blank or unreadable cells become NaT.

    Reads ISO 8601 (with or without offset), "9/17/2026 18:30:12" (day-first when some day is
    above 12), "Sep 1, 2023 10:22 AM", "... UTC", Google's "GMT+8" and "PHT". Values without an
    offset are Manila local time. Zone names like "PST" are ambiguous (Google means Pacific)
    and are left unreadable rather than guessed.
    """
    text = values.astype("str").fillna("").str.strip()
    text = text.str.replace(
        _GMT_OFFSET, lambda m: f" {m[1]}{int(m[2]):02d}:{m[3] or '00'}", regex=True
    )
    text = text.str.replace(_LOCAL_ZONE, "", regex=True)
    filled = ~_blank(text)
    aware = filled & text.str.contains(_HAS_OFFSET)
    dayfirst = bool((text.str.extract(_SLASH_DATE)[0].astype(float) > 12).any())
    parts = []
    if aware.any():
        parts.append(
            _to_datetime(text[aware], utc=True, dayfirst=dayfirst).dt.tz_convert(MANILA_TZ)
        )
    if (filled & ~aware).any():
        naive = _to_datetime(text[filled & ~aware], utc=False, dayfirst=dayfirst)
        parts.append(naive.dt.tz_localize(MANILA_TZ))
    if not parts:
        return pd.Series(pd.NaT, index=values.index, dtype=_TIMESTAMP)
    parsed = pd.concat([p.dt.as_unit(TIME_UNIT) for p in parts])
    return parsed.reindex(values.index).astype(_TIMESTAMP)


def _to_datetime(text: pd.Series, *, utc: bool, dayfirst: bool) -> pd.Series:
    """Fast ISO 8601 pass first, then a per-value "mixed" pass for whatever is left."""
    parsed = pd.to_datetime(text, format="ISO8601", utc=utc, errors="coerce")
    retry = parsed.isna()
    if retry.any():
        with warnings.catch_warnings():  # "could not infer format" chatter for odd formats
            warnings.simplefilter("ignore", UserWarning)
            again = pd.to_datetime(
                text[retry], format="mixed", dayfirst=dayfirst, utc=utc, errors="coerce"
            )
        parsed = parsed.where(~retry, again)
    return parsed


def _parse_times_column(values: pd.Series, *, column: str, where: str) -> pd.Series:
    """Parse a timestamp column, failing loudly only when most of it is unreadable."""
    parsed = _parse_timestamps(values)
    failed = parsed.isna() & ~_blank(values)
    if failed.any():
        examples = ", ".join(repr(v[:40]) for v in values[failed].unique()[:3])
        if failed.sum() > 0.5 * (~_blank(values)).sum():
            raise DataError(
                f"{where}: couldn't read the times in column {column!r} (e.g. {examples}). "
                "Use a format like 2026-09-17 18:30 or 2026-09-17T18:30:00+08:00."
            )
        log.warning(
            "%s: %d unreadable times in column %r (e.g. %s); their lead time will be unknown",
            where,
            int(failed.sum()),
            column,
            examples,
        )
    return parsed


_TRUE_WORDS = {"yes", "y", "true", "t", "1", "1.0", "✓", "✔"}
_FALSE_WORDS = {"no", "n", "false", "f", "0", "0.0"}


def _parse_bool(values: pd.Series, *, column: str, where: str) -> pd.Series:
    """yes/no/true/false/1/0 -> True/False; blank -> NA so the caller can pick a default."""
    keys = values.map(_value_key)
    bad = sorted(set(keys) - _TRUE_WORDS - _FALSE_WORDS - _BLANK_VALUES)
    if bad:
        raise DataError(
            f"{where}: column {column!r} should be yes/no (or true/false, 1/0); "
            f"found {', '.join(repr(b) for b in bad[:5])}."
        )
    out = pd.Series(pd.NA, index=values.index, dtype="boolean")
    out[keys.isin(_TRUE_WORDS)] = True
    out[keys.isin(_FALSE_WORDS)] = False
    return out


def _parse_number(values: pd.Series, *, column: str, where: str) -> pd.Series:
    """Numbers like "60", "1,000" or "60.0"; blank -> NaN."""
    cleaned = values.str.replace(",", "", regex=False).str.strip()
    numbers = pd.to_numeric(cleaned.where(~_blank(cleaned)), errors="coerce")
    bad = numbers.isna() & ~_blank(cleaned)
    if bad.any():
        examples = ", ".join(repr(v[:20]) for v in values[bad].unique()[:3])
        raise DataError(f"{where}: column {column!r} should hold numbers; found {examples}.")
    return numbers.astype("float64")


def _parse_guests(values: pd.Series) -> pd.Series:
    """Extra heads per RSVP; blank or unreadable -> 0."""
    numbers = pd.to_numeric(values.str.strip().str.lstrip("+"), errors="coerce")
    return numbers.fillna(0).clip(lower=0).round().astype("int64")


# A check-in cell counts only when it plainly says the person came. Blanks, "No", unfamiliar
# text and "x" (absent, in Philippine attendance sheets, where "/" is present) all don't.
_PRESENT_MARKS = {"yes", "y", "true", "1", "/", "✓", "✔", "attended", "present", "checked in"}


def _checked_in(values: pd.Series) -> pd.Series:
    """True where a cell is a known "present" mark, a check-in time or a check-in count > 0.

    Unfamiliar text, such as "Onsite" from a registration question, is not a check-in.
    """
    keys = values.map(_value_key)
    count = pd.to_numeric(values, errors="coerce")
    stamped = values.str.contains(r"\d", regex=True) & _parse_timestamps(values).notna()
    return keys.isin(_PRESENT_MARKS) | (count > 0) | stamped


# RSVP words of every platform in one vocabulary (they don't clash). None: the RSVP doesn't count.
_GOING = ("yes", "going", "attending", "approved", "registered", "confirmed", "accepted")
_TICKETED = ("complete", "completed", "paid", "checked in", "incomplete", "unassigned")
_WAITLISTED = ("waitlist", "waitlisted", "wait list", "waiting list", "waiting", "on waitlist")
_DROPPED = ("no", "not going", "declined", "rejected", "cancelled", "canceled", "maybe")
_UNCONFIRMED = ("invited", "pending", "pending approval", "void", "voided", "refunded")
_STATUS_WORDS: dict[str, str | None] = {
    **dict.fromkeys(_GOING + _TICKETED, "going"),
    **dict.fromkeys(_WAITLISTED, "waitlist"),
    **dict.fromkeys(_DROPPED + _UNCONFIRMED, None),
}


def _parse_status(values: pd.Series, *, column: str, where: str) -> pd.Series:
    """Map platform RSVP words to "going"/"waitlist"; dropped RSVPs become NaN."""
    keys = values.map(_value_key)
    unknown = sorted(set(keys) - set(_STATUS_WORDS))
    if unknown:
        shown = ", ".join(repr(k) if k else "(blank)" for k in unknown[:5])
        raise DataError(
            f"{where}: unknown RSVP status in column {column!r}: {shown}. Expected words like "
            "Yes / No / Waitlist (Meetup), approved / waitlist / declined (Luma), or "
            "going / waitlist (canonical)."
        )
    return keys.map(_STATUS_WORDS).astype("str")


# --- export formats ---------------------------------------------------------------------------
# Column names are matched after _column_key(), so "RSVPed on", "rsvped_on" and "RSVPED ON " are
# the same.

_EMAIL = ("email", "email address", "e mail", "e mail address", "your email", "email add")
_NAME = (
    "name",
    "full name",
    "fullname",
    "your name",
    "complete name",
    "enter your name",
    "name first and last",
    "first and last name",
    "name optional",
    "participant name",
    "attendee name",
    "guest name",
    "pangalan",
    "buong pangalan",
)
_FIRST_NAME = ("first name", "firstname", "given name")
_LAST_NAME = ("last name", "lastname", "surname", "family name", "last name or surname")
_GUESTS = ("guests", "guest count", "number of guests", "plus ones")
# Columns an organizer adds to an export to record check-ins. "Attendance" and "Present" are
# left out: as registration questions they usually mean "will you attend (onsite)?".
_CHECKED_IN = ("attended", "checked in", "checked in at", "check in time", "showed up")

_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    # Meetup has no per-event check-in column ("Events attended" and "No shows" are all-time
    # totals and would leak the future), so only a column the organizer added counts.
    "meetup": {
        "email": ("email",),
        "name": _NAME,
        "first_name": ("first name",),
        "last_name": ("last name",),
        "rsvp_at": ("rsvped on", "rsvpd on"),
        "status": ("rsvp",),
        "guests": ("guests",),
        "attended": _CHECKED_IN,
    },
    # Luma has no guest count: a group registration repeats the guest once per ticket.
    "luma": {
        "email": ("email",),
        "name": ("name",),
        "first_name": ("first name",),
        "last_name": ("last name",),
        "rsvp_at": ("created at", "registered at"),
        "status": ("approval status",),
        "attended": ("checked in at", "has joined event"),  # the latter for online events
    },
    # ti.to: one row per ticket; check-ins are "Check-ins: <list title>" columns (see below).
    "tito": {
        "email": ("ticket email", "email"),
        "name": ("ticket full name", "name"),
        "first_name": ("ticket first name",),
        "last_name": ("ticket last name",),
        "order_email": ("order email",),
        "order_name": ("order name",),
        "rsvp_at": ("ticket created date", "created date", "order created date"),
        "status": ("void status", "ticket state", "state"),
        "attended": _CHECKED_IN,
    },
    # Sheets writes "Email Address"; the Forms "Download responses (.csv)" file, "Username".
    "google_forms": {
        "email": (*_EMAIL, "username"),
        "name": _NAME,
        "first_name": _FIRST_NAME,
        "last_name": _LAST_NAME,
        "rsvp_at": ("timestamp", "submitted at", "submission time"),
        "guests": _GUESTS,
        "attended": _CHECKED_IN,
    },
    "canonical": {
        "event_id": ("event id",),
        "attendee_id": ("attendee id",),
        "email": _EMAIL,
        "name": _NAME,
        "rsvp_at": ("rsvp at",),
        "status": ("status",),
        "guests": ("guests",),
        "attended": ("attended",),
        "source": ("source",),
    },
}

# A column only one platform writes is enough to recognize it.
_MARKERS = {
    "luma": {
        "approval status",
        "has joined event",
        "eth address",
        "solana address",
        "survey response rating",
        "survey response feedback",
    },
    "tito": {
        "ticket full name",
        "ticket first name",
        "ticket email",
        "ticket created date",
        "ticket reference",
        "order reference",
        "unique ticket url",
        "void status",
    },
    "meetup": {"rsvped on", "rsvpd on"},
}
# Meetup's member-list export shares these but has no RSVP column.
_MEETUP_MEMBER_COLUMNS = {"member id", "user id", "joined group on", "url of member profile"}

_LABELS = {
    "meetup": "Meetup",
    "luma": "Luma",
    "tito": "ti.to",
    "google_forms": "Google Forms",
    "canonical": "canonical",
}
_FORMAT_HELP = (
    "Supported exports: Meetup attendee list (Name, Member ID, RSVP, RSVPed on, Guests); "
    "Luma guest CSV (name, email, created_at, approval_status, checked_in_at); "
    "ti.to tickets (Ticket Full Name, Ticket Email, Ticket Created Date); "
    "Google Forms responses (Timestamp, Email Address, Name); "
    "or canonical (rsvp_at, status, plus attendee_id, email or name)."
)


def detect_format(columns: Iterable[str]) -> str:
    """Name the platform that wrote an export, from its column names alone.

    Returns one of ``FORMATS``. "google_forms" also covers any plain sign-up sheet with a
    Timestamp, email or name column. Raises :class:`DataError` when nothing fits.
    """
    columns = list(columns)
    keys = {_column_key(c) for c in columns}
    if {"rsvp at", "status"} <= keys and keys & {"attendee id", "email", "name"}:
        return "canonical"
    for fmt in ("luma", "tito", "meetup"):
        if keys & _MARKERS[fmt]:
            return fmt
    if keys & _MEETUP_MEMBER_COLUMNS:
        if "rsvp" in keys:
            return "meetup"
        raise DataError(
            "This looks like a Meetup member list, not the attendee list of one event. On the "
            "event's page, open Manage attendees and download the attendee details."
        )
    if "timestamp" in keys or keys & set(_EMAIL + _NAME + _FIRST_NAME):
        return "google_forms"
    seen = ", ".join(repr(str(c)) for c in columns[:12]) or "(none)"
    raise DataError(
        f"Couldn't tell which platform this export comes from. Columns seen: {seen}. {_FORMAT_HELP}"
    )


# --- export -> canonical RSVP rows ------------------------------------------------------------
# Rows carry two private columns while loading, used to match check-in lists: _email_key and
# _name_key (salted hashes, never raw values). They are dropped before anything is returned.


def _identity_columns(
    table: pd.DataFrame, fields: dict[str, tuple[str, ...]], prefix: str = ""
) -> tuple[pd.Series, pd.Series]:
    """Raw (email, name) per row; a first + last name pair stands in for a full-name column."""
    blank = pd.Series("", index=table.index, dtype="str")
    email_col = _find(table, fields.get(f"{prefix}email", ()))
    name_col = _find(table, fields.get(f"{prefix}name", ()))
    emails = table[email_col] if email_col else blank
    if name_col:
        names = table[name_col]
    else:
        first = _find(table, fields.get(f"{prefix}first_name", ()))
        last = _find(table, fields.get(f"{prefix}last_name", ()))
        names = blank
        if first or last:
            names = (table[first] if first else blank) + " " + (table[last] if last else blank)
    # An email without "@" is a placeholder such as Meetup's "Email hidden • Upgrade to Pro",
    # a name like "-" or "n/a" is no name, and people type emails into name fields.
    emails = emails.where(emails.str.contains("@", regex=False), "")
    names = names.where(~_blank(names), "")
    misplaced = (emails == "") & names.str.contains("@", regex=False)
    return emails.where(~misplaced, names), names.where(~misplaced, "").str.strip()


def _platform_rows(
    table: pd.DataFrame, fmt: str, event_id: str, salt: str, where: str
) -> pd.DataFrame:
    """Canonical rows (plus match keys and a check-in mark) from a platform export."""
    fields = _FIELDS[fmt]
    emails, names = _identity_columns(table, fields)
    if fmt == "tito":  # unassigned tickets have no attendee yet; they belong to the buyer
        order_emails, order_names = _identity_columns(table, fields, prefix="order_")
        unassigned = (emails == "") & (names == "")
        emails = emails.where(~unassigned, order_emails)
        names = names.where(~unassigned, order_names)
    member_ids = _meetup_member_ids(table) if fmt == "meetup" else None
    no_member_ids = member_ids is None or (member_ids == "").all()
    if (emails == "").all() and (names == "").all() and no_member_ids:
        seen = ", ".join(repr(c) for c in table.columns[:12])
        raise DataError(
            f"{where}: found no name or email column in this {_LABELS[fmt]} export. "
            f"Columns seen: {seen}."
        )

    rows = pd.DataFrame(index=table.index)
    rows["event_id"] = event_id
    rows["_email_key"] = _hash_column(emails, salt)
    rows["_name_key"] = _hash_column(names, salt)
    attendee = rows["_email_key"]
    if member_ids is not None:
        # Meetup's id survives renames and both export layouts, and keeps two "Juan Dela Cruz"
        # apart, so it beats the email (which only Pro groups can see).
        members = ("meetup-member:" + member_ids).where(member_ids != "", "")
        attendee = _hash_column(members, salt).fillna(attendee)
    rows["attendee_id"] = attendee.fillna(rows["_name_key"])

    time_col = _find(table, fields.get("rsvp_at", ()))
    if time_col is None:
        rows["rsvp_at"] = pd.Series(pd.NaT, index=table.index, dtype=_TIMESTAMP)
    else:
        rows["rsvp_at"] = _parse_times_column(table[time_col], column=time_col, where=where)
    rows["status"] = _platform_status(table, fmt, where)
    guests_col = _find(table, fields.get("guests", ()))
    rows["guests"] = _parse_guests(table[guests_col]) if guests_col else 0
    rows["_present"] = _present(table, fmt)
    rows["source"] = fmt

    no_identity = rows["attendee_id"].isna()
    if no_identity.any():
        log.warning("%s: skipped %d rows with no name or email", where, int(no_identity.sum()))
    rows = rows[~no_identity & rows["status"].notna()]
    # Luma and ti.to repeat a person once per ticket, so their duplicate rows are guests.
    return _merge_duplicates(rows, heads_add_up=fmt in ("luma", "tito"), where=where)


def _present(table: pd.DataFrame, fmt: str) -> pd.Series:
    """Whether each row checked in; on ti.to, a check-in on any list counts."""
    if fmt == "tito":
        lists = [c for c in table.columns if _column_key(c).startswith("check ins")]
        if lists:
            return pd.concat([_checked_in(table[c]) for c in lists], axis=1).any(axis=1)
    col = _find(table, _FIELDS[fmt].get("attended", ()))
    return _checked_in(table[col]) if col else pd.Series(False, index=table.index)


_PROFILE_ID = re.compile(r"/members/(\d+)")


def _meetup_member_ids(table: pd.DataFrame) -> pd.Series:
    """Meetup's numeric member id per row ("" when unknown), the same in old and new exports.

    The profile URL always holds it; the old "User ID" column says "user 228332596" or a custom
    handle, the newer "Member ID" a bare number.
    """
    ids = pd.Series("", index=table.index, dtype="str")
    url_col = _find(table, ("url of member profile",))
    if url_col:
        ids = table[url_col].str.extract(_PROFILE_ID)[0].fillna("")
    id_col = _find(table, ("member id", "user id"))
    if id_col:
        given = table[id_col].str.replace(r"^user\s+", "", regex=True, case=False)
        ids = ids.where(ids != "", given)
    return ids


def _platform_status(table: pd.DataFrame, fmt: str, where: str) -> pd.Series:
    """ "going", "waitlist" or NaN (dropped) per row."""
    going = pd.Series("going", index=table.index, dtype="str")
    col = _find(table, _FIELDS[fmt].get("status", ()))
    if col is None:  # Google Forms has no status; a ticket without a state is a ticket
        return going
    if fmt == "tito" and _column_key(col) == "void status":
        voided = ~_blank(table[col]) & ~table[col].map(_value_key).isin(_FALSE_WORDS)
        return going.where(~voided)
    return _parse_status(table[col], column=col, where=where)


def _merge_duplicates(rows: pd.DataFrame, *, heads_add_up: bool, where: str) -> pd.DataFrame:
    """One row per person per event, keeping the earliest RSVP.

    With ``heads_add_up`` (one row per ticket), the extra rows become guests and the party
    counts as present if any of its tickets checked in. Otherwise they are double submissions.
    """
    key = ["event_id", "attendee_id"]
    rows = rows.sort_values("rsvp_at", kind="stable", na_position="last")
    duplicated = rows.duplicated(key)
    if not duplicated.any():
        return rows
    if heads_add_up:
        grouped = rows.groupby(key, sort=False)
        merged = grouped.first()  # earliest row's values, keys from the first ticket
        merged["guests"] = grouped["guests"].sum() + grouped.size() - 1
        merged["status"] = grouped["status"].agg(
            lambda s: "going" if (s == "going").any() else s.iloc[0]
        )
        merged["_present"] = grouped["_present"].any()
        rows = merged.reset_index()
        log.info("%s: counted %d extra tickets as guests", where, int(duplicated.sum()))
    else:
        rows = rows[~duplicated]
        log.info(
            "%s: dropped %d repeated RSVPs (kept each person's first)", where, int(duplicated.sum())
        )
    return rows


def _canonical_rows(table: pd.DataFrame, event_id: str, salt: str, where: str) -> pd.DataFrame:
    """Rows from a file already in canonical form (e.g. written by ``write_dataset``)."""
    fields = _FIELDS["canonical"]
    rows = pd.DataFrame(index=table.index)
    id_col = _find(table, fields["event_id"])
    rows["event_id"] = table[id_col].where(table[id_col] != "", event_id) if id_col else event_id
    emails, names = _identity_columns(table, fields)
    rows["_email_key"] = _hash_column(emails, salt)
    rows["_name_key"] = _hash_column(names, salt)
    attendee = rows["_email_key"].fillna(rows["_name_key"])
    given_col = _find(table, fields["attendee_id"])
    if given_col:
        given = table[given_col]
        # An id that is really an email or a name gets hashed like one; hashes pass through.
        looks_raw = given.str.contains(r"@|\s", regex=True)
        hashed = _hash_column(given.where(looks_raw, ""), salt)
        attendee = given.where(given != "").where(~looks_raw, hashed).fillna(attendee)
    rows["attendee_id"] = attendee
    time_col = _find(table, fields["rsvp_at"])
    rows["rsvp_at"] = _parse_times_column(table[time_col], column=time_col, where=where)
    status_col = _find(table, fields["status"])
    rows["status"] = _parse_status(table[status_col], column=status_col, where=where)
    guests_col = _find(table, fields["guests"])
    rows["guests"] = _parse_guests(table[guests_col]) if guests_col else 0
    attended_col = _find(table, fields["attended"])
    rows["attended"] = (
        _parse_bool(table[attended_col], column=attended_col, where=where).astype("float64")
        if attended_col
        else np.nan
    )
    source_col = _find(table, fields["source"])
    rows["source"] = (
        table[source_col].where(table[source_col] != "", "canonical") if source_col else "canonical"
    )
    no_identity = rows["attendee_id"].isna()
    if no_identity.any():
        log.warning("%s: skipped %d rows with no attendee_id", where, int(no_identity.sum()))
    rows = rows[~no_identity & rows["status"].notna()]
    return _merge_duplicates(rows, heads_add_up=False, where=where)


def _attended_from_checkins(rows: pd.DataFrame) -> pd.Series:
    """``attended`` from each row's check-in, per event (rules in the module docstring)."""
    present = rows["_present"]
    tracked = present.groupby(rows["event_id"]).transform("any")
    attended = pd.Series(np.nan, index=rows.index)
    attended[tracked & (rows["status"] == "going")] = 0.0
    attended[tracked & present] = 1.0
    return attended


def _read_export(path: Path, event_id: str, salt: str, where: str) -> tuple[pd.DataFrame, str]:
    """Canonical rows (with private match keys) and the detected format of one export."""
    table = _read_table(path, where)
    try:
        fmt = detect_format(table.columns)
    except DataError as err:
        raise DataError(f"{where}: {err}") from None
    if fmt == "canonical":
        return _canonical_rows(table, event_id, salt, where), fmt
    rows = _platform_rows(table, fmt, event_id, salt, where)
    rows["attended"] = _attended_from_checkins(rows)
    return rows.drop(columns="_present"), fmt


# --- check-in lists ---------------------------------------------------------------------------

_CHECKIN_FIELDS = {
    "email": ("ticket email", *_EMAIL),
    "name": ("ticket full name", *_NAME),
    "first_name": ("ticket first name", *_FIRST_NAME),
    "last_name": ("ticket last name", *_LAST_NAME),
}


def _read_checkins(path: Path, salt: str, where: str) -> tuple[set[str], set[str], set[str]]:
    """Hashed (emails, names of people listed without an email, all names) who checked in."""
    table = _read_table(path, where)
    emails, names = _identity_columns(table, _CHECKIN_FIELDS)
    if (emails == "").all() and (names == "").all():
        if len(table.columns) != 1:
            seen = ", ".join(repr(c) for c in table.columns[:12])
            raise DataError(
                f"{where}: expected a column of emails or names (e.g. 'Email' or 'Name'). "
                f"Columns seen: {seen}."
            )
        # A bare list with no header row: the "header" is the first person.
        values = pd.Series([table.columns[0], *table.iloc[:, 0]], dtype="str")
        is_email = values.str.contains("@", regex=False)
        emails, names = values.where(is_email, ""), values.where(~is_email, "")
    else:
        mark_col = _find(
            table, ("has joined event", *_CHECKED_IN, "attendance", "present", "check in")
        )
        if mark_col:  # a full guest export: only rows marked present checked in
            present = _checked_in(table[mark_col])
            emails, names = emails[present], names[present]
    email_keys = _hash_column(emails, salt)
    name_keys = _hash_column(names, salt)
    return (
        set(email_keys.dropna()),
        set(name_keys[email_keys.isna()].dropna()),
        set(name_keys.dropna()),
    )


def _apply_checkins(
    rows: pd.DataFrame, path: Path, event_id: str, salt: str, where: str
) -> pd.DataFrame:
    """Fill ``attended`` for one event from its check-in list, unless the export already did.

    Emails are matched to emails; names are matched when one side has no email.
    """
    in_event = rows["event_id"] == event_id
    if not in_event.any():
        return rows
    if rows.loc[in_event, "attended"].notna().any():
        log.info("%s: the RSVP export already has check-ins, so this list is not used", where)
        return rows
    emails, names_without_email, names = _read_checkins(path, salt, where)
    if not (emails or names):
        log.warning("%s: nobody is marked as checked in; attendance stays unknown", where)
        return rows
    has_email = rows["_email_key"].notna()
    matched = (
        rows["_email_key"].isin(emails)
        | rows["_name_key"].isin(names_without_email)
        | (~has_email & rows["_name_key"].isin(names))
        | rows["attendee_id"].isin(emails | names)
    )
    going = rows["status"] == "going"
    rows = rows.copy()
    rows.loc[in_event & going, "attended"] = 0.0
    rows.loc[in_event & matched, "attended"] = 1.0
    log.info(
        "%s: matched %d of %d RSVPs to the check-in list",
        where,
        int((in_event & matched).sum()),
        int(in_event.sum()),
    )
    return rows


# --- events and weather -----------------------------------------------------------------------

_EVENT_ALIASES = {
    "id": "event_id",
    "event": "event_id",
    "name": "title",
    "event_name": "title",
    "event_title": "title",
    "start_time": "start",
    "starts_at": "start",
    "start_at": "start",
    "start_datetime": "start",
    "datetime": "start",
    "date_time": "start",
    "location": "venue",
    "place": "venue",
    "lat": "latitude",
    "lon": "longitude",
    "lng": "longitude",
    "long": "longitude",
    "online": "is_online",
    "virtual": "is_online",
    "food": "food_provided",
    "with_food": "food_provided",
    "has_food": "food_provided",
    "max_capacity": "capacity",
    "rsvp_limit": "capacity",
    "limit": "capacity",
    "max_attendees": "capacity",
}
_ONLINE_WORDS = re.compile(
    r"online|zoom|google meet|meet\.google|virtual|livestream|webinar", re.IGNORECASE
)
_UNLIMITED = {"unlimited", "none", "no limit", "∞"}
_EVENTS_HELP = (
    "Expected columns: event_id and start (required), plus any of title, series, venue, city, "
    "latitude, longitude, is_online, capacity, food_provided."
)


def _compact(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())


def _series_for(series: str, event_id: str, title: str) -> str:
    """Match a series name loosely ("wfh-pythonista" -> "WFH Pythonista"); guess it if blank."""
    names = {_compact(name): name for name in SERIES}
    if series.strip():
        return names.get(_compact(series), "Other")
    haystack = _compact(f"{event_id} {title}")
    return next(
        (name for key, name in names.items() if name != "Other" and key in haystack), "Other"
    )


def _read_events(path: Path) -> pd.DataFrame:
    where = "events.csv"
    if not path.exists():
        raise DataError(
            f"No events.csv in {path.parent}. {_EVENTS_HELP} "
            "Run `ilang-pizza demo` to see a complete example folder."
        )
    table = _read_table(path, where)
    keys = {col: _column_key(col).replace(" ", "_") for col in table.columns}
    exact = set(keys.values()) & set(EVENT_COLUMNS)
    renamed = {
        col: key
        if key in exact or _EVENT_ALIASES.get(key) in exact
        else _EVENT_ALIASES.get(key, key)
        for col, key in keys.items()
    }
    table = table.rename(columns=renamed)
    table = table.loc[:, ~table.columns.duplicated()]  # e.g. both "lon" and "lng": keep the first
    if "start" not in table and "date" in table and "time" in table:
        table["start"] = table["date"] + " " + table["time"]
    missing = [c for c in ("event_id", "start") if c not in table]
    if missing:
        seen = ", ".join(repr(c) for c in table.columns) or "(none)"
        raise DataError(
            f"{where} is missing required column(s): {', '.join(missing)}. "
            f"Columns seen: {seen}. {_EVENTS_HELP}"
        )
    for col in (
        "title",
        "series",
        "venue",
        "city",
        "latitude",
        "longitude",
        "is_online",
        "capacity",
        "food_provided",
    ):
        if col not in table:
            table[col] = ""

    events = pd.DataFrame(index=table.index)
    events["event_id"] = table["event_id"]
    if (events["event_id"] == "").any():
        raise DataError(f"{where}: {(events['event_id'] == '').sum()} row(s) have no event_id.")
    repeated = sorted(set(events["event_id"][events["event_id"].duplicated()]))
    if repeated:
        raise DataError(f"{where}: event_id must be unique; repeated: {', '.join(repeated[:5])}.")
    events["start"] = _parse_timestamps(table["start"])
    unreadable = events["start"].isna()
    if unreadable.any():
        bad = ", ".join(
            f"{eid} ({raw!r})"
            for eid, raw in zip(events["event_id"][unreadable], table["start"][unreadable])
        )
        raise DataError(
            f"{where}: couldn't read the start of {bad}. "
            "Use a format like 2026-09-17 18:30 (Manila time) or 2026-09-17T18:30:00+08:00."
        )
    events["title"] = table["title"].where(table["title"] != "", events["event_id"])
    events["series"] = [
        _series_for(s, e, t)
        for s, e, t in zip(table["series"], events["event_id"], events["title"])
    ]
    events["venue"] = table["venue"]
    events["city"] = table["city"]
    events["latitude"] = _parse_number(table["latitude"], column="latitude", where=where)
    events["longitude"] = _parse_number(table["longitude"], column="longitude", where=where)
    said_online = (table["venue"] + " " + table["city"]).str.contains(_ONLINE_WORDS)
    guessed_online = said_online | ((table["venue"] == "") & (events["series"] == "WFH Pythonista"))
    online = _parse_bool(table["is_online"], column="is_online", where=where)
    events["is_online"] = online.fillna(guessed_online).astype(bool)
    capacity = table["capacity"].where(~table["capacity"].map(_value_key).isin(_UNLIMITED), "")
    capacity = _parse_number(capacity, column="capacity", where=where)
    events["capacity"] = capacity.where(capacity > 0)  # 0 or less means "no limit"
    food = _parse_bool(table["food_provided"], column="food_provided", where=where)
    events["food_provided"] = food.fillna(~events["is_online"]).astype(bool)
    # schema: online events have no venue, city or coordinates
    for col, empty in (("venue", ""), ("city", ""), ("latitude", np.nan), ("longitude", np.nan)):
        events[col] = events[col].where(~events["is_online"], empty)
    return canonical_events(events)


_RAIN_COLUMNS = (
    "precipitation mm",
    "precipitation",
    "precipitation sum",
    "precipitation sum mm",
    "rainfall mm",
    "rainfall",
    "rain mm",
    "rain",
    "precip mm",
    "precip",
)


def _read_weather(path: Path) -> pd.DataFrame:
    """Daily rain from weather.csv (date, precipitation_mm), PAGASA conventions included."""
    where = "weather.csv"
    table = _read_table(path, where)
    date_col = _find(table, ("date", "day", "time"))
    rain_col = _find(table, _RAIN_COLUMNS)
    if date_col is None or rain_col is None:
        seen = ", ".join(repr(c) for c in table.columns)
        raise DataError(
            f"{where}: expected columns date and precipitation_mm (daily rain in mm). "
            f"Columns seen: {seen}."
        )
    days = _parse_timestamps(table[date_col])
    rain = table[rain_col].str.casefold().replace({"t": "0", "trace": "0"})
    rain = pd.to_numeric(rain, errors="coerce")
    rain = rain.mask(rain == -1, 0.0)  # PAGASA writes -1 for a trace of rain
    rain = rain.mask(rain < 0)  # and -999 for missing
    frame = pd.DataFrame({"date": days.dt.date, "precipitation_mm": rain}).dropna()
    if frame.empty and len(table):
        raise DataError(f"{where}: none of the rows has a readable date and rain amount.")
    if len(frame) < len(table):
        log.warning(
            "%s: skipped %d rows with no readable date or rain", where, len(table) - len(frame)
        )
    # several rows per day (e.g. one per station) are averaged
    return canonical_weather(frame.groupby("date", as_index=False)["precipitation_mm"].mean())


# --- canonical frames -------------------------------------------------------------------------


def _as_manila(values: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(values)
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize(MANILA_TZ)
    return stamps.dt.tz_convert(MANILA_TZ).dt.as_unit(TIME_UNIT)


def canonical_events(events: pd.DataFrame) -> pd.DataFrame:
    """``events`` with the schema's columns and dtypes, ordered by start."""
    out = events.loc[:, EVENT_COLUMNS].copy()
    for col in ("event_id", "title", "series", "venue", "city"):
        out[col] = out[col].fillna("").astype("str")
    out["start"] = _as_manila(out["start"])
    for col in ("latitude", "longitude", "capacity"):
        out[col] = pd.to_numeric(out[col]).astype("float64")
    for col in ("is_online", "food_provided"):
        out[col] = out[col].astype(bool)
    return out.sort_values(["start", "event_id"]).reset_index(drop=True)


def canonical_rsvps(rsvps: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """``rsvps`` with the schema's columns and dtypes, grouped by event and in RSVP order.

    Events follow ``events``' order (start order) when given, else event_id order. Within an
    event, rows are in RSVP time order, so the waitlist reads as a queue; missing times last.
    """
    out = rsvps.loc[:, RSVP_COLUMNS].copy()
    for col in ("event_id", "attendee_id", "status", "source"):
        out[col] = out[col].astype("str")
    out["rsvp_at"] = _as_manila(out["rsvp_at"])
    out["guests"] = out["guests"].fillna(0).astype("int64")
    out["attended"] = out["attended"].astype("float64")
    if events is not None:
        rank = out["event_id"].map({e: i for i, e in enumerate(events["event_id"])})
    else:
        rank = out["event_id"]
    order = pd.DataFrame({"rank": rank, "rsvp_at": out["rsvp_at"]}, index=out.index)
    order = order.sort_values(["rank", "rsvp_at"], kind="stable", na_position="last")
    return out.loc[order.index].reset_index(drop=True)


def canonical_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Daily rain with ``date`` as ``datetime.date`` and ``precipitation_mm`` as float, by date."""
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(weather["date"]).dt.date.astype(object),
            "precipitation_mm": weather["precipitation_mm"].astype("float64"),
        }
    )
    return out.sort_values("date").reset_index(drop=True)


# --- the whole folder -------------------------------------------------------------------------


def _table_files(folder: Path) -> list[Path]:
    """Spreadsheet-like files in ``folder`` and below, skipping hidden and Excel lock files."""
    if not folder.is_dir():
        return []
    return [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in _TABLE_SUFFIXES
        and not path.name.startswith((".", "~$"))
    ]


def _unknown_event_message(where: str, unknown: list[str], known: set[str]) -> str:
    close = difflib.get_close_matches(unknown[0], sorted(known), n=3)
    hint = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
    more = f" (and {len(unknown) - 1} more)" if len(unknown) > 1 else ""
    return (
        f"{where}: there is no event {unknown[0]!r}{more} in events.csv.{hint} "
        "Name each export after its event_id (e.g. rsvps/pymeets-2026-09-17.csv), "
        "or add the event to events.csv."
    )


def load_rsvp_export(
    path: Path, event_id: str, *, salt: str, attendance_path: Path | None = None
) -> pd.DataFrame:
    """Read one export into canonical RSVP rows (``schema.RSVP_COLUMNS``) for ``event_id``.

    A canonical file's own event_id column wins over the argument. ``attendance_path`` is a
    check-in list, used only when the export itself has no check-ins.
    """
    path = Path(path)
    rows, _ = _read_export(path, event_id, salt, where=path.name)
    if attendance_path is not None:
        attendance_path = Path(attendance_path)
        rows = _apply_checkins(rows, attendance_path, event_id, salt, where=attendance_path.name)
    return canonical_rsvps(rows)


def load_dataset(data_dir: Path | str) -> Dataset:
    """Load events, RSVP exports, check-in lists and rain from an organizer's data folder."""
    data_dir = Path(data_dir).expanduser()
    if not data_dir.is_dir():
        raise DataError(
            f"Data folder not found: {data_dir}. Point it at a folder with events.csv and "
            "rsvps/, or run `ilang-pizza demo` to create a demo folder."
        )
    events = _read_events(data_dir / "events.csv")
    known = set(events["event_id"])
    salt = get_or_create_salt(data_dir)

    parts, file_for_event = [], {}
    for path in _table_files(data_dir / "rsvps"):
        where = path.relative_to(data_dir).as_posix()
        rows, fmt = _read_export(path, path.stem, salt, where)
        event_ids = set(rows["event_id"]) | ({path.stem} if fmt != "canonical" else set())
        unknown = sorted(event_ids - known)
        if unknown:
            raise DataError(_unknown_event_message(where, unknown, known))
        for event_id in sorted(event_ids):
            if event_id in file_for_event:
                raise DataError(
                    f"Two files have RSVPs for {event_id!r}: {file_for_event[event_id]} and "
                    f"{where}. Keep one export per event."
                )
            file_for_event[event_id] = where
        parts.append(rows)
    if not parts:
        raise DataError(
            f"No RSVP exports in {data_dir / 'rsvps'}. Put one export per event there, "
            "named after its event_id (e.g. rsvps/pymeets-2026-09-17.csv or .xlsx)."
        )
    rsvps = pd.concat(parts, ignore_index=True)

    attendance_dir = data_dir / "attendance"
    for path in _table_files(attendance_dir):
        where = path.relative_to(data_dir).as_posix()
        if path.stem not in known:
            log.warning("%s Ignoring it.", _unknown_event_message(where, [path.stem], known))
            continue
        rsvps = _apply_checkins(rsvps, path, path.stem, salt, where)

    rsvps = canonical_rsvps(rsvps, events)
    weather_path = data_dir / "weather.csv"
    weather = _read_weather(weather_path) if weather_path.exists() else None
    synthetic = (data_dir / SYNTHETIC_MARKER).exists() or (
        len(rsvps) > 0 and bool((rsvps["source"] == "synthetic").all())
    )
    dataset = Dataset(events=events, rsvps=rsvps, synthetic=synthetic, weather=weather)
    dataset.validate()
    return dataset

"""Loader tests. The sample exports in tests/fixtures/exports use made-up people and
example.com addresses; their headers follow real Meetup, Luma, ti.to and Google Forms files."""

from __future__ import annotations

import datetime as dt
import logging
import shutil
import stat
from pathlib import Path
from textwrap import dedent

import numpy as np
import openpyxl
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ilang_pizza.data.loaders import (
    SALT_FILE,
    DataError,
    _hash_column,
    _parse_status,
    _parse_timestamps,
    detect_format,
    get_or_create_salt,
    hash_identity,
    load_dataset,
    load_rsvp_export,
)
from ilang_pizza.schema import MANILA_TZ, RSVP_COLUMNS, Dataset

FIXTURES = Path(__file__).parent / "fixtures" / "exports"

FAKE_PEOPLE = [
    "Juan Dela Cruz",
    "Maria Santos",
    "Pedro Penduko",
    "Ana Reyes",
    "Jose Bautista",
    "Liza Mendoza",
    "Grace Dizon",
    "Mark Fernandez",
    "Tomas Navarro",
    "Joy Castillo",
    "Leo Pascual",
    "Kim Flores",
    "Bea Villanueva",
    "Miguel Torres",
    "Nina Ramos",
    "Rina Lopez",
    "Paolo Cruz",
    "Carlo Garcia",
    "Walk In",
]
# Other things that identify people in the samples: member ids, a handle, phone numbers,
# profile URLs, IP addresses and answers to event questions.
OTHER_RAW_VALUES = [
    "100000001",
    "mariasantos",
    "+63917",
    "example-python-group",
    "203.0.113",
    "gst-fake",
    "Typing tips",
    "Vegetarian",
    "No pork",
    "Example Corp",
    "Example School",
    "Data analyst",
]


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A writable copy of the sample folder (loading writes a salt file into it)."""
    return Path(shutil.copytree(FIXTURES, tmp_path / "data"))


@pytest.fixture(scope="module")
def sample(tmp_path_factory: pytest.TempPathFactory) -> tuple[Dataset, str]:
    """The sample folder loaded once, with its salt, for tests that only read it."""
    root = Path(shutil.copytree(FIXTURES, tmp_path_factory.mktemp("sample") / "data"))
    return load_dataset(root), get_or_create_salt(root)


@pytest.fixture(scope="module")
def dataset(sample: tuple[Dataset, str]) -> Dataset:
    return sample[0]


@pytest.fixture(scope="module")
def salt(sample: tuple[Dataset, str]) -> str:
    return sample[1]


def by_person(ds: Dataset, event_id: str, salt: str, *identities: str) -> dict[str, pd.Series]:
    """RSVP rows of an event keyed by the raw identity they were hashed from."""
    rows = ds.rsvps_for(event_id).set_index("attendee_id", drop=False)
    return {who: rows.loc[hash_identity(who, salt)] for who in identities}


def meetup(member_id: int) -> str:
    return f"meetup-member:{member_id}"


def write_files(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(text).lstrip(), encoding="utf-8")
    return root


MINIMAL_EVENTS = """
    event_id,start
    pymeets-2026-09-17,2026-09-17 18:30
"""
MINIMAL_FORM = """
    Timestamp,Email Address,Name
    9/1/2026 10:00:00,juan.delacruz@example.com,Juan Dela Cruz
"""


# --- format detection -------------------------------------------------------------------------

MEETUP_CLASSIC = [
    "Name",
    "User ID",
    "Title",
    "Event Host",
    "RSVP",
    "Guests",
    "RSVPed on",
    "Joined Group on",
    "URL of Member Profile",
    "Any dietary requirements?",
]
MEETUP_CONSOLIDATED = [
    "Name",
    "Title",
    "Member ID",
    "Email",
    "Location",
    "Joined Group on",
    "Last visited group on",
    "Last Attended",
    "All time RSVPs",
    "All time 'Yes' RSVPs",
    "All time 'No' RSVPs",
    "Events attended",
    "No shows",
    "Intro",
    "Photo",
    "Role",
    "URL of Member Profile",
    "Event Host",
    "RSVP",
    "Guests",
    "RSVPed on",
    "Q: What do you want to learn?",
]
LUMA_2024 = [
    "api_id",
    "name",
    "email",
    "phone_number",
    "created_at",
    "approval_status",
    "custom_source",
    "has_joined_event",
    "amount",
    "amount_discount",
    "currency",
    "coupon_code",
    "eth_address",
    "solana_address",
    "survey_response_rating",
    "survey_response_feedback",
    "ticket_type_id",
    "ticket_name",
]
LUMA_2026 = [
    "guest_id",
    "name",
    "first_name",
    "last_name",
    "email",
    "phone_number",
    "created_at",
    "approval_status",
    "checked_in_at",
    "utm_source",
    "qr_code_url",
    "amount",
    "amount_tax",
    "amount_discount",
    "currency",
    "coupon_code",
    "eth_address",
    "solana_address",
    "survey_response_rating",
    "survey_response_feedback",
    "ticket_type_id",
    "ticket_name",
]
TITO_TICKETS = [
    "Number",
    "Ticket Created Date",
    "Ticket Last Updated Date",
    "Ticket",
    "Ticket Full Name",
    "Ticket First Name",
    "Ticket Last Name",
    "Ticket Email",
    "Ticket Company Name",
    "Ticket Phone Number",
    "Event",
    "Void Status",
    "Price",
    "Discount Status",
    "Ticket Reference",
    "Tags",
    "Unique Ticket URL",
    "Unique Order URL",
    "Order Reference",
    "Order Name",
    "Order Email",
    "Order Company Name",
    "Order Phone Number",
    "Order Discount Code",
    "Order IP",
    "Order Created Date",
    "Order Completed Date",
    "Payment Reference",
    "Check-ins: Main Entrance",
]


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (MEETUP_CLASSIC, "meetup"),
        (MEETUP_CONSOLIDATED, "meetup"),
        (["  name ", "MEMBER ID", "rsvp", "guests"], "meetup"),
        (LUMA_2024, "luma"),
        (LUMA_2026, "luma"),
        (TITO_TICKETS, "tito"),
        (["Timestamp", "Email Address", "Full Name", "Company"], "google_forms"),
        (["Timestamp", "Name", "Will you bring a laptop?"], "google_forms"),
        (["Email", "Name"], "google_forms"),  # a plain sign-up sheet
        (
            ["event_id", "attendee_id", "rsvp_at", "status", "guests", "attended", "source"],
            "canonical",
        ),
        ([" Event ID", "Email", "RSVP_at", "STATUS"], "canonical"),
    ],
)
def test_detect_format(columns: list[str], expected: str) -> None:
    assert detect_format(columns) == expected


def test_detect_format_rejects_a_meetup_member_list() -> None:
    with pytest.raises(DataError, match="member list"):
        detect_format(["Name", "Member ID", "Joined Group on", "URL of Member Profile"])


def test_detect_format_explains_what_it_expected() -> None:
    with pytest.raises(DataError) as err:
        detect_format(["Ticket", "Seat"])
    message = str(err.value)
    assert "'Ticket', 'Seat'" in message
    for platform in ("Meetup", "Luma", "ti.to", "Google Forms", "canonical"):
        assert platform in message


# --- each sample export -----------------------------------------------------------------------


def test_meetup_consolidated_export_with_attendance_list(dataset: Dataset, salt: str) -> None:
    rows = dataset.rsvps_for("pymeets-2026-08-20")
    people = by_person(
        dataset,
        "pymeets-2026-08-20",
        salt,
        meetup(100000001),
        meetup(100000002),
        meetup(100000003),
        meetup(100000004),
        meetup(100000005),
        meetup(100000007),
    )
    assert len(rows) == 6  # "No" is dropped
    assert hash_identity(meetup(100000006), salt) not in set(rows["attendee_id"])
    # two different members called Juan Dela Cruz stay two people
    assert people[meetup(100000001)]["attendee_id"] != people[meetup(100000005)]["attendee_id"]
    assert people[meetup(100000002)]["guests"] == 1  # "+1"
    assert people[meetup(100000004)]["status"] == "waitlist"
    # the check-in list matches names regardless of case and spacing
    assert people[meetup(100000002)]["attended"] == 1.0
    assert people[meetup(100000003)]["attended"] == 1.0
    # all-time "Events attended"/"No shows" columns are not this event's attendance
    assert people[meetup(100000001)]["attended"] == 0.0
    assert people[meetup(100000007)]["attended"] == 0.0
    assert np.isnan(people[meetup(100000004)]["attended"])  # waitlisted, not checked in
    assert people[meetup(100000003)]["rsvp_at"] == pd.Timestamp("2026-07-28 08:02", tz=MANILA_TZ)


def test_meetup_classic_xls_is_tab_separated_text(dataset: Dataset, salt: str) -> None:
    people = by_person(
        dataset,
        "pyworks-2026-08-29",
        salt,
        meetup(100000001),
        meetup(100000002),
        meetup(100000008),
        meetup(100000009),
    )
    # "user 100000001" and the custom handle "mariasantos" resolve to the same member ids as
    # the newer export, via the profile URL
    assert set(dataset.rsvps_for("pymeets-2026-08-20")["attendee_id"]) >= {
        people[meetup(100000001)]["attendee_id"],
        people[meetup(100000002)]["attendee_id"],
    }
    assert people[meetup(100000002)]["guests"] == 2
    assert people[meetup(100000009)]["status"] == "waitlist"  # "Waiting List"
    assert people[meetup(100000001)]["rsvp_at"] == pd.Timestamp("2026-08-10 19:05:11", tz=MANILA_TZ)
    # no check-in column and no attendance list: unknown, not no-shows
    assert dataset.rsvps_for("pyworks-2026-08-29")["attended"].isna().all()


def test_meetup_xlsx_matches_the_same_export_as_csv(data_dir: Path, tmp_path: Path) -> None:
    csv_path = data_dir / "rsvps" / "pymeets-2026-08-20.csv"
    table = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(list(table.columns))
    for record in table.to_dict("records"):  # Excel stores numbers and dates natively
        record["Member ID"] = int(record["Member ID"])
        record["RSVPed on"] = dt.datetime.strptime(record["RSVPed on"], "%B %d, %Y %H:%M")
        sheet.append([value if value != "" else None for value in record.values()])
    xlsx_path = tmp_path / "pymeets-2026-08-20.xlsx"
    workbook.save(xlsx_path)

    from_csv = load_rsvp_export(csv_path, "pymeets-2026-08-20", salt="pepper")
    from_xlsx = load_rsvp_export(xlsx_path, "pymeets-2026-08-20", salt="pepper")
    assert_frame_equal(from_csv, from_xlsx)


@pytest.mark.parametrize(
    ("name", "separator", "encoding"),
    [
        ("windows.csv", ",", "cp1252"),  # Excel on Windows
        ("unicode-text.tsv", "\t", "utf-16"),  # Excel's "Unicode Text"
        ("european.csv", ";", "utf-8-sig"),  # Excel with a comma decimal separator
    ],
)
def test_reads_other_encodings_and_separators(
    tmp_path: Path, name: str, separator: str, encoding: str
) -> None:
    text = "Timestamp,Email Address,Full Name\n9/1/2026 10:00:00,,Niño Peña\n"
    path = tmp_path / name
    path.write_bytes(text.replace(",", separator).encode(encoding))
    rows = load_rsvp_export(path, "pymeets-2026-09-17", salt="pepper")
    assert rows["attendee_id"].tolist() == [hash_identity("niño peña", "pepper")]
    assert rows["rsvp_at"].tolist() == [pd.Timestamp("2026-09-01 10:00", tz=MANILA_TZ)]


def test_luma_online_export_uses_has_joined_event(dataset: Dataset, salt: str) -> None:
    rows = dataset.rsvps_for("wfh-pythonista-2026-09-05")
    people = by_person(
        dataset,
        "wfh-pythonista-2026-09-05",
        salt,
        "juan.delacruz@example.com",
        "ana.reyes@example.com",
        "tomas.navarro@example.net",
    )
    assert len(rows) == 3  # pending_approval, declined and invited are dropped
    assert people["juan.delacruz@example.com"]["attended"] == 1.0
    assert people["ana.reyes@example.com"]["attended"] == 0.0
    assert people["tomas.navarro@example.net"]["attended"] == 1.0
    # created_at is UTC with milliseconds
    expected = pd.Timestamp("2026-08-21 18:00:00.123", tz=MANILA_TZ)
    assert people["ana.reyes@example.com"]["rsvp_at"] == expected


def test_luma_in_person_export_counts_group_tickets_as_guests(dataset: Dataset, salt: str) -> None:
    rows = dataset.rsvps_for("pymeets-2026-09-17")
    people = by_person(
        dataset,
        "pymeets-2026-09-17",
        salt,
        "juan.delacruz@example.com",
        "maria.santos@example.com",
        "bea.villanueva@example.com",
        "miguel.torres@example.org",
    )
    assert len(rows) == 4  # Juan's two tickets are one party; the declined RSVP is dropped
    assert people["juan.delacruz@example.com"]["guests"] == 1
    assert people["juan.delacruz@example.com"]["attended"] == 1.0
    assert people["maria.santos@example.com"]["attended"] == 0.0
    assert people["miguel.torres@example.org"]["attended"] == 1.0
    assert people["bea.villanueva@example.com"]["status"] == "waitlist"
    assert np.isnan(people["bea.villanueva@example.com"]["attended"])


def test_tito_tickets_export(dataset: Dataset, salt: str) -> None:
    rows = dataset.rsvps_for("pyworks-2026-09-26")
    people = by_person(
        dataset,
        "pyworks-2026-09-26",
        salt,
        "juan.delacruz@example.com",
        "maria.santos@example.com",
        "tomas.navarro@example.net",
        "Paolo Cruz",
    )
    assert len(rows) == 4  # the voided ticket is dropped
    assert set(rows["status"]) == {"going"}
    # the unassigned ticket on Juan's order is his guest; one of them checked in
    assert people["juan.delacruz@example.com"]["guests"] == 1
    assert people["juan.delacruz@example.com"]["attended"] == 1.0
    assert people["tomas.navarro@example.net"]["attended"] == 1.0  # check-in count "1"
    assert people["maria.santos@example.com"]["attended"] == 0.0
    assert people["Paolo Cruz"]["attended"] == 0.0  # no email: identified by name
    stamp = people["juan.delacruz@example.com"]["rsvp_at"]
    assert stamp == pd.Timestamp("2026-09-10 10:15:03", tz=MANILA_TZ)  # "02:15:03 UTC"


def test_google_forms_export_with_attendance_list(dataset: Dataset, salt: str) -> None:
    rows = dataset.rsvps_for("pymeets-2026-10-15")
    people = by_person(
        dataset,
        "pymeets-2026-10-15",
        salt,
        "juan.delacruz@example.com",
        "maria.santos@example.com",
        "rina.lopez@example.com",
        "Paolo Cruz",
    )
    assert len(rows) == 4  # Maria's second submission is a duplicate
    assert set(rows["status"]) == {"going"}
    assert people["maria.santos@example.com"]["rsvp_at"] == pd.Timestamp(
        "2026-10-01 21:40:05", tz=MANILA_TZ
    )
    assert people["juan.delacruz@example.com"]["attended"] == 1.0  # email, any case
    assert people["Paolo Cruz"]["attended"] == 1.0  # name, when neither side has an email
    assert people["maria.santos@example.com"]["attended"] == 0.0
    assert people["rina.lopez@example.com"]["attended"] == 0.0


def test_upcoming_event_with_blank_checkins_is_unknown(dataset: Dataset) -> None:
    rows = dataset.rsvps_for("pykids-2026-11-07")
    assert len(rows) == 3
    assert rows["attended"].isna().all()
    assert dataset.upcoming_event_ids() == ["pyworks-2026-08-29", "pykids-2026-11-07"]


def test_loaded_frames_follow_the_schema(dataset: Dataset) -> None:
    assert list(dataset.rsvps.columns) == RSVP_COLUMNS
    assert str(dataset.rsvps["rsvp_at"].dtype) == "datetime64[us, Asia/Manila]"
    assert dataset.rsvps["guests"].dtype == np.int64
    assert set(dataset.rsvps["source"]) == {"meetup", "luma", "tito", "google_forms"}
    assert not dataset.synthetic
    assert dataset.weather is None
    for _, rows in dataset.rsvps.groupby("event_id"):
        assert rows["rsvp_at"].is_monotonic_increasing  # waitlists read as queues


def test_text_columns_hold_no_placeholder_strings(dataset: Dataset) -> None:
    for frame in (dataset.events, dataset.rsvps):
        for col in frame.columns:
            if frame[col].dtype == object or pd.api.types.is_string_dtype(frame[col]):
                assert frame[col].notna().all(), col
                assert not set(frame[col]) & {"nan", "None", "NaN", "<NA>"}, col


def test_blanks_stay_missing_instead_of_turning_into_text() -> None:
    """On pandas 2, astype(str) writes missing values as "nan"/"None"; none may leak through."""
    keys = _hash_column(pd.Series(["juan.delacruz@example.com", "", "  "]), "pepper")
    assert keys.isna().tolist() == [False, True, True]
    words = pd.Series(["Yes", "No", "declined", "Waiting List"])
    statuses = _parse_status(words, column="RSVP", where="test")
    assert statuses.isna().tolist() == [False, True, True, False]
    stamps = _parse_timestamps(pd.Series(["2026-09-17 18:30", None, np.nan], dtype=object))
    assert stamps.isna().tolist() == [False, True, True]


# --- privacy ----------------------------------------------------------------------------------


def test_no_raw_pii_survives_loading(data_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="ilang_pizza"):
        ds = load_dataset(data_dir)
    text = "\n".join(frame.to_csv() for frame in (ds.events, ds.rsvps)) + caplog.text
    assert "@" not in text
    lowered = text.casefold()
    for raw in FAKE_PEOPLE + OTHER_RAW_VALUES:
        assert raw.casefold() not in lowered, raw
    assert all(not col.startswith("_") for col in ds.rsvps.columns)


def test_canonical_files_hash_ids_that_look_like_emails_or_names(tmp_path: Path) -> None:
    root = write_files(
        tmp_path / "data",
        {
            "events.csv": MINIMAL_EVENTS,
            "rsvps/pymeets-2026-09-17.csv": """
                attendee_id,rsvp_at,status,guests,attended
                juan.delacruz@example.com,2026-09-01 10:00,going,0,1
                Maria Santos,2026-09-02 10:00,going,1,0
                3f2a9c0d1e2b4a5f,2026-09-03 10:00,waitlist,0,
            """,
        },
    )
    ds = load_dataset(root)
    salt = get_or_create_salt(root)
    assert list(ds.rsvps["attendee_id"]) == [
        hash_identity("juan.delacruz@example.com", salt),
        hash_identity("maria santos", salt),
        "3f2a9c0d1e2b4a5f",  # already a hash: kept
    ]
    assert list(ds.rsvps["source"]) == ["canonical"] * 3
    assert ds.rsvps["attended"].tolist()[:2] == [1.0, 0.0]


def test_hash_identity_normalizes_before_hashing() -> None:
    assert hash_identity(" Juan.DelaCruz@Example.COM ", "s") == hash_identity(
        "juan.delacruz@example.com", "s"
    )
    assert hash_identity("Juan  DELA\tcruz", "s") == hash_identity("juan dela cruz", "s")
    digest = hash_identity("juan dela cruz", "s")
    assert len(digest) == 16 and int(digest, 16) >= 0
    assert hash_identity("juan dela cruz", "other salt") != digest
    with pytest.raises(ValueError):
        hash_identity("   ", "s")


def test_salt_is_created_once_and_kept(tmp_path: Path) -> None:
    salt = get_or_create_salt(tmp_path)
    path = tmp_path / SALT_FILE
    assert len(salt) == 32 and int(salt, 16) >= 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert get_or_create_salt(tmp_path) == salt
    assert get_or_create_salt(tmp_path / "elsewhere") != salt


def test_ids_are_stable_per_folder_and_differ_across_folders(
    data_dir: Path, tmp_path: Path
) -> None:
    first = load_dataset(data_dir).rsvps["attendee_id"]
    again = load_dataset(data_dir).rsvps["attendee_id"]
    other = load_dataset(shutil.copytree(FIXTURES, tmp_path / "other")).rsvps["attendee_id"]
    assert first.tolist() == again.tolist()
    assert not set(first) & set(other)


# --- attendance lists -------------------------------------------------------------------------


def test_export_checkins_win_over_an_attendance_list(data_dir: Path) -> None:
    salt = get_or_create_salt(data_dir)
    write_files(
        data_dir, {"attendance/pymeets-2026-09-17.csv": "Email\nmaria.santos@example.com\n"}
    )
    ds = load_dataset(data_dir)
    people = by_person(
        ds, "pymeets-2026-09-17", salt, "juan.delacruz@example.com", "maria.santos@example.com"
    )
    assert people["juan.delacruz@example.com"]["attended"] == 1.0
    assert people["maria.santos@example.com"]["attended"] == 0.0


def test_headerless_attendance_list(data_dir: Path) -> None:
    salt = get_or_create_salt(data_dir)
    write_files(data_dir, {"attendance/pyworks-2026-08-29.csv": "Grace Dizon\nmark fernandez\n"})
    ds = load_dataset(data_dir)
    people = by_person(
        ds, "pyworks-2026-08-29", salt, meetup(100000001), meetup(100000008), meetup(100000009)
    )
    assert people[meetup(100000008)]["attended"] == 1.0  # the first line is a person
    assert people[meetup(100000001)]["attended"] == 0.0
    assert people[meetup(100000009)]["attended"] == 1.0  # waitlisted but checked in


def test_registration_questions_are_not_mistaken_for_checkins(tmp_path: Path) -> None:
    form = """
        Timestamp,Email Address,Attendance,Attended
        9/1/2026 10:00:00,juan.delacruz@example.com,Onsite,/
        9/2/2026 10:00:00,maria.santos@example.com,Online,x
        9/3/2026 10:00:00,ana.reyes@example.com,Onsite,
    """
    root = write_files(
        tmp_path / "data", {"events.csv": MINIMAL_EVENTS, "rsvps/pymeets-2026-09-17.csv": form}
    )
    # an "Attended" column the organizer filled in at the door: "/" present, "x" absent
    assert load_dataset(root).rsvps["attended"].tolist() == [1.0, 0.0, 0.0]
    # without it, the "Attendance" question ("Onsite"/"Online") says nothing about who came
    without = "\n".join(line.rsplit(",", 1)[0] for line in dedent(form).strip().splitlines())
    write_files(root, {"rsvps/pymeets-2026-09-17.csv": without})
    assert load_dataset(root).rsvps["attended"].isna().all()


def test_load_rsvp_export_with_attendance_path(data_dir: Path) -> None:
    rows = load_rsvp_export(
        data_dir / "rsvps" / "pymeets-2026-10-15.csv",
        "pymeets-2026-10-15",
        salt="pepper",
        attendance_path=data_dir / "attendance" / "pymeets-2026-10-15.csv",
    )
    assert list(rows.columns) == RSVP_COLUMNS
    assert sorted(rows["attended"]) == [0.0, 0.0, 1.0, 1.0]


# --- events, timestamps, weather --------------------------------------------------------------


def test_events_are_read_forgivingly(dataset: Dataset) -> None:
    ev = dataset.events.set_index("event_id")
    assert ev.loc["pymeets-2026-08-20", "series"] == "PyMeets"  # "pymeets"
    assert ev.loc["wfh-pythonista-2026-09-05", "series"] == "WFH Pythonista"
    assert ev.loc["pyworks-2026-09-26", "series"] == "PyWorks"  # blank: guessed from the id
    assert ev.loc["pymeets-2026-10-15", "start"] == pd.Timestamp("2026-10-15 18:30", tz=MANILA_TZ)
    assert ev.loc["pyworks-2026-08-29", "start"] == pd.Timestamp("2026-08-29 13:00", tz=MANILA_TZ)
    online = ev.loc["wfh-pythonista-2026-09-05"]  # venue "Zoom", is_online left blank
    assert online["is_online"] and online["venue"] == "" and np.isnan(online["latitude"])
    assert np.isnan(online["capacity"]) and not online["food_provided"]
    assert np.isnan(ev.loc["pyworks-2026-09-26", "capacity"])
    assert (
        not ev.loc["pymeets-2026-09-17", "is_online"]
        and ev.loc["pymeets-2026-09-17", "food_provided"]
    )
    assert ev.loc["pykids-2026-11-07", "food_provided"]  # blank: in-person events have food
    assert list(dataset.events["start"]) == sorted(dataset.events["start"])


def test_events_need_only_event_id_and_start(tmp_path: Path) -> None:
    root = write_files(
        tmp_path / "data",
        {
            "events.csv": """
                Event ID,Start,Series,Online
                pycon-2026-09-17,2026-09-17 09:00,PyCon PH,
                pymeets-2026-09-24,2026-09-24 18:30,,yes
            """,
            "rsvps/pycon-2026-09-17.csv": MINIMAL_FORM,
        },
    )
    ev = load_dataset(root).events.set_index("event_id")
    assert ev.loc["pycon-2026-09-17", "series"] == "Other"
    assert ev.loc["pycon-2026-09-17", "title"] == "pycon-2026-09-17"
    assert not ev.loc["pycon-2026-09-17", "is_online"]
    assert ev.loc["pymeets-2026-09-24", "series"] == "PyMeets"
    assert ev.loc["pymeets-2026-09-24", "is_online"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-09-17 18:30", "2026-09-17 18:30:00"),
        ("2026-09-17T18:30:00+08:00", "2026-09-17 18:30:00"),
        ("2026-09-17T10:30:00Z", "2026-09-17 18:30:00"),
        ("2026-09-17T10:30:00.250Z", "2026-09-17 18:30:00.250"),
        ("2026-09-17 10:30:00 UTC", "2026-09-17 18:30:00"),
        ("9/17/2026 18:30:12", "2026-09-17 18:30:12"),
        ("9/17/2026 6:30:12 PM", "2026-09-17 18:30:12"),
        ("2026/09/17 6:30:12 PM GMT+8", "2026-09-17 18:30:12"),  # not UTC-8
        ("2026/09/17 10:30:12 AM GMT", "2026-09-17 18:30:12"),
        ("September 1, 2025 5:51", "2025-09-01 05:51:00"),
        ("May 16, 2018 8:30 AM", "2018-05-16 08:30:00"),
        ("04/28/2017 17:15:32", "2017-04-28 17:15:32"),
        ("2026-09-17 18:30 PHT", "2026-09-17 18:30:00"),
        ("", None),
        ("soon", None),
    ],
)
def test_parse_timestamps(text: str, expected: str | None) -> None:
    parsed = _parse_timestamps(pd.Series([text], dtype="str")).iloc[0]
    if expected is None:
        assert pd.isna(parsed)
    else:
        assert parsed == pd.Timestamp(expected, tz=MANILA_TZ)
        assert str(parsed.tz) == MANILA_TZ


def test_parse_timestamps_switches_to_day_first_when_a_day_is_above_12() -> None:
    parsed = _parse_timestamps(pd.Series(["17/09/2026 18:30", "01/10/2026 09:00"], dtype="str"))
    assert parsed.tolist() == [
        pd.Timestamp("2026-09-17 18:30", tz=MANILA_TZ),
        pd.Timestamp("2026-10-01 09:00", tz=MANILA_TZ),
    ]


def test_weather_csv(data_dir: Path) -> None:
    write_files(
        data_dir,
        {
            "weather.csv": """
                Date,Station,Precipitation (mm)
                2026-09-17,Port Area,12.5
                2026-09-17,Science Garden,17.5
                9/18/2026,Port Area,T
                2026-09-19,Port Area,-1
                2026-09-20,Port Area,-999
            """
        },
    )
    weather = load_dataset(data_dir).weather
    assert list(weather.columns) == ["date", "precipitation_mm"]
    assert weather["date"].tolist() == [
        dt.date(2026, 9, 17),
        dt.date(2026, 9, 18),
        dt.date(2026, 9, 19),
    ]
    assert weather["precipitation_mm"].tolist() == [15.0, 0.0, 0.0]  # stations averaged; trace


def test_synthetic_flag_comes_from_the_marker(data_dir: Path) -> None:
    (data_dir / "SYNTHETIC").write_text("demo\n")
    assert load_dataset(data_dir).synthetic


# --- friendly errors --------------------------------------------------------------------------


def test_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="Data folder not found.*ilang-pizza demo"):
        load_dataset(tmp_path / "nope")


def test_missing_events_csv(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="No events.csv.*event_id and start"):
        load_dataset(write_files(tmp_path / "data", {"rsvps/x.csv": MINIMAL_FORM}))


def test_events_csv_without_start(tmp_path: Path) -> None:
    root = write_files(tmp_path / "data", {"events.csv": "event_id,date\npymeets-1,2026-09-17\n"})
    with pytest.raises(DataError, match=r"missing required column\(s\): start.*'event_id', 'date'"):
        load_dataset(root)


def test_events_csv_with_unreadable_start(tmp_path: Path) -> None:
    root = write_files(
        tmp_path / "data", {"events.csv": "event_id,start\npymeets-1,next Thursday\n"}
    )
    with pytest.raises(DataError, match="couldn't read the start of pymeets-1 .'next Thursday'."):
        load_dataset(root)


def test_events_csv_with_repeated_event_id(tmp_path: Path) -> None:
    events = "event_id,start\npymeets-1,2026-09-17 18:30\npymeets-1,2026-09-18 18:30\n"
    with pytest.raises(DataError, match="repeated: pymeets-1"):
        load_dataset(write_files(tmp_path / "data", {"events.csv": events}))


def test_rsvp_file_for_an_event_missing_from_events_csv(tmp_path: Path) -> None:
    root = write_files(
        tmp_path / "data",
        {"events.csv": MINIMAL_EVENTS, "rsvps/pymeets-2026-09-71.csv": MINIMAL_FORM},
    )
    with pytest.raises(DataError) as err:
        load_dataset(root)
    message = str(err.value)
    assert "rsvps/pymeets-2026-09-71.csv" in message
    assert "Did you mean 'pymeets-2026-09-17'?" in message


def test_unknown_export_format_names_the_file_and_columns(tmp_path: Path) -> None:
    root = write_files(
        tmp_path / "data",
        {"events.csv": MINIMAL_EVENTS, "rsvps/pymeets-2026-09-17.csv": "Seat,Row\n1,A\n"},
    )
    with pytest.raises(
        DataError, match=r"rsvps/pymeets-2026-09-17.csv: .*Columns seen: 'Seat', 'Row'"
    ):
        load_dataset(root)


def test_unknown_rsvp_status(tmp_path: Path) -> None:
    export = "Name,Member ID,RSVP,RSVPed on\nJuan Dela Cruz,1,Siguro,2026-09-01 10:00\n"
    root = write_files(
        tmp_path / "data", {"events.csv": MINIMAL_EVENTS, "rsvps/pymeets-2026-09-17.csv": export}
    )
    with pytest.raises(DataError, match="unknown RSVP status in column 'RSVP': 'siguro'") as err:
        load_dataset(root)
    assert "Juan" not in str(err.value)


def test_mostly_unreadable_rsvp_times(tmp_path: Path) -> None:
    export = "Timestamp,Name\nlast week,Juan Dela Cruz\nyesterday,Maria Santos\n"
    root = write_files(
        tmp_path / "data", {"events.csv": MINIMAL_EVENTS, "rsvps/pymeets-2026-09-17.csv": export}
    )
    with pytest.raises(DataError, match="couldn't read the times in column 'Timestamp'"):
        load_dataset(root)


def test_binary_xls_asks_for_xlsx_or_csv(tmp_path: Path) -> None:
    root = write_files(tmp_path / "data", {"events.csv": MINIMAL_EVENTS})
    (root / "rsvps").mkdir()
    (root / "rsvps" / "pymeets-2026-09-17.xls").write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    with pytest.raises(DataError, match="old binary Excel file.*save it as .xlsx or .csv"):
        load_dataset(root)


def test_two_files_for_one_event(tmp_path: Path) -> None:
    root = write_files(
        tmp_path / "data",
        {
            "events.csv": MINIMAL_EVENTS,
            "rsvps/pymeets-2026-09-17.csv": MINIMAL_FORM,
            "rsvps/luma/pymeets-2026-09-17.csv": MINIMAL_FORM,
        },
    )
    with pytest.raises(DataError, match="Two files have RSVPs for 'pymeets-2026-09-17'"):
        load_dataset(root)


def test_no_rsvp_exports(tmp_path: Path) -> None:
    root = write_files(tmp_path / "data", {"events.csv": MINIMAL_EVENTS})
    with pytest.raises(DataError, match="No RSVP exports"):
        load_dataset(root)

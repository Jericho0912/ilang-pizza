from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from ilang_pizza.features.calendar_ph import (
    days_since_payday,
    days_to_next_payday,
    holiday_name,
    is_long_weekend,
    is_non_working_day,
    is_payday,
    is_petsa_de_peligro,
    is_ph_holiday,
    is_wet_season,
    next_payday,
    previous_payday,
)
from ilang_pizza.schema import MANILA_TZ


@pytest.mark.parametrize(
    "d",
    [
        date(2025, 1, 15),  # Wed
        date(2025, 1, 31),  # Fri, 31-day month
        date(2025, 4, 30),  # Wed, 30-day month
        date(2025, 2, 28),  # Fri, February in a common year
        date(2024, 2, 29),  # Thu, leap day
    ],
)
def test_weekday_paydays_are_the_15th_and_the_last_day(d):
    assert is_payday(d)
    assert is_payday(d, adjust_weekend=False)
    assert days_to_next_payday(d) == 0
    assert days_since_payday(d) == 0


def test_feb_28_is_not_payday_in_a_leap_year():
    assert not is_payday(date(2024, 2, 28))
    assert days_to_next_payday(date(2024, 2, 28)) == 1


@pytest.mark.parametrize(
    ("nominal", "paid"),
    [
        (date(2025, 3, 15), date(2025, 3, 14)),  # Sat -> Fri
        (date(2025, 6, 15), date(2025, 6, 13)),  # Sun -> Fri
        (date(2025, 5, 31), date(2025, 5, 30)),  # Sat month end -> Fri
        (date(2025, 8, 31), date(2025, 8, 29)),  # Sun month end -> Fri
        (date(2026, 2, 28), date(2026, 2, 27)),  # Sat Feb 28 -> Fri
    ],
)
def test_weekend_paydays_move_to_the_friday_before(nominal, paid):
    assert is_payday(paid)
    assert not is_payday(nominal)
    assert is_payday(nominal, adjust_weekend=False)
    assert not is_payday(paid, adjust_weekend=False)


def test_days_after_an_early_payday_count_to_the_next_month():
    # Nov 30, 2025 is a Sunday, so November's second payday is Friday the 28th. The weekend
    # after it waits for Monday, Dec 15.
    assert next_payday(date(2025, 11, 29)) == date(2025, 12, 15)
    assert days_to_next_payday(date(2025, 11, 29)) == 16
    assert days_since_payday(date(2025, 11, 30)) == 2
    assert days_to_next_payday(date(2025, 11, 29), adjust_weekend=False) == 1


def test_payday_counts_cross_month_and_year_ends():
    assert days_to_next_payday(date(2025, 12, 31)) == 0  # Wed
    assert days_since_payday(date(2026, 1, 1)) == 1
    assert previous_payday(date(2026, 1, 14)) == date(2025, 12, 31)
    assert days_to_next_payday(date(2026, 1, 1)) == 14
    # Feb 2026: the 15th is a Sunday, so the next payday after the 28th is Fri, Mar 13.
    assert days_to_next_payday(date(2026, 2, 28)) == 13
    assert days_to_next_payday(date(2026, 2, 28), adjust_weekend=False) == 0
    assert days_since_payday(date(2024, 3, 1)) == 1  # after the leap day


@pytest.mark.parametrize(
    ("d", "expected"),
    [
        (date(2025, 1, 9), False),  # 6 days before the 15th
        (date(2025, 1, 10), True),  # 5 days before: first day of the window
        (date(2025, 1, 14), True),  # the day before payday
        (date(2025, 1, 15), False),  # payday itself is not peligro
        (date(2025, 1, 16), False),  # just paid
        (date(2025, 1, 26), True),  # 5 days before Jan 31
    ],
)
def test_petsa_de_peligro_window_edges(d, expected):
    assert is_petsa_de_peligro(d) is expected


def test_petsa_de_peligro_follows_the_weekend_adjustment():
    # Mar 15, 2025 is a Saturday: pay comes Friday the 14th, so Saturday is just after payday.
    assert is_petsa_de_peligro(date(2025, 3, 13))
    assert not is_petsa_de_peligro(date(2025, 3, 14))
    assert not is_petsa_de_peligro(date(2025, 3, 15))
    assert is_petsa_de_peligro(date(2025, 3, 14), adjust_weekend=False)


def test_petsa_de_peligro_window_size():
    assert is_petsa_de_peligro(date(2025, 1, 8), window=7)
    assert not is_petsa_de_peligro(date(2025, 1, 14), window=0)
    with pytest.raises(ValueError):
        is_petsa_de_peligro(date(2025, 1, 14), window=-1)


@pytest.mark.parametrize(
    ("d", "name"),
    [
        (date(2025, 6, 12), "Independence Day"),
        (date(2025, 12, 25), "Christmas Day"),
        (date(2025, 12, 30), "Rizal Day"),
        (date(2025, 4, 18), "Good Friday"),
    ],
)
def test_known_ph_holidays(d, name):
    assert is_ph_holiday(d)
    assert holiday_name(d) == name


def test_ordinary_and_special_working_days_are_not_holidays():
    assert not is_ph_holiday(date(2025, 10, 22))
    assert holiday_name(date(2025, 10, 22)) is None
    # The EDSA anniversary was a special *working* day in 2025.
    assert not is_ph_holiday(date(2025, 2, 25))


def test_timestamps_are_accepted():
    evening = pd.Timestamp("2025-04-18 18:30", tz=MANILA_TZ)  # Good Friday
    assert is_ph_holiday(evening)
    assert is_long_weekend(evening)
    assert days_to_next_payday(evening) == 12  # Wed, Apr 30
    assert is_payday(datetime(2025, 1, 15, 9, 0))


@pytest.mark.parametrize(
    ("d", "expected"),
    [
        (date(2025, 4, 16), False),  # Wed before Holy Week
        (date(2025, 4, 17), True),  # Maundy Thursday
        (date(2025, 4, 19), True),  # Black Saturday
        (date(2025, 4, 20), True),  # Easter Sunday ends the run
        (date(2025, 4, 21), False),  # back to work
        (date(2025, 8, 23), True),  # Sat before National Heroes Day (Mon)
        (date(2025, 8, 25), True),  # National Heroes Day
        (date(2025, 8, 26), False),
        (date(2025, 10, 18), False),  # an ordinary Saturday
        (date(2025, 10, 19), False),  # and Sunday: two days is just a weekend
        (date(2025, 1, 29), False),  # midweek holiday (Chinese New Year)
        (date(2025, 12, 31), True),  # Dec 30 to Jan 1, across the year end
        (date(2026, 1, 1), True),
    ],
)
def test_long_weekends(d, expected):
    assert is_long_weekend(d) is expected


def test_non_working_days():
    assert is_non_working_day(date(2025, 10, 18))  # Sat
    assert is_non_working_day(date(2025, 12, 25))  # Thu holiday
    assert not is_non_working_day(date(2025, 10, 22))  # Wed


@pytest.mark.parametrize(
    ("d", "expected"),
    [
        (date(2025, 5, 31), False),
        (date(2025, 6, 1), True),
        (date(2025, 11, 30), True),
        (date(2025, 12, 1), False),
        (date(2026, 1, 15), False),
    ],
)
def test_wet_season_is_june_to_november(d, expected):
    assert is_wet_season(d) is expected

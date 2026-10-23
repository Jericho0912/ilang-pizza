"""Philippine calendar effects on turnout: paydays, holidays, long weekends, rainy season.

Most PH employers pay twice a month, on the 15th and on the last day (*kinsenas at
katapusan*). The broke stretch just before payday is *petsa de peligro*. Holidays come from
the ``holidays`` package: regular holidays and special non-working days count, while special
*working* days do not.

Every function takes a ``datetime.date``. A ``datetime`` or ``pandas.Timestamp`` also works and
is reduced to its own calendar date, so pass Manila-local timestamps.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from functools import cache

import holidays

PAYDAY_MID_MONTH = 15
WET_SEASON_MONTHS = range(6, 12)  # June to November, the PAGASA rainy season

_ONE_DAY = timedelta(days=1)


def _as_date(d: date) -> date:
    # datetime (and pandas.Timestamp) subclass date but refuse to compare with plain dates
    return d.date() if isinstance(d, datetime) else d


def _paid_on(payday: date, adjust_weekend: bool) -> date:
    """When a nominal payday is actually paid: a Saturday or Sunday moves to the Friday before."""
    if adjust_weekend and payday.weekday() >= 5:
        return payday - timedelta(days=payday.weekday() - 4)
    return payday


def _paydays(year: int, month: int, adjust_weekend: bool) -> tuple[date, date]:
    last = calendar.monthrange(year, month)[1]
    return (
        _paid_on(date(year, month, PAYDAY_MID_MONTH), adjust_weekend),
        _paid_on(date(year, month, last), adjust_weekend),
    )


def _shift_month(year: int, month: int, step: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + step
    return index // 12, index % 12 + 1


def next_payday(d: date, *, adjust_weekend: bool = True) -> date:
    """The first payday on or after ``d``."""
    d = _as_date(d)
    mid, end = _paydays(d.year, d.month, adjust_weekend)
    # Next month's mid-month payday falls in next month, so it is always after d.
    following, _ = _paydays(*_shift_month(d.year, d.month, 1), adjust_weekend)
    return next(payday for payday in (mid, end, following) if payday >= d)


def previous_payday(d: date, *, adjust_weekend: bool = True) -> date:
    """The most recent payday on or before ``d``."""
    d = _as_date(d)
    mid, end = _paydays(d.year, d.month, adjust_weekend)
    # Last month's end-of-month payday falls in last month, so it is always before d.
    _, before = _paydays(*_shift_month(d.year, d.month, -1), adjust_weekend)
    return next(payday for payday in (end, mid, before) if payday <= d)


def is_payday(d: date, *, adjust_weekend: bool = True) -> bool:
    """True on the 15th and the last day of the month (or the Friday before, if on a weekend)."""
    return next_payday(d, adjust_weekend=adjust_weekend) == _as_date(d)


def days_to_next_payday(d: date, *, adjust_weekend: bool = True) -> int:
    """Days until the next payday; 0 on payday itself."""
    return (next_payday(d, adjust_weekend=adjust_weekend) - _as_date(d)).days


def days_since_payday(d: date, *, adjust_weekend: bool = True) -> int:
    """Days since the last payday; 0 on payday itself."""
    return (_as_date(d) - previous_payday(d, adjust_weekend=adjust_weekend)).days


def is_petsa_de_peligro(d: date, window: int = 5, *, adjust_weekend: bool = True) -> bool:
    """True from ``window`` days before a payday up to the day before it.

    Payday itself does not count: the money is in. With the default ``window=5`` and a payday
    on the 15th, the 10th to the 14th are *petsa de peligro*.
    """
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")
    return 1 <= days_to_next_payday(d, adjust_weekend=adjust_weekend) <= window


@cache
def _ph_holidays(year: int) -> dict[date, str]:
    # The "public" category holds regular holidays and special non-working days. The package's
    # "workday" category (special working days such as the EDSA anniversary) is left out.
    return dict(holidays.country_holidays("PH", years=year, categories=("public",)))


def holiday_name(d: date) -> str | None:
    """The PH holiday's name (several are joined with "; "), or None on an ordinary day."""
    d = _as_date(d)
    return _ph_holidays(d.year).get(d)


def is_ph_holiday(d: date) -> bool:
    """True on a regular holiday or a special non-working day."""
    return holiday_name(d) is not None


def is_non_working_day(d: date) -> bool:
    """True on Saturdays, Sundays and PH holidays."""
    d = _as_date(d)
    return d.weekday() >= 5 or is_ph_holiday(d)


def is_long_weekend(d: date) -> bool:
    """True when ``d`` is part of a run of 3 or more consecutive non-working days.

    A plain Saturday and Sunday is a run of 2, so it is not a long weekend; add a Friday or
    Monday holiday (or Holy Week) and every day of the run is.
    """
    d = _as_date(d)
    if not is_non_working_day(d):
        return False
    run = 1
    for step in (-_ONE_DAY, _ONE_DAY):
        day = d + step
        while run < 3 and is_non_working_day(day):
            run += 1
            day += step
    return run >= 3


def is_wet_season(d: date) -> bool:
    """True from June to November, the PAGASA rainy season."""
    return _as_date(d).month in WET_SEASON_MONTHS

# Data guide

Ilang pizza? learns from your past events: who RSVP'd, and who actually came. This guide shows how
to export that from Meetup, Luma, ti.to and Google Forms, and how to lay the files out. Everything
stays on your laptop.

- [What you need](#what-you-need)
- [Folder layout](#folder-layout)
- [events.csv](#eventscsv)
- [RSVP exports: Meetup](#meetup), [Luma](#luma), [ti.to](#tito), [Google Forms](#google-forms),
  [anything else](#anything-else-the-canonical-format)
- [Check-in lists](#check-in-lists-attendanceevent_idcsv)
- [weather.csv](#weathercsv-optional)
- [Privacy tips](#privacy-tips)
- [Sources](#sources)

## What you need

- **One row per event** in `events.csv`.
- **One RSVP export per event**, straight from the platform. The format is detected from the
  column names, so you don't need to edit it.
- **Check-ins for past events**, either in the export itself or in a separate check-in list.
  Unknown is not a no-show: an event without check-ins teaches the model nothing, but it doesn't
  teach it anything wrong either. The more past events with check-ins, the better; `doctor` warns
  below 6.

## Folder layout

```text
data/<name>/
  events.csv                  one row per event (required)
  rsvps/<event_id>.csv        one export per event: .csv, .tsv or .xlsx
  attendance/<event_id>.csv   optional check-in list, for exports without check-ins
  weather.csv                 optional daily rain: date, precipitation_mm
  .ilang-pizza-salt           created on first load; never commit or share it
```

- Name each file after its event: `rsvps/pymeets-2026-09-17.csv` belongs to the event whose
  `event_id` is `pymeets-2026-09-17`.
- Old `.xls` workbooks can't be read. Open them in Excel or Google Sheets and save as `.xlsx` or
  `.csv`. From a workbook, only the first sheet is read.
- Check the folder with `ilang-pizza doctor --data data/<name>`. Its error messages name the file
  and column to fix.

## events.csv

Made-up example rows:

```csv
event_id,title,series,start,venue,city,latitude,longitude,is_online,capacity,food_provided
pymeets-2026-09-17,PyMeets September 2026,PyMeets,2026-09-17 18:30,Host office,Taguig,14.5509,121.0503,no,80,yes
wfh-pythonista-2026-09-24,WFH Pythonista September 2026,WFH Pythonista,2026-09-24 20:00,,,,,yes,,no
pymeets-2026-11-19,PyMeets November 2026,PyMeets,2026-11-19 18:30,Host office,Makati,,,no,60,yes
```

| Column | Required | What to put | If blank |
| --- | --- | --- | --- |
| `event_id` | yes | A unique, file-name-safe id. Your export files are named after it. | — |
| `start` | yes | Start time in Manila time, e.g. `2026-09-17 18:30`, or with an offset, e.g. `2026-09-17T18:30:00+08:00` | — |
| `title` | no | Shown in forecasts and emails | the `event_id` |
| `series` | no | `PyMeets`, `PyWorks`, `WFH Pythonista`, `PyKids` or `Other`. Spelling is forgiving (`wfh-pythonista` works). | guessed from the id and title |
| `venue`, `city` | no | Where an in-person event happens | — |
| `latitude`, `longitude` | no | The venue's coordinates, for the rain lookup | central Metro Manila |
| `is_online` | no | `yes` or `no` | `yes` if the venue mentions Zoom, Google Meet, online and so on, or it's a WFH Pythonista with no venue |
| `capacity` | no | How many heads the room fits. `0` or `unlimited` means no limit. | no limit |
| `food_provided` | no | `yes` or `no` | `yes` in person, `no` online |

Yes/no columns accept `yes/no`, `true/false`, `1/0` and `y/n`, and an `x` (a ticked box) counts
as yes. Common alternative headers work too: `name` for the title, `location` for the venue, `lat`
and `lng`, `online`, `limit`, and separate `date` and `time` columns.

## RSVP exports, by platform

### Meetup

1. Open the event on meetup.com as an organizer.
2. Look for **Organizer tools**, then **Manage attendees**, and pick **Attendee details** from the
   download dropdown. That's the path in Meetup's help article on downloading an attendee list.
   Meetup redesigns these menus from time to time, so if yours look different, look for a
   download option on the event's attendee list.
3. Save the spreadsheet as `rsvps/<event_id>.xlsx` (or `.csv`).

The loader reads `Name`, `Member ID` (or `User ID`), `RSVP` (Yes / No / Waitlist), `RSVPed on`
and `Guests`, plus an attendance column (`Attendance`, `Attended` and similar) if your file has
one. "No" rows are dropped, and the member ID keeps two people with the same name apart.

**Taking attendance.** Meetup has check-in built in. Scan members' QR codes in the Meetup app, or
check people in by hand on the web or in the app. Check-in opens 1 hour before the event and stays
open until 24 hours after it ends. After that, you can still update attendance by hand from the
attendee list. Download the list after the event. If your download has no attendance column, put
who came in `attendance/<event_id>.csv` ([below](#check-in-lists-attendanceevent_idcsv)).

Use the organizer download, not a browser extension that scrapes attendee pages.

### Luma

1. Go to the event's **Manage** page, open the **Guests** tab and click **Download CSV** (the
   download icon at the top right). Luma emails you a one-time Confirm Access code first.
2. If you've filtered the list, choose **Download All Guests**.
3. Save it as `rsvps/<event_id>.csv`.

The CSV has each guest's name, email, registration time (`created_at`), `approval_status`, ticket
type and, for in-person events, check-in status. Luma keeps a check-in time per ticket; look for a
`checked_in_at` column. Approved guests count as going and `waitlist` as waitlisted. Declined,
pending and invited rows are dropped. With group registration, Luma writes one row per ticket, and
the loader counts the extra tickets as guests.

**Check-in at the door:** open Luma's scanner (on the web, **Check In Guests** on the manage page;
in the app, **Manage** then **Check In**), or search the guest list and check people in by hand.

For online events, Luma records who clicked the join link instead of a check-in (look for a column
like `has_joined_event`), and the loader counts a join as attending. That's generous: clicking a
link isn't the same as staying for the talk.

### ti.to

1. In the event's admin, open the **Attendees** section, click the blue cog at the top right and
   choose **Export**. Pick CSV (ti.to recommends CSV UTF-8).
2. Save it as `rsvps/<event_id>.csv`.

The loader reads `Ticket Full Name` (or first and last name), `Ticket Email`, `Ticket Created Date`
and the ticket's state, plus a check-in column if the export has one. Voided, cancelled and
refunded tickets are dropped, and an unassigned ticket counts for the person who bought it.

**Check-ins:** use a ti.to check-in list (create it under the event's check-in lists in the admin,
then scan with the ti.to check-in app). After the event, the check-in list's **Actions** menu can
export just the checked-in attendees, with the time each one checked in. Save that file as
`attendance/<event_id>.csv`.

### Google Forms

1. Open the registration form, go to the **Responses** tab, click **More** (⋮) and choose
   **Download responses (.csv)**. Or click **Link to Sheets** and download the sheet as CSV.
2. Save it as `rsvps/<event_id>.csv`.

The loader reads `Timestamp`, `Email Address` (or `Email`), a name column (`Name`, `Full name`,
`Complete name`, `Pangalan`, or first and last name) and an optional guests column (`Guests`,
`Number of guests`, `Plus ones`). Every response counts as going, since forms have no waitlist,
and only a person's first submission is kept. Collect email addresses in the form's settings so
check-ins match reliably.

**Check-ins:** keep a door list (another form, a spreadsheet, or a paper sheet typed up later) and
save it as `attendance/<event_id>.csv`.

### Anything else: the canonical format

If your platform isn't listed, make a CSV with these columns:

```csv
email,rsvp_at,status,guests,attended
juan@example.com,2026-09-01 10:22,going,0,1
maria@example.com,2026-09-02 21:05,going,1,0
pedro@example.com,2026-09-15 08:40,waitlist,0,
```

- `email` (or `name`): who RSVP'd. It's hashed on load.
- `rsvp_at`: when they RSVP'd, in Manila time.
- `status`: `going` or `waitlist`.
- `guests`: extra heads on the same RSVP (optional; default 0).
- `attended`: `1` came, `0` no-show, blank unknown (optional).

## Check-in lists: attendance/\<event_id\>.csv

Use a check-in list when an event's export has no check-ins of its own. If the export already has
them, the list is ignored.

```csv
Email,Name,Attended
juan@example.com,Juan dela Cruz,/
maria@example.com,Maria Santos,x
,Pedro Penduko,/
```

- It needs an email or a name column: `Email`, `Email Address`, `Ticket Email`, `Name`,
  `Full name`, `Pangalan`, or first and last name. A bare one-column list with no header works too.
- With a mark column (`Attended`, `Attendance`, `Present`, `Checked in at`, `Check-in time` and
  similar), only the rows marked present count, so a full guest export works as a check-in list.
  `/`, `yes`, `✓` or a time mean present; `x`, `no` or `absent` mean absent; blank means unmarked.
  Yes, `/` and `x` work like a Philippine attendance sheet.
- People are matched by email first. Names are used when one side has no email; case and spacing
  don't matter, but spelling does.
- Everyone going who isn't on the list counts as a no-show for that event. Waitlisted people on the
  list count as attended. If nobody is marked present, attendance for that event stays unknown.

## weather.csv (optional)

Daily rain for event days, if you have better data than the default lookup:

```csv
date,precipitation_mm
2026-09-17,12.4
2026-09-24,0
```

- The rain column can also be called `rainfall`, `rain_mm` or `precipitation`. PAGASA conventions
  work: `T` or `-1` means trace (counted as 0), and `-999` means missing. Several rows for one day,
  such as one per station, are averaged.
- Days in this file take priority. Other days fall back to Open-Meteo, which receives only the
  venue's coordinates and the date and is cached on your laptop; `--no-weather` turns it off.
  Online events never get rain.
- PAGASA station data is available on request, for example through the eFOI portal
  ([foi.gov.ph](https://www.foi.gov.ph/agencies/pagasa/)), and comes with PAGASA's own terms of
  use.

## Privacy tips

- **Keep `data/` out of git.** The repo's `.gitignore` ignores everything in `data/` except
  `data/README.md`, plus every salt file. Before you commit, `git status` should list nothing under
  `data/`. Exports don't belong in issues, pull requests, screenshots or group chats either.
- **Guard the salt.** `.ilang-pizza-salt` is created on first load, readable only by you. It keeps
  attendee ids stable from one run to the next, which is how attendee history works, and makes the
  ids useless for matching people across folders. Don't commit or share it. If you lose it while
  you still have the raw exports, nothing breaks: the next load makes a new salt and new ids.
- **Delete raw exports once they're hashed.** There's no separate training step to wait for:
  TabPFN learns from your past events every time you run a command, so the folder is the model's
  memory. To keep that memory without keeping raw emails and names, write a hashed copy once,
  check it, then delete the originals:

  ```bash
  mkdir data/pythonph-hashed
  cp data/pythonph/.ilang-pizza-salt data/pythonph-hashed/   # same salt, so new exports still match
  python -c "from pathlib import Path; from ilang_pizza.data.loaders import load_dataset; \
  from ilang_pizza.data.synth import write_dataset; \
  write_dataset(load_dataset('data/pythonph'), Path('data/pythonph-hashed'))"
  ilang-pizza doctor --data data/pythonph-hashed
  ```

  The copy holds only salted hashes, RSVP times, statuses, guest counts and check-ins. New exports
  can go straight into it; repeat the copy into a fresh folder whenever you want to scrub them.
- **Tell attendees.** Say in your registration form that RSVP and check-in records are used, on an
  organizer's laptop, to plan food and seating.
- **Hand over carefully.** When organizer duties change hands, pass the folder (salt included)
  over privately, or delete it.

## Sources

Export steps above come from each platform's help pages, current as of October 2026:

- Meetup: [Downloading an event attendee list](https://help.meetup.com/hc/en-us/articles/39887158477965-How-do-I-download-print-my-event-attendee-list),
  [Managing attendees and tracking attendance](https://help.meetup.com/hc/en-us/articles/9389668230541-Manage-attendees-and-track-attendance-for-your-Meetup-event),
  [The new attendee list and QR code check-in](https://help.meetup.com/hc/en-us/articles/45352992737677-Introducing-the-new-attendee-list-and-QR-code-check-in)
- Luma: [Download Guest List as CSV](https://help.luma.com/p/download-guest-csv),
  [Managing Your Guest List](https://help.luma.com/p/managing-your-guest-list),
  [Check In Guests for In-Person Events](https://help.luma.com/p/check-in)
- ti.to: [Downloading attendee and order data](https://help.tito.io/en/articles/5411873-how-can-i-download-attendee-and-order-data),
  [Check-ins](https://intercom.help/tito/en/articles/2006183-check-ins)
- Google Forms: [View and download responses](https://support.google.com/docs/answer/139706)
- Open-Meteo: [Licence](https://open-meteo.com/en/licence) and [terms](https://open-meteo.com/en/terms)

The exact header names the loader recognizes are listed in `src/ilang_pizza/data/loaders.py`.
If your export isn't recognized, `doctor` prints the columns it saw. Please open an issue with
those column names (never the data).

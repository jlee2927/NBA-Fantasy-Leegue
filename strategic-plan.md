# PuntFit Strategic Build Plan

Working document for Claude Code. Written September 2026, targeting the
2026-27 draft season (drafts peak mid-October).

Claude Code should treat this file as the source of truth for scope and
sequencing. Work one workstream at a time, in the order below, with a commit
and passing tests at each checkpoint. If a change would cross into a later
workstream, stop and say so rather than expanding scope.

---

## 1. What the product does

A draft-day assistant for category-league fantasy basketball.

Core loop:

1. User enters their first-round pick.
2. User chooses how many categories to punt (1, 2, or 3). The app proposes the
   punt builds that fit that player and the user confirms one.
3. Each round, the app re-ranks every available player by punt-adjusted value
   and recommends the best fits, updating as picks come off the board and as
   the user's own roster fills out.

Three ways to run that loop: solo mock draft against simulated managers, live
assistant alongside a draft happening elsewhere (manual pick entry), and a
shared league draft with other real people. The third is the most expensive to
build; see workstream 5 and the open decisions.

---

## 2. Current state

- Valuation engine: z-scores with volume-weighted FG%/FT% impact, iterative
  player pool, punt-fit scoring. Working, 11 tests passing.
- Web app: draft room UI, punt build picker, mock draft against simulated
  managers, live assistant with manual pick entry, projected standings.
  Python plus Flask.
- Data: `fetch_nba.py` pulls hoopR nightly CSV releases and writes
  `data/projections.json`. It is the only module that knows where data comes
  from, and that stays true.
- `puntfit/marcel.py`: Marcel projections from five Basketball Monster season
  exports, with a self-tuning backtest. Written but not yet wired into the
  engine.

Keep the Flask stack. The business plan floated Next.js as an option, but a
rewrite four weeks before draft season buys nothing the user can see.

---

## 3. Target architecture

Five layers, each replaceable without touching the others:

```
data/        fetch_nba.py (NBA), fetch_prospects.py (college/G-League/intl)
projections/ marcel.py (veterans), rookies.py (linear translation)
valuation/   z-scores, punt-fit, scarcity, roster balance
draft/       draft state machine: pool, picks, rosters, recommendations
app/         Flask routes and UI: mock, live assistant, league room, player pages
```

The draft state machine is the piece to get right. Mock draft, live assistant,
and league draft are three drivers of the same state machine, not three
implementations. If simulated managers and real managers do not both just call
`DraftState.make_pick()`, the design is wrong.

---

## 4. Workstreams

### WS1: Wire Marcel into the engine, and move to 9-cat (blocking everything else)

The engine currently values players off last season's stats. Change it to
consume Marcel projections. Turnovers land in the same workstream because they
touch every layer and doing them later means re-backtesting everything.

**Turnovers (9-cat).** Order of operations:

1. Re-export the five Basketball Monster season sheets with TO/g included.
   `marcel.py` already detects a `to/g` column and picks up turnovers
   automatically, including skipping the aging adjustment for them, since
   fewer turnovers is the improvement.
2. Valuation: turnovers are a negative category, so the z-score is inverted
   before summing. Verify the volume-weighted treatment matches how FG%/FT%
   are handled, because turnover value scales with usage the same way.
3. Punt builds: punt-TO becomes a build option and one of the most common
   real ones, since it pairs naturally with high-usage guards. The punt-fit
   scorer needs to suggest it for the right first-round picks.
4. Rerun the Marcel backtest with turnovers included and check the tuned k for
   TO against the other counting stats.

Default the app to 9-cat, since that is the Yahoo default, and keep 8-cat
selectable at league setup.

- Engine reads `data/marcel_projections.json` for the projection set.
- Keep per-game as the valuation basis (H2H category leagues are per-game).
  Marcel writes `*_pg` fields plus `proj_g` and `proj_mpg` for this.
- FG% and FT% come from projected FGM/FGA and FTM/FTA, never from regressed
  percentages.
- `fetch_nba.py` stays the only data-aware module. Marcel reads season sheets
  from `data/seasons/`, writes to `data/`, and nothing else imports xlrd.

Acceptance: the existing 11 tests still pass, the 5 Marcel tests pass, and a
before/after ranking diff is written to `docs/ranking-diff.md` for human
review. Large moves are expected (injured stars regress up, career years
regress down). Unexplainable moves mean a bug.

### WS2: Rookies and players without NBA history

Marcel cannot project a player with no NBA seasons, and every draft has 10 to
20 relevant ones. Without this, the app silently omits or zero-values them,
which is a trust-killer on draft day.

Scope for v1: the top 30 prospects. That covers everyone taken in a standard
12-team draft. Two-way players and deep G-League callups are out.

Approach, deliberately simple:

1. Pull the prospect's last season from college (NCAA), the G League, or an
   overseas league.
2. Convert to per-40-minute rates.
3. Apply a per-stat, per-league translation factor to get per-40 NBA rates.
4. Multiply by expected NBA minutes per game to get per-game projections.
5. Flag the player as `projection_source: "translated"` with a confidence
   level, and surface that flag in the UI.

Two things Claude Code must not do here. Do not invent translation factors.
Derive them empirically: take players from the last 5 to 8 years who played in
one of these leagues and then in the NBA, compare per-40 rates, and fit one
factor per stat per league. Write them to a checked-in config file with the
sample size behind each. And do not silently guess minutes. Expected minutes
is the single largest error source, so it should be an explicit input with a
default from draft slot and depth chart, editable by the user in the UI.

Because the pool is only 30 players, a hand-maintained
`data/prospects_2026.csv` holding each prospect's last pre-NBA season is an
acceptable input and is faster than building a second ingestion pipeline
under deadline. The translation factors still get fitted from historical data,
and `fetch_prospects.py` remains the only module that knows where prospect
numbers came from, whether that is an API or a CSV someone typed.

Known translation direction, to sanity-check the fitted numbers against:
scoring and usage drop substantially from NCAA to NBA, rebounding and blocks
drop moderately, assists and steals travel better, three-point rate often
rises while efficiency falls. G League translates closer to 1.0 than NCAA.

Data sources to verify before building: sportsdataverse publishes NCAA men's
basketball data in the same nightly CSV format as hoopR, and `nba_api` exposes
G League stats through the same endpoints with a different league ID. Confirm
both work from a restricted network before designing around them. If neither
is workable, fall back to a small hand-maintained CSV of the top 30 prospects,
which is roughly 30 minutes of manual work per season and is not shameful.

Acceptance: every player in a standard draft pool has a projection and a
labelled source. Backtest the translation on the 2024 and 2025 rookie classes
and report mean absolute error per category in `docs/rookie-backtest.md`.

### WS3: Player pages and career statistics

Read-only, low risk, and it feeds SEO, which is in the marketing plan.

- Per-player page: career season-by-season table, this year's projection with
  the punt-adjusted value under each of the user's candidate builds, and the
  category z-scores that drive the recommendation.
- Reachable from any recommendation card, because "why is this guy ranked
  here" is the question that decides whether a user trusts the tool.
- hoopR has full history back to 2002, so career tables come from the existing
  pipeline. Precompute to static JSON; no live queries during a draft.

### WS4: Mock draft hardening

Mock drafts are what make the product usable before October, so they carry the
product's credibility.

- Simulated managers need distinct behaviour: some run best-available, some
  run their own punt build, some reach on positional need. Uniform bots make
  the draft feel fake and teach the user nothing.
- Configurable league size (8, 10, 12, 14), roster slots, and category set.
- Post-draft screen: projected category standings, punt build execution grade,
  and the categories the roster actually won.
- Target: a 12-team, 13-round mock completes with no user-visible stall, and
  recommendation recompute stays under 200ms per pick.

### WS5: Leagues (create, join, shared draft room)

This is the workstream that changes the shape of the project. Everything above
runs in one browser session with no accounts, no database, and no server state
worth protecting. Leagues bring accounts, a database, invite links, a draft
clock, autodraft on timeout, reconnect handling, and concurrent-pick conflict
resolution. It is the largest single block of work in this plan and it carries
the only real security and privacy surface in the product.

Recommended shape if it is in scope, in ascending cost order:

- **Tier 1, shared room by link.** No accounts. Commissioner creates a room,
  gets a link, others join with a display name. State in SQLite or Postgres,
  polling every 2 to 3 seconds rather than websockets. Pick timer with
  autodraft. This covers a friends-and-group-chat draft, which is the actual
  use case, and it can ship in days rather than weeks.
- **Tier 2, accounts and persistence.** Email login, saved leagues, draft
  history, multi-league support. Only worth it if Tier 1 shows people running
  real drafts in the room.
- **Tier 3, real-time and sync.** Websockets, Yahoo/ESPN draft import. This is
  the v2 line in the business plan and should stay there.

**Tier 1 is in scope for this season, targeted at October 10**, which is
ahead of the mid-October draft peak but after the October 1 launch of the solo
product. Build it only once WS1 to WS4 are done and committed. Nothing in
Tier 1 may reach into the engine: the league room drives the same
`DraftState` the mock draft drives, and if it needs an engine change, the
design is wrong.

Tier 1 hard limits, so this does not grow:

- No accounts, no passwords, no email. A room link and a display name.
- One room, one draft. No saved history, no rejoining a finished draft.
- Polling, not websockets.
- Pick timer with autodraft to best-available-by-punt-fit on timeout.
- Rooms expire after 7 days. Nothing personal stored beyond a display name,
  which keeps the privacy surface near zero.

If October 1 slips, Tier 1 is still the first thing to cut. A user with the
live assistant already gets the core value in a league draft hosted elsewhere.

---

## 5. Sequencing against the calendar

| Window | Work |
|---|---|
| Now to Oct 1 | WS1 (Marcel plus 9-cat), WS2 (top 30 prospects), WS4. Mobile layout. Ads and affiliate placements. Beta post to r/fantasybball. |
| Oct 1 | Launch the solo product: mock draft and live assistant. |
| Oct 1 to Oct 10 | WS5 Tier 1 league room. WS3 player pages if time allows. |
| Oct 10 to Oct 31 | Peak drafts. Bug fixes only. No feature work, no engine changes. |
| Nov to Dec | Retention and revenue review, v2 go/no-go. |

Freeze the engine one week before launch. A valuation change during draft
season with no time to backtest it is the fastest way to ship bad advice on
the one day it matters.

---

## 6. Standing rules for Claude Code

- Run the full test suite before and after every workstream. Never commit with
  failing tests.
- Any new projection or valuation logic needs a backtest against prior seasons
  with numbers written to `docs/`, not an assertion that it should be better.
- Never hardcode a constant that could be fitted from data. If it must be
  hardcoded, put it in a config file with a comment explaining its source.
- Precompute everything. No network calls during a draft session.
- No NBA or team logos, names, or marks in the UI. Original branding only.
- Data caveats that must stay documented in code: Basketball Monster exports
  carry age as of export date rather than season age, hold only the top 234
  players, and have no turnovers column unless it is added to the export.

---

## 7. Decisions made

1. **Leagues: yes, this season.** WS5 Tier 1 only, link-based room, targeted at
   October 10, built after the October 1 solo launch.
2. **9-cat.** Turnovers go in with WS1. 8-cat stays selectable.
3. **Prospects: top 30 only.** A curated CSV input is acceptable.

Still open:

- **Hosting.** Fly.io suits the current Flask app and gives Tier 1 a place to
  put a small Postgres or SQLite database. Vercel would push toward a rewrite,
  which this calendar cannot absorb. Decide before WS5 starts, not before WS1.
- **Room persistence.** SQLite on a single Fly volume is the simplest thing
  that works for Tier 1 and is a one-line change to Postgres later. Confirm at
  WS5 kickoff.

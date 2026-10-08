# Ramp Polarimeter

Polarization and extinction ratio through an EOM ramp, measured with a
Thorlabs ELL14 rotating the analyzer (LPVIS100-MP2) in front of the PDA10A2,
and the MSO-X 2014A read through [Scope Grab](../scope-grab-multi). The
window follows Scope Grab's layout: controls on the left, plot tabs with a
plot bar on the right, log underneath.

**From VS Code** (the `Python Projects` workspace): pick the
**Ramp Polarimeter** configuration in Run and Debug and press F5, or open
`polarimeter.py` and press the Run button. **Ramp Polarimeter tests** runs the
test suite. Outside VS Code: `python polarimeter.py`, `python -m rampol`, or
`Start Polarimeter.bat`.

Runs on either Python on this PC (Anaconda or the system 3.13): numpy,
matplotlib, pyvisa and pyserial. pyserial sits in the per-user site-packages
both share; if an activated conda env hides that folder, the driver puts it
back on the path itself. Scope Grab is loaded by file path from `scope_grab_path` in
the config (default `Python Projects/scope-grab-multi/scope_grab.py`, at
dcf2c0b or later). Close Scope Grab's panel before connecting: the two cannot
drive the scope at once.

## What one scan measures

The ramp is repetitive, so the scan steps the analyzer through a set of angles
(default 0-350 deg every 10 deg) and records an averaged trace at each. That
gives I(t, theta) - transmission at every time sample and every analyzer
angle - and three measurements come out of it:

1. **Per-sample harmonic fit.** At each time sample,
   `I(theta) = a0 + c2 cos 2theta + s2 sin 2theta` across the angles gives the
   azimuth `psi = atan2(s2, c2)/2` (the analyzer angle of maximum
   transmission), `Imax = a0 + B`, `Imin = a0 - B`, the visibility `B/a0` and
   `ER_fit = Imax/Imin` - along the whole record, with uncertainties from the
   fit residual. With >= 9 angles over >= 300 deg it also fits 1-theta
   (beam walk or wobble as the mount turns) and 4-theta (retardance in the
   analyzer) terms as diagnostics.
2. **Extinction ratio from the dips.** With the analyzer fixed at theta_k the
   ramp sweeps the polarization past the crossed position theta_k + 90 deg at
   some time; near it `I(t) = Imin + Imax sin^2(psi(t) - theta_k - 90)` with
   psi(t) known from (1). Fitting the dip takes Imin from samples next to the
   null rather than as the small difference of two large fit terms. Every
   analyzer angle gives one point on the rising ramp and one on the falling
   ramp: 5 deg steps give an ER every 5 deg of rotation, 36 per ramp.
3. **Null refinement** for the static parts (rest, hold, after the ramp),
   where the polarization does not sweep through any null: a few analyzer
   angles within a few degrees of crossed, at a sensitive V/div, fitted for
   Imin. The same scan folder grows the extra steps.
4. **Direct ER, no fit in the values** (`analysis.direct_er`, also
   `tools/direct_er.py`). At every crossing, Imin is the crossed angle's own
   trace at its minimum (4 us boxcar) and Imax is the trace of the analyzer
   90 deg away at the same instant: both measured. Static stretches use the
   angle nearest crossed; a point where that angle sat so far from crossed
   that the offset alone gives over half the Imin is marked offset-limited
   and left out of the minimum. In the window it follows the Apply switches
   (drift and per-angle transmission corrected); the command-line tool keeps
   the raw traces, as first run (test-4: 16.4 raw, 16.7 corrected - the
   per-angle transmission is the +-2 % between them).

Imax everywhere is the transmission 90 deg from the minimum at the same
instant (measured for direct and bias points, fitted for ER_fit, dips and
refine), never the overall maximum of the record.

**Which ER to believe.** `ER_fit` needs the light level to be the same at
every angle, so slow intensity drift between angles limits it: a 1e-3 gain
error between angles moves Imin by ~1e-3 x Imax, so ER_fit means little above
~1000 unless drift is controlled better than that. The scan therefore returns
to a reference angle every N angles (default 6), corrects the drift from
those returns, and reports how well they predict each other (leave-one-out);
the Extinction tab draws `1 / that scatter` as the limit. The dip and refine
measurements each come from a single angle's captures and do not have this
limit. The analyzer's own ER is 1.4e8 at 843 nm (Thorlabs LPVIS100 data:
1.37e8 at 840 nm, 1.46e8 at 844 nm, 80.6 % transmission); the 1e4 used before
6 Oct 2026 was the sheet's minimum over 550-1500 nm. A config still holding
1e4 is moved to 1.4e8 on load. The table's `ER light` divides it out.

**What a linear analyzer cannot tell.** It measures S0, S1 and S2 only, so
`ER = (1 + p) / (1 - p)` with `p = sqrt(S1^2 + S2^2) / S0` is the same for an
elliptical beam and for a partly depolarized one, and the handedness of any
ellipticity is not measured. The Poincare tab draws |S3| = sqrt(1 - p^2)
and the ellipticity tan chi = sqrt(Imin / Imax) under the assumption of full
polarization. A quarter-wave plate before the analyzer gives S3 with its
sign (not built in yet).

## Using it

The mode tabs are laid out as numbered steps in the order of use (Ramp scan:
1 scope, 2 analyzer angles, 3 shots and dark / background, 4 name, check,
start; AWG: 1 waveform, 2 dry run -> load -> ON, 3 measure in the hold, the
Sequence; Fixed rotations: 1 rotations, 2 at each rotation, 3 preview / dry
run / run). What is set once lives behind each tab's **Settings...**.

**Plan tab.** Before anything moves: *Preview plan* (Ramp scan), the
Sequence's *Preview* (AWG) and *Preview* (Fixed rotations) lay out every step
in the order it will be taken - the analyzer angle, what the AWG puts on X1
and X2, the shots - against the estimated time, and log it as a table
(`rampol/plan.py`). The time is counted in trigger periods, as a single
shot waits for the first trigger after the scope re-arms: a shot is
readout (~0.6 s) + the screen's span rounded up to whole periods, and what
happens between steps (analyzer ~0.75 s, an AWG change ~0.6 s + the settle)
runs in that wait. On 7 Oct this gave 689 s for a sequence that took 690 s
(at a 0.27 s trigger and a 270 ms screen, a shot every 1.08 s, not 0.6 s),
and the spin-echo scans to the second once the trigger period was entered
as the measured 6.26 s (the setting said 5.6). The screen's span is the AWG
tab's (when *set the scope from* is ticked and the AWG drives) or the
preset's timebase. Which tab does what with the AWG: a **Ramp scan** never
touches it (it records whatever plays; its plan says so); the **AWG tab's
Sequence** loads one ramp per (X1, X2) end point and runs a ramp scan for
each at the Ramp scan tab's angles (while it runs, the scan shown is drawn
in full and every other one over it, *compare scans* ticked, each
re-analysed as its steps come in) - the same absolute angles for every
ramp; **Fixed rotations** holds each rotation and steps the analyzer around
the null it finds there (angles counted from that null, marked * in the
table). So "+-2 deg around the null at X1 = 0, 10 ... 40 deg, 16 shots" is
Fixed rotations (rotations `0:40:10`, X1 share 1, null +-2 in 5 points, 16
shots); "the same five absolute analyzer angles at each of five ramps" is the
Sequence.

1. **Connect** (the Hardware pane, one row per instrument: address, Connect,
   what is connected) the scope (VISA address blank = first MSO-X found),
   the ELL14 (COM3 on this PC) and the BK Precision 4063B AWG (blank = first
   4063B found; the row shows its model and the VISA resource it is open
   on). With *Connect on open* ticked (the default) the window connects the
   scope and the ELL14 when it opens, each on its own: one that is off or
   held by another program is logged and the other still connects. The AWG
   is never connected on open (CH1 is often live from the ILC panel, and an
   open session keeps its own GUI out); its Connect, or the first AWG
   operation, connects it. Connecting sends only `*IDN?` - the outputs are
   left as they are. The scope and the AWG share one VISA resource
   manager on the default (NI) VISA that neither closes - EOM-ILC's fix for
   the 26 Aug traps (one close killing the other's session; Keysight's
   ktvisa32 half-loading once NI's is in the process). `Simulate` runs
   the whole window against a software bench instead, which is also what
   the tests use.

   **Copying.** Status lines (position, connections, the corrections line,
   results) are selectable text: drag or double-click, Ctrl+C; right-click
   copies the whole line. Right-click on any plot copies its x and y at the
   mouse (tab-separated, pastes into two spreadsheet cells).
2. **Channels**: give each scope channel a role. One `PD` is required.
   `MonX1`/`MonX2` add the rotation predicted from the Trek monitors (17.55 /
   17.52 deg per monitor volt from the measured V_pi; sign and offset are
   matched to the light). `Ref` (a pick-off before the analyzer, not yet
   installed) would normalise intensity per sample. `Marker`/`Other` are
   recorded but not analysed.
3. **Preset** and **Scope settings...**: picking a preset fills the shot
   settings (mode, shots, points, trigger wait, repetition period); `Apply to
   scope` writes its scope settings AND each recorded channel's V/div and
   offset by role (PD 1 V/div +2.7 V, monitors 1 V/div +2.5 V, commands
   2 V/div +4 V, marker 2 V/div +2 V; DC, displayed), reads every value back
   and drains the scope's error queue - the scope takes a command it does not
   like without a word, so the log says which settings did not land.
   `Spin echo` is the experiment's two-leg sequence, its record set from
   the **sequence fields** shown under the preset: legs `spacing` ms apart
   (the echo time, 16.667 by default), each motion `motion` ms long (9.5),
   and how much to keep `before` the trigger and `after` the second motion
   ends (12 / 12). The timebase is worked out from them (rounded up to two
   figures: 5.1 ms/div from -12 to +39 ms by default; the line under the
   fields says what *Apply to scope* will write), HRES, trigger sweep NORMAL
   (AUTO would self-trigger in a 10 s gap), 8 shots at 10 s. The ILC-target
   comparison takes its leg gap from `spacing` too. (It was `Spin echo 16.7
   ms (2 legs)` before 7 Oct 2026; a config naming it is moved over.) The **Scope settings** window is laid out from the profile's own
   tables (timebase, acquisition, trigger, every channel): Read, edit, Apply
   changes (only edited fields are written), and **Save as preset** keeps them
   with the shot settings under a name. It shows the time span the record
   covers - on the MSO-X a LEFT reference sits one division in from the edge,
   so the record starts at position - 1 div (measured). Set V/div so the PD
   stays on screen at every angle; the log and Diagnostics flag off-screen
   samples.
4. **Scan**: angles (Malus repeats every 180 deg, so 0-170 is a full set;
   0-355 adds the 1- and 4-theta diagnostics), order, mode, shots, dither,
   ref returns, and `rep s`, the trigger period, which only feeds the time
   estimate.
   - `single` (default): one HRES shot per file, read at `Points (single)`,
     averaged here like the ILC. At a 10 s repetition this costs nothing over
     scope averaging and keeps every shot: the scatter gives the error bars,
     and a shot the intensity lock missed is dropped (`lock_tol`, 0.6 % of the
     brightest pre-trigger level; the log says how many).
   - `average`: the scope averages `shots / blocks` triggers per block
     (:DIGitize); an averaged MSO-X record reads out at **7680 points**. Worth
     it only at a fast repetition rate (the AWG ramp at 3.7 Hz).
   - The channel offsets step across `dither codes` ADC codes over the shots
     (or blocks), so the MSO-X per-code error pattern averages out.
   - Every angle is approached from below (`backoff` deg), so backlash always
     lands on the same side. A scan leaves the scope's acquisition type,
     average count and offsets as it found them.
   - Records holding both spin-echo legs are segmented per leg: rest, up 1,
     hold 1, down 1, after 1, up 2, ... after 2.
   **Dark (PD covered)** and **Background (beam blocked)**, each *measure*
   (asks you, takes it, asks you to undo it), *reuse latest* (the newest one
   in another scan at the same PD V/div AND offset - the scope's offset error
   depends on both: -34 mV at +2.65 V on 5 Oct) or *none*. The dark is the
   PD with no light at all; the background is the beam blocked before the
   EOMs with the room as during the scan, so it also holds the stray light.
   The background is subtracted when there is one (it contains the dark),
   else the dark; with both, the corrections line shows the stray light
   apart. Scans before 6 Oct 2026 have one "dark", taken with the beam
   blocked - a background in this sense; the numbers are the same.
   **Scan name**: spaces become underscores (`133 ms SE offset -2V` ->
   `133_ms_SE_offset_-2V`). Windows would take the spaces; Scope Grab's
   Compare box would not - it splits its `KEY:RUNS` entries at whitespace,
   so a scan's key must not contain one. Before 7 Oct 2026 they became
   dashes. A name already used counts up (`test_4` -> `test_5`, `scan` ->
   `scan_2`); a stopped scan of that name is offered for resuming first.
   A scan records the spin-echo sequence fields with its plan when its
   preset is built from them.

   **Stray light at a fine V/div** (on by default, 5 mV/div): when the
   scan measures both a dark and a background, each prompt also reads its
   kind at that V/div (0 V one division below centre). There the scope's
   offset error is the same in both and cancels, and background - dark is
   the stray light to ~0.05 mV - at 1 V/div each is +-1.2-1.5 mV and their
   difference said nothing (a background 0.83 mV below the dark, 7 Oct).
   What is subtracted is then the offset at the scan's V/div (from the dark
   there and from the background there minus the stray light, weighted by
   their errors) plus the stray light; the corrections line shows each
   part. With the background reused, the newest such pair is reused too.
   A dark or background at another V/div gets files of its own
   (`<name>_bg_5mVdiv_001`): before 7 Oct 2026 it took the scan's own
   names and wrote over them (null refine's background did; no bench scan
   had one).

   **Rename / edit...** (next to the Scan box) corrects the shown scan
   after the fact: its name, the preset and spin-echo sequence it actually
   ran (leg spacing, motion, before / after), and notes. A rename moves the
   folder and every file named after the scan (captures, sidecars, the
   manifest, exported figures under `analysis/`), the manifest's file
   lists, the lab-log row, and any other scan citing it (a borrowed dark or
   background). It is all-or-nothing on disk: a file held open (a viewer,
   Excel, OneDrive) refuses it and nothing changes. Every correction goes
   into the manifest's `edits` with the old value, so the record still says
   what was planned. The notes go to the lab log's `notes` column. The
   ILC-target comparison takes its leg spacing from the scan's own sequence
   when it has one. A brief exported before a rename keeps the old name in
   its text: export it again.
   **Check scope** (and `check first`, on by default, before every scan)
   judges the settings - trigger sweep, channels displayed, DC coupling, a
   pre-trigger stretch, trigger wait longer than the repetition - then takes
   ONE shot the way the scan will and judges that: each channel's range
   against its screen and the converter's edge (dither included), screen use
   of the PD, whether there is light, whether the monitors/commands ramp and
   the ramps sit inside the record. One shot is enough for the PD: the ramp
   sweeps the polarization through 180 deg, so at any analyzer angle it
   passes maximum transmission somewhere in the record. A FAIL asks before
   the scan starts; the findings are kept in the manifest (`precheck`).
5. **Set zero from rest**: by the campaign convention analyzer 0 is aligned
   with the polarization at the EO zero. After a scan, this sets the zero
   (the mount angle of analyzer 0) to the scan's fitted rest azimuth. Never
   trust the engraving.
6. **Analyzer** tab, two parts. *Null refine: precise Imin of the shown
   scan where it is flat*. A scan reads every angle at the V/div the
   brightest one needs, so where the rotation stands still its Imin is a
   code or two. This steps the analyzer to `offsets` (deg) around each flat
   stretch's crossed angle - windows `auto` (rest, holds, after) or `t1-t2`
   in ms - at a sensitive PD V/div, and adds the captures to the same scan
   (the Extinction tab's null-refine points). It takes a background at that
   V/div first (block the beam when asked). *Plan (log only)* lists what it
   would measure.
7. *Find the min / max transmission angle*, of either
   - **static light (line trigger)**: the light as it is with nothing
     ramping - the experiment runs on, mains-synchronous, so the scope
     triggers on LINE (2 ms/div from the trigger) and the PD is averaged
     over exactly one line period (`line` Hz), which takes the 60 Hz out;
     the trigger and timebase are put back afterwards; or
   - **record window**: a time in the experiment's own record on its
     trigger (the rest before the ramp, `-10:-0.5`; a hold).

   *Malus scan 0-180* steps the analyzer every `scan step` deg and fits the
   whole curve, weighted by each point's error: the maximum and minimum
   angles, Imax, Imin and the ER. It **autoranges**: each angle is read at
   the most sensitive V/div that holds it (predicted from the points so
   far, one step coarser on a clip, re-read when a much finer one would
   hold it), with the offset putting the floor (no light) three divisions
   below centre so a dark / background can be read at the same setting -
   near crossed that is mV/div instead of one ADC code at 1 V/div. The plot
   shows every point coloured by the V/div it was read at, linear and log.

   **Dark and background for the analyzer** (*Dark* / *Background*: measure
   after, reuse latest, none). The scope's own offset error moves with V/div
   and offset (-34 mV at 1 V/div, 2.65 V offset), so what the PD reads with
   no light has to be known at each setting a reading used. *Measure after*
   does exactly that: once the readings are done it asks you to block the
   beam (background) and / or cover the PD (dark), reads each setting used,
   in the same trigger and timebase, subtracts it reading by reading (the
   background when there is one - it contains the dark and the stray
   light), and stores it in `<outdir>/analyzer_offsets.json`. *Reuse latest*
   takes the newest stored one at the same V/div and an offset within
   max(2 div, 5 %), measured in the same light mode. Find subtracts it from
   the level it reports; the AWG tab's Find does the same.
   *Find and go there* refines one: 4 angles give the azimuth, then the
   analyzer steps +-deg around crossed (at the most sensitive V/div that
   holds it) or aligned, the dip is fitted and the analyzer is left there.
   *Make it analyzer 0* sets the zero so crossed reads 0. Under an
   AWG-held rotation: the AWG tab's Find.

   The scope is set as for a ramp scan (*set the scope from the preset
   first*, on by default): the selected preset is written and read back -
   what *Apply to scope* does - and the scan's pre-run settings check runs
   (a FAIL asks before going on). The acquisition is the scan's: its
   trigger wait, readout points and offset dither, with the Find tab's
   shots (a fixed 10 s wait used to time out on the ~10 s spin-echo
   sequence). With *timebase to the window* (on by default) the timebase
   is zoomed onto the window while measuring - its length plus 10 % either
   side on a 1-2-5 step, the trigger kept on screen for a window before it,
   readout capped at 20k points - and put back afterwards; without it, a
   record window the timebase does not cover is refused before the
   analyzer moves. *Set scope as ramp scan* writes the preset and runs the
   settings check on its own (as *Apply to scope* in the Ramp scan tab).
   Find, the Malus scan, the AWG tab's Find, bias runs and dry runs show
   their step and the time left (at the pace so far) under the Stop
   button; Find and the Malus scan read only the PD channel. Static light then switches to the LINE
   trigger and its own timebase, and puts the preset's back; the AWG tab's
   Find keeps the preset's channels and acquisition but puts its own
   record's timebase on screen, and refuses a LINE trigger.

A scan that stops can be resumed: start a scan with the same name and answer
Yes.

**What is applied, up front.** Under the plot bar: switches for the dark /
background subtraction, the per-angle transmission and dropping missed-lock
shots (with the existing drift correction), and one line saying what each
correction did to the shown scan - what was subtracted and where it came
from, the drift, the per-angle gains, the shots dropped. The per-angle gains are
fitted only from a record whose polarization sweeps >= 20 deg and come out
within 0.8..1.25: with the polarization standing still they trade off
against the Malus terms (7 Oct, an X1 0 deg ramp: -0.83..1.57, Imax 3.8 V
for a measured 5.9 V). A sequence member that does not sweep takes them from
the sibling that sweeps furthest at the same angles (same mount), and the
line says which. The Corrections tab
draws them, and *Borrow dark / background from another scan...* applies an
earlier measurement to the shown scan after the fact (written to its
manifest; *Remove borrowed* takes it out).

**What made a measurement (provenance).** Every scan and bias-run manifest
gets a `provenance` block when it starts: the git commit of
ramp-polarimeter, EOM-ILC and Scope Grab (with `*` where a repository had
uncommitted changes), and the ILC state files the ILC target tab points at -
name, channel, iteration, file time, the target's fingerprint (the fields of
`eomilc.corrections.target_fingerprint`) and a hash of the drive. Those are
the files configured here, not necessarily what the bench plays. Loading a
scan logs the one-line form.

**Lab log.** `<outdir>/lab_log.csv`: one row per ramp scan (key numbers:
angles, shots, V/div, what was subtracted, rotation range, the lowest direct
ER and where, ER_fit at rest and in the holds, drift, residual, versions),
per bias run and per found angle. A scan's row is written when it is loaded
with every Apply switch and the drift correction on, and replaced - not
duplicated - when it is loaded again; a `notes` cell typed by hand is kept.
*Lab log* on the plot bar opens it. A log open in Excel cannot be written;
the window says so and the row goes in next time.

**Export brief** (plot bar) writes the shown scan's standard figure set at
print size into `<scan>/analysis/brief/` - traces, map, residual map,
rotation, extinction, Poincare, corrections, diagnostics - with
`summary.json` (the numbers, every direct ER point, the switches used, the
provenance) and `summary.md` (the numbers, a segment table, each figure
with a factual caption), drawn with the window's current settings.

## Per-angle transmission

The analyzer's throughput to the PD depends on the mount angle: on 5 Oct 2026
it rose 2 % from 0 to 180 deg (mostly a 1-theta term, 1.75 %), i.e. the beam
walking on the detector as the polarizer turns. Left in, it raised the fit
residual from the 1.1 mV shot noise to 12.8 mV and moved the fitted angles by
up to 0.5 deg on the ramps. With >= 8 angles over >= 150 deg the analysis now
fits one transmission factor per angle together with the Malus law (the ramp
sweeping the polarization through 180 deg is what separates the two) and
divides it out; Diagnostics shows the factors. They absorb slow intensity
drift between angles as well.

## Comparing with the ILC target (ILC target tab)

The *ILC target* tab (or the same thing from the command line):

```
python tools/target_compare.py SCAN_FOLDER --x1 ../EOM-ILC/run/drive_P92PX1H.state.npz --x2 ../EOM-ILC/run/drive_P92PX2A.state.npz [--line-ref DRIVE_OFF_SCAN] [--pd-delay-us 3.1]
```

The reference is the target the ILC was given (monitor volts per crystal,
90 deg x V/V90 each), not the rotation the monitors predict. The experiment
plays the ILC drive time-compressed (5 Oct 2026: rise 4.61 -> 4.16 ms, hold
1.24 -> 0.68 ms), so the comparison fits that time map from the recorded
command (CmdX1/CmdX2; the monitors without one) and maps the target the same
way. It writes into `SCAN_FOLDER/analysis/target_compare/`:

- `fig1_time_map` - recorded command against the time-mapped ILC drive
- `fig2_rotation_vs_target` - light and target rotation, light - target and monitors - target
- `fig3_error_structure` - the differences against target rotation; leg 2 - leg 1
- `fig4_correction` - the correction and the per-crystal target change
- `fig5_line_ripple` - the undriven stretches with the 60 Hz family (fitted in the line-reference scan, or here only for show)
- `target_<name>_played.csv` - the ILC target with the experiment's timing: what the ILC should converge to for this sequence (`time_us,voltage_V`, EOM volts, 2 us; loads with `run_ilc.load_target`)
- `corr_<name>_optical.csv` - the target correction for that crystal, in EOM-ILC's correction format (`eomilc/corrections.py`), for the ILC panel's **Corrections** tab
- `summary.json` - time map, gain/offset/delay decomposition per leg, leg 2 - leg 1, line ripple, correction sizes

**The correction is minus (light - monitors)**, low-passed (2 kHz), split
between the crystals (half each by default), zero where the target rests.
Not minus (light - target): the ILC removes the monitors' own error by itself
once it runs on the played target, and a target change built on light -
target removes it a second time (the first version of this tool wrote such
`*_optcorr.csv` files on 5 Oct; it now deletes them). Each file carries the
statistical sigma per sample and the target it was measured against, so the
ILC panel can refuse noise, refuse the wrong campaign and check the Trek
limits.

**Line ripple.** The experiment's trigger is line-synchronous, so 60 Hz and
its harmonics sit at a fixed phase in every record and survive averaging
(about 1 V per crystal, ~25 mdeg on the light). A correction formed from such
a record would carry an anti-ripple tied to this sequence's phase. The ramp
record cannot measure it - its undriven stretches are short and carry the
Trek settling and the crystal memory (on test-4 the 60 Hz estimate moved
2-5x with 0.3 ms changes of the windows) - so give a **line reference**: a
scan of the same sequence with the ramps disabled (8 angles is plenty). It is
fitted over the whole record and subtracted before the correction is formed.
Without one the file says `line_removed: false` and the ILC panel warns.

**PD delay.** The photodiode chain's own delay (an anti-alias RC: 3.1 us) is
measurement, not light. Give it, and the light is read that much later
before it is compared; otherwise the 7 us light-monitor lag seen on test-4 is
partly the filter, and correcting it would put a real error on the light.

## AWG mode (AWG tab)

The 4063B plays a waveform into the Treks on the bench trigger (EXT burst),
CH1 -> X1 -> EO1, CH2 -> X2 -> EO2, through EOM-ILC's upload path and checks
(`rampol/awg.py`). The tab follows the order of use: the AWG's state at the
top with **Park** and **Outputs OFF** (always usable) and a line saying
whether the safety rules are on (red if one is off); **1 Waveform** (a ramp
or two ILC drive files - only the chosen one's fields show - and Preview);
**2** Dry run on scope -> Load to AWG -> Outputs ON; **3** Find min / max in
the hold; and the **Sequence**. What is set once - the dt grid, idle trims,
EOM calibration, bench trigger, the scope's span, dry-run wiring, the
sequence's settle time and the two safety rules - is in **Settings...**.
**Reach.** The 4063B gives +-10 V per channel; this program caps it at
9.6 V. With the 1 Sep 2026 calibration that is ~94 deg on X1 (9.82 deg/V)
and ~99.6 deg on X2 (10.38 deg/V), a little less with the idle trim on top
(`awg.max_deg`): no single crystal reaches 180 deg, and the pair reaches it
only with an X1 share of ~0.45-0.52 (`awg.share_range`). The ramp's fields
say what each crystal gets in deg and volts as they are typed - red, with
the shares that would work, when one is out of reach; the Sequence flags
end points out of reach (a grid leaves them out, a pairs list will not
start); the Fixed rotations tab says how far its split reaches.
The order, each step its own button:

1. **Preview**: the waveform per channel and the rotation it gives (from the
   EOM calibration), and the checks: Trek limits, the 9.6 V cap, the 100 mV
   idle cap, <= 16384 points (the count itself does not matter: in DDS mode
   the channel plays the whole record in 1/FRQ, and FRQ is set to 1/record
   on load and checked there), the record under 80 % of the
   trigger period, a warning for long holds at kV.
   - **ramp**: idle -> the rotation -> idle, split between the crystals
     (`split` on X1), cosine or linear edges, lead / rise / hold / fall /
     after on a `dt` grid; the record is their sum (shown under the fields).
     The defaults make it the ILC's (11 ms at 2 us, 5501 points, 90.893 Hz).
     Another length needs the channels set up for a new FRQ, and setting up
     stops the burst for a moment, so the channel free-runs whatever it
     holds: with the outputs live (never-float) the session first plays a
     flat idle on the current record, then sets up, then uploads. **Park**
     and the end of anything go back to the ILC's 11 ms record at idle, so
     the ILC panel's FRQ check passes afterwards. Not yet done live on the
     bench: the dry run does the same change into the scope first. With the
     Spin echo preset, a record longer than the legs' spacing warns: leg 2's
     trigger would come mid-burst and be ignored. **Idle** blank = the ILC state files' first sample (the learned
     trim, X1 ~+20-26 mV, X2 ~+78-81 mV): the AWG holds the first sample
     between bursts, and file zero parks the EOMs at -9 / -41 V.
   - **ILC drives**: two `run/drive_<stem>_iNN.csv` (AWG volts; a target in
     EOM volts is refused), checked against their own state's target.
   **The scope's span** is set apart from the AWG record (*scope from
   `before` ms before the trigger to `after` ms past the record*,
   rounded up to two figures), on Load, for Find in the hold and at the
   start of a Sequence (around its longest record; unticked, the Sequence
   warns when the screen misses part of a ramp - on 7 Oct the AWG bench
   ramp preset's 1.5 ms/div, set for the ILC's 11 ms, cut the fall off a
   14 ms ramp). The dry run always shows the whole record.

   **With a ramp scan.** The scan does not touch the AWG: Load, dry run,
   Outputs ON, then start a ramp scan in its tab - it records on the same
   bench trigger, the plots refresh after every step, and Park / Outputs
   OFF stay usable. A scan started while this window's AWG plays (ON, not
   parked) records it in its manifest (`drive`: the waveform, its names,
   record, idle, whether its dry run passed), and the lab log's `ilc`
   column says so instead of naming the ILC state files.
   **Sequence** (X1 ends, X2 ends, pairs / grid; *Dry run all*, *Start
   sequence*): ramps with each crystal's own end point (deg; lists as
   `0:90:30` or `0, 45, 90`; *pairs* takes them together - a single value
   goes with every entry of the other - *grid* every X1 with every X2), the
   rest of the ramp from the fields above. *Start sequence* makes one ramp
   scan per ramp, named `<scan name>_X1_<a>_X2_<b>`, at the Ramp scan tab's
   angles, shots, refs and dark / background, and runs them with the AWG
   loaded live between them: *interleaved (per angle)* measures every ramp at
   one analyzer angle before moving on (the analyzer stays put - each step
   says `stayed` - and slow drift is shared alike), *one setting at a time*
   does each ramp's scan whole (fewer AWG changes). A waveform change waits
   `seq_settle_s` (1 s) before the next shot. Each member also measures its
   own hold near crossed (AWG settings, on by default): `seq_null_points` (3)
   angles across +-`seq_null_half_deg` (3) around the hold's crossed angle
   and the bright angle 90 deg from it, after the shared grid, leaving out
   any within 1 deg of the grid. The hold is crossed at `seq_crossed_deg`
   (crossed at rest; 0 is the Find zero, a loaded scan's rest azimuth + 90
   is the measured value and the log says it) + `seq_sense` x (X1 + X2),
   sense -1 as measured on 7 Oct 2026. Why: with a 22.5 deg grid the 15 /
   30 / 60 / 75 deg holds had no angle within 6-9 deg of crossed, so their
   static ER was a useless bound and their hold Imin unmeasured; three
   angles across the null also give Imin(t) and the null's angle through the
   hold. The steps are marked `hold_null` and the angles kept in
   `plan.hold_angles`. Start (and the Plan tab) also says what one scope
   sample means on the fastest ramp (`points` over the screen x the peak
   deg/ms) and warns above 0.3 deg per sample: 20000 points over a 270 ms
   screen were 2.0 deg per sample on 7 Oct (141 deg/ms at the peak of a 1 ms cosine edge to 90 deg), which made the edges and the
   crossing ERs sampling-limited. One dark / background (and
   stray-light pair) is taken in the first scan and lent to the others. Each
   scan's manifest has the ramp it ran (`drive`, with `ends_deg`) and its
   place in the sequence (`plan.series`). Every ramp must have passed a dry
   run (*Dry run all* plays each into the scope); the AWG ends parked on the
   ILC's record. Stopped, it resumes with *Start sequence* under the same
   scan name; a member resumed from the Ramp scan tab would be measured with
   whatever the AWG plays, so that is refused. At the end every scan of the
   sequence is put in the Compare tab.
   The dry run reads the idle level again at 20 mV/div centred on it
   (`awg.IDLE_VDIV`): at the shared setting the screen is centred mid-swing,
   where the scope's offset error is ~-22 mV per volt of offset - on 7 Oct a
   flat 0 V on CH2 read -20, -72 and -121 mV as the centre went 0, 2.3 and
   4.6 V, and the 100 mV idle check failed on the scope, not the AWG.
2. **Dry run on scope**: the AWG's outputs go to two scope channels (`Dry
   run: AWG CH1 -> scope CH3, CH2 -> CH4` by default) - a BNC tee keeps the
   next stage's input driven - while the Treks do NOT drive the EOMs (HV
   disabled or outputs disconnected). Each output is played alone first, the
   other at idle, so the cabling is checked (a swap is caught even when both
   carry the same shape); then both. Each trace is fitted to what was meant
   (`v = gain x u((t - delay) / scale) + offset`) and must pass: gain within
   4 % (the scope's own accuracy is 3 %), time scale within 5e-4 (a record
   played at the wrong FRQ shows here), delay under 20 us, shape within 1.5 %
   rms of the swing, shots within 5 us of each other (triggered, not
   free-running), idle within 100 mV of the meant level (the generator's own
   zero-code error, -12 / -40 mV, is expected and shown). The scope's V/div,
   offset and timebase are put back; its trigger is left alone (the bench
   trigger). A pass is remembered by the waveform's name (the hash of its
   samples) for the session; the record goes to `<outdir>/awg_dryrun/`
   (`.json` + the traces in `.npz`) and the lab log. The AWG plot tab then
   shows what the scope saw against what was meant, and the residual.
3. Reconnect the Treks. **Load to AWG** / **Outputs ON** (asks first). With
   **require a dry run** ticked (the default), nothing that has not passed a
   dry run this session is loaded onto live outputs or switched on.
4. **Find min / max** in the hold (after `settle` ms: scope overdrive
   recovery; the Trek's last 0.1 % takes 10-20 ms), or a ramp scan with the
   waveform playing (Ramp scan tab).

**Never float** (ticked by default, BOTH outputs - which output reaches the
X2 path's FPGA/buffer stage, whose output goes high (-4 to -5.7 kV) on a
floating input, depends on the cabling): the program never switches an output
off. Waveform changes are made live (as EOM-ILC's uploads are), and the end of
anything - closing the window, Disconnect, a dry run, a bias run - is
**Park**: an idle-level waveform with the outputs ON. **Outputs OFF** then
asks first. Unticking the rule asks too, and is only for when that stage is
bypassed on both channels. Park and OFF work while a measurement runs, and
are carried out in the order pressed. After the AWG has been used here it
plays idle (parked): the experiment's own ramps come back when its drive is
put back (the ILC panel uploads it).

A waveform's name is a hash of its samples (`RP<ch><8 hex>`), so the same
one is selected again rather than stored again (the 4063B cannot delete over
SCPI). A channel found ON that this window did not switch on is taken over,
left ON, when it plays one of this program's waveforms (the window was closed
- it parks on close - and opened again; `Connect` does it at once). One
playing another program's waveform (the ILC panel's) is refused; **Park**
takes it over without switching it off, for when nothing else drives it. With
what the AWG holds unknown, a record-length change plays the new waveform's
flat idle first - a flat record plays the same at any length. Outputs are switched one at a time and read back; if either fails,
nothing is left on.

## EOM calibration (EOM calibration... button)

One chain per crystal, every number visible and editable (`rampol/calib.py`):
`monitor V = gain x (AWG V - idle)`, `kV = monitor V / (monitor V per kV)`,
`deg = 90 x kV / V90`; the pair turns the light by the sum of the two. The
AWG waveforms and bias runs (rotation -> AWG volts), the scans' rotation from
the monitors and the ILC-target comparison all use it. Defaults: the 1 Sep
2026 optical calibration (gain 0.5594 / 0.5924, V90 5.1283 / 5.1374 kV).
*From EOM-ILC* reads `eomilc.config` as it is now; *Fit to the loaded bias
run* takes each crystal's gain from the points' monitor against AWG volts
and scales both V90s by the static transfer curve's light / monitors gain
(the crystals are driven together there, so their V90s are not separable
from one run). The converter turns any of AWG V, monitor V, kV or degrees
into the others. *Apply and save* stores it in the config with its source
and date; every dry-run record carries the calibration it was made with.

## Fixed rotations (Fixed rotations tab; `bias points` in the code and files)

A ramp scan reads every angle at the V/div the brightest needs, so near a
null the scope resolves ~1 code: on test-4 the rest/hold minima (2.6-3 mV)
were one step at 1 V/div - ER ~1800 there is the scope's floor. Mid-ramp
(166-281 mV, ER 18-31) they were well resolved.

*Preview* draws every plateau of the plan on the AWG plot tab - both AWG
outputs and the rotation each gives, the window it measures in shaded, which
have passed a dry run - and logs their peaks; the AWG tab's Sequence has
the same button.

Fixed rotations ("bias points") hold the EOMs at each rotation in a list with the AWG (4063B; close its
GUI; CH1 -> X1, CH2 -> X2; plateaus on the bench trigger, EXT, each checked
with EOM-ILC's limit check before upload). They use the AWG tab's session,
idle levels and its two rules: with 'require a dry run', *Dry run on scope*
in this tab first plays every distinct plateau of the plan into the scope
(wiring checked on the first one that moves), and Start refuses a plan with
any plateau that has not passed; under never-float the bias changes are
made live and the run ends parked. The plateaus are the ILC's record length
(5501 points), so the ILC's FRQ check passes afterwards. Per bias:

1. 4 analyzer angles at the normal V/div: the azimuth and Imax;
2. the analyzer stepped +-`null` deg around the crossed position at the most
   sensitive V/div that keeps the scan on screen (the ladder goes up a step
   when a reading clips): I = Imin + K sin^2(theta - theta_n) -> theta_n,
   Imin;
3. the bright angle at the normal V/div: Imax.

ER = Imax / Imin, both measured and dark-subtracted at the V/div each was
taken at (the beam is blocked once, at the start, for every V/div the run
can use). An unresolved Imin gives a lower bound. K / (Imax - Imin) checks
the Malus shape. The window starts `settle` ms into the hold: the plateau's
edges overdrive the scope at mV/div and the MSO-X needs ~3 ms to recover.

theta_n - 90 is the static polarization azimuth, so the run is also the
rotator's **static transfer curve**: light against the monitors' prediction,
over 0-180 deg. The tab draws it with the shown scan's ramp light - monitors
(from its ILC comparison) on the same axes: if the ramps' +-2-3 deg, ~90 deg
pattern is also there statically, it is the optics (QWP alignment or
retardance), not the dynamics. `updown` order repeats the list downwards
(hysteresis). Results: `<outdir>/<name>/bias.json` + `point_NN.npz` (mean
traces, decimated). The beam itself is not measured for depolarization vs
ellipticity - that needs a QWP in front of the analyzer (a second scan gives
S3).

## Files

Each scan is a folder `<outdir>/<name>/` (default outdir
`Python Projects/scope_data/polarimetry`):

| file | what |
|---|---|
| `<name>_scan.json` | manifest: plan, channel roles, scope and mount identity, the analyzer zero, provenance (software versions, ILC state), every step with target, landed angle, clock, V/div and offset, file names; rewritten after every step |
| `<name>_a045.00_001.npz` + `.txt` | one dither block at analyzer 45.00 deg - Scope Grab's NPZ and sidecar, written by its own `write_capture` |
| `<name>_ref003_001...` | reference-angle returns |
| `<name>_dark_001...` | beam blocked |
| `<name>_n2_a137.25_001...` | null refine, window 2 |
| `analysis/brief/` | Export brief: figures, summary.json, summary.md |
| `analysis/direct_er/` | `tools/direct_er.py` output |
| `<outdir>/awg_dryrun/<time>_<label>.json/.npz` | every AWG dry run: the waveform(s), the wiring, each step's fit and verdict, the calibration in use; the scope traces |

Next to the scan folders: `lab_log.csv`, one row per scan, bias run and found
angle.

The capture names split as Scope Grab expects (`prefix_NNN`), so its Compare
box opens any angle: key `<name>_a045.00`, runs `1-4`.

## Tabs

Every figure's **Save** button opens on `<the shown scan's folder>/saved_figures/`
(the output folder for the tabs that are not about a scan), the file named
for the scan and the tab - `<scan>_Extinction_vs_rotation.png`,
`<scan>_Malus_t12.000ms.png`, `compare_<scans>.png` - with _2, _3 ... when
it exists; PNG at 150 dpi, or PDF / SVG.

With no scan loaded, the Compare tab draws the compared scans (the
difference from the first of them), and so do the tabs with a *compare
scans* box: the first compared scan stands in for the shown one.

Two switches recur. **log y** (Traces, Malus, Extinction, Shots, Compare,
Fixed rotations): an extinction ratio goes log; a light level, which can go
below zero once the dark is subtracted, goes symmetric-log (linear within
+-1 mV). **compare scans** (Extinction, Malus, Angle, Poincaré,
Diagnostics): the scans picked in the Compare tab (*Compare selected*) are
drawn there too, one colour each - green, red, purple, ... - while the shown
scan keeps its own colours. The Malus tab sets them against the shown scan
at the same TIME, which means the same thing only for the same sequence.

| tab | shows |
|---|---|
| Traces | PD at every analyzer angle (colour = angle), dark dashed; monitors below |
| Map | I(t, theta) / Imax(t), with the fitted null psi + 90 drawn over it; or the Malus-fit residual (mV, or per standard error) per angle and time, with the rms per angle beside it: a bad angle, clipping, a missed lock or drift shows as a row or a patch |
| Malus | I vs analyzer angle at the cursor time, the fit, residuals |
| Angle | rotation from rest with +-1 SD, the monitor prediction (zero at rest before the first motion, like the light), their difference in mdeg - so it starts at zero and shows where they part; with *compare scans*, each compared scan's rotation and its own light - monitors (a hold-time series side by side). Before 7 Oct 2026 the offset was the whole record's mean difference, which long holds pulled 0.35-0.8 deg off at rest |
| Extinction | the extinction ratio along the record by method (marker): measured at a crossing (circles), dip fit (triangles), measured static (squares; grey hollow where the angle was too far from crossed), null refine (diamonds), the per-sample Malus fit as a grey line with its drift limit; colour = leg (first / second transport, shaded on the time axis), filled = on a ramp out from rest, hollow = on the ramp back, half-filled = standing still (rest, hold, after) - taken from the point's segment; +-1 sigma bars (asymmetric: ER goes as 1/Imin); a lower bound is its marker with a dotted line going up (not an arrow: its head read as the dip fit's triangle) (the true ER is above it; a method with smaller noise gives a higher bound for the same unresolved Imin, so dip-fit bounds sit above the measured ones), hidden by unticking *lower bounds*. *What are these?* opens the explanation of every family. x = time or rotation (rotation lines the two legs up). **Export CSV... / Copy CSV**: every value as a table (`analysis.er_csv`) - method, leg, direction, time, rotation, analyzer angle, ER, +-1 sigma, lower-bound flag, Imin, Imax, and the per-sample fit's ER in 2-deg rotation bins - under a '#' header saying what the scan was, what was subtracted, the crossings' Imin spread across analyzer angles and the ER it limits to, and the drift limit; `pandas.read_csv(path, comment='#')` reads it |
| Poincaré | the linear Stokes parameters in the rest frame on the sphere (coloured by time), |S3| and the ellipticity angle chi vs time assuming full polarization (handedness not measured), the ellipse at the cursor against the rest ellipse |
| Diagnostics | ref returns vs time, per-angle transmission (or 1-theta and 4-theta amplitudes with it off), residual vs block SEM, landing error and off-screen samples per step; a line under each panel says what it shows and what it should look like |
| Table | per-segment medians, every refine, dip and direct ER; Save CSV |
| Shots | the data behind every number: pick steps (several with ctrl/shift), a channel, and any of single shots straight from the files, the average the fit uses, +-1 SE, the min-max over the shots, the dropped (missed-lock) shots dashed, the analyzer 90 deg away; a shot list (`1, 3-5`) and a time window - zoom with the toolbar and the view re-reads the files at full resolution. *Crossed at cursor* picks the angle nearest crossed at the cursor time with its partner: the direct-ER view |
| Build | how the angles become the polarization: a time slider; top every angle's averaged trace as fitted, bottom left the points at that instant (and before corrections) with the Malus fit a0 + B cos 2(theta - psi), its maximum and null, bottom right the rotation with the instant marked |
| Corrections | what is subtracted (dark / background traces and levels, borrowed ones dashed), the reference drift, the per-angle transmission, shots kept and dropped per step |
| Compare | the shown scan against up to 6 others picked in a list: rotation, the difference from the shown scan (smoothed, with the shown scan's +-1 SD), ER_fit and the direct points; each with the window's Apply switches |
| Find angle | the last Find angle scan and its fit |
| AWG | the AWG tab's waveform per channel, the rotation it gives, the hold and the Find window; after a dry run, what the scope saw against what was meant and the residual |
| Fixed rotations | static ER vs rotation (Imax/Imin and from the null curvature), light - monitors static (and the shown scan's ramp), Imin with the V/div it was read at, the last null scan |
| ILC target | the ILC comparison's figures for the shown scan |

Click a time on Map, Angle, Extinction or the Poincaré tab's time plot to move
the cursor (Malus, Build and Poincaré follow it). A cursor move updates the
tabs in place - the lines, the Malus points, the ellipse - without rebuilding
or re-laying-out the figure (0.1-0.2 s instead of 0.3-0.6 s on test-4); the
Map draws block means over ~4000 columns (a 19 x 100k mesh took 2.4 s).

## Traps this is built around

- **The mount's zero is not the optical zero.** The 1 Sep campaign found
  nominal 0 at +1.95 deg. Fit it (`Set zero from rest`).
- **Sample 0 of every MSO-X record carries a ~1.2 V artefact**: the first
  `trim` (10) samples are dropped.
- **The MSO-X converter has a fixed error pattern per 40.25 mV code** at
  1 V/div that averaging keeps: hence the offset dither, on by default.
- **Overdrive recovery.** At a sensitive V/div the bright part of the trace
  is far off screen, and the front end then reads a few % of the level for
  ~1 ms after it comes back. A refine window right after a bright excursion
  (the hold, after the mid-ramp) can carry this; the rest window before the
  ramp cannot. Compare the two.
- **Intensity drift reads as polarization** in any fit across angles. The ref
  returns measure it; a pick-off reference PD would remove it per sample.
- **In a hold the light does not quite follow Malus** (7 Oct 2026, XEO1
  series: fit residual 4-6 mV against 0.9 mV at rest). It is an additive
  per-analyzer-angle pattern of up to 0.3 % of Imax, the same for every held
  voltage at a given angle and not symmetric under analyzer + 180 deg, so
  not a polarization state - the beam on the detector changes while the
  field is on. It biases each scan's hold azimuth by up to ~0.1 deg and
  raises the hold Imin by its size. The load log flags it
  (`analysis.malus_check`, also in the lab-log summary).
- **A static stretch is named by the motion before it** (`segments`): after
  an up it is the hold, after a down the after-ramp rest unless the ramp came
  only part of the way back. Until 7 Oct 2026 a hold had to be past 45 deg,
  so the 15 and 30 deg holds were called "after".
- **A static ER needs an angle at crossed.** The nearest angle sits `off`
  deg from it, and Imax sin^2(off) of what it reads is that offset: once it
  is above the noise the point is a lower bound (`bound_from` offset), and
  over half the reading it is useless (`offset_limited`). Each such point
  names the angle that would have been crossed (`crossed_deg`); the sequence
  measures them by itself (hold-null angles above).
- **A borrowed dark / background goes stale**: ~2 mV over a day at 1 V/div,
  which is the whole rest ER floor. The corrections line and the load log say
  its age past 2 h (`analysis.offset_age_h`).
- **Inter-channel skew** of ~97 ns between the CH1/CH2 and CH3/CH4 pairs:
  0.015 deg at the fastest ramp rate - negligible here, noted for anything
  finer.

## Tests

```
python tests/run_tests.py
```

The suites run side by side (~3.5 min, as long as `test_gui` alone; one
after another they took ~5). `run_tests.py gui awg` runs only those,
`-q` prints only the failures and the times, `--serial` runs them in turn.

| suite | covers |
|---|---|
| `test_ell14.py` | the driver against a fake serial port (the real mount's IN reply), the approach-from-below wrapper |
| `test_analysis.py` | the harmonic fit exact on noise-free data, its uncertainties checked by pulls (unit spread), lower bounds; a 72-angle simulated scan written and read back: rotation, rest azimuth, drift correction, 142 dip ERs against the model, direct ERs against the model and their Imax against the fit, the residual map at unit noise, the Stokes identities, segments, monitor prediction; provenance (this repository's commit, an ILC state's fingerprint) and the lab log's update-in-place |
| `test_checks.py` | the pre-run check against the simulator: a hand-changed scope put back by a preset and confirmed, a silently refused setting reported, and each failure it should catch (AUTO sweep, channel off, wait <= repetition, AC coupling, clipping, off screen, small signal, ramp cut off, no light, no pre-trigger) |
| `test_awg.py` | ramps (the ILC's record, ends at idle, the hold at the rotation), every check (length, trigger period, duty, idle and AWG caps), ILC drive files (header, a target refused, keepers against their targets pass and as u x gain fail), the session against the simulated AWG (names reused, a foreign ON refused, OFF for a change, park, a live FRQ change refused, CH2 refusing ON leaves nothing on); the rules (ON and live loads refused without a dry run, OFF needs force, the end is park); the dry run into the simulated scope (passes and sees the zero-code error, puts the scope back; catches swapped cables and a wrong FRQ); the calibration (conversions, applied everywhere, a bias-run fit) |
| `test_bias.py` | the plan (AWG volts, plateaus, the Trek limit check), the null fit and ER, and a whole bias run on the simulated bench (AWG plateaus into the bench model): ER at 0-90 deg against the model, a 2 deg static rotator error recovered, outputs off and scope restored at the end |
| `test_gui.py` | the window against the simulator: connect, dark, scan, every tab drawn, cursor, null refine of rest/hold/after against the model ER, a fixed-rotation run from its tab, rename / edit (files, manifest, lab log, citing scans, a held file refused), the direct points, residual map, Poincaré, Compare, Export brief, provenance and lab-log rows; config sandboxed, window off screen |

## What has and has not run on hardware

- **Verified, 5 Oct 2026:** the ELL14 (S/N 11400318, firmware 13, COM3)
  answered `in`, `gs` and `gp` through this driver: 143360 pulses/rev, status
  OK.
- **Run on hardware 5 Oct 2026:** ELL14 motion and full ramp scans
  (16-ms-spin-echo-test-1..5).
- **Not yet run on hardware:** the AWG mode and bias points (AWG control from this program),
  the null refine at a sensitive V/div, and an optical correction applied
  through the ILC. These are tested against the simulator only. A first bias
  session: a short list (`0, 90`), 4 shots, watching the first plateau on
  the scope before the full 0-180.

## Provenance

`rampol/ell14.py` is the reviewed driver from EOM-ILC's
`eomilc_polarization_finetune` (a556aff); this repository now holds the
canonical copy.

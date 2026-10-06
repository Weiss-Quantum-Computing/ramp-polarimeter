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

1. **Connect** the scope (VISA address blank = first MSO-X found) and the
   ELL14 (COM3 on this PC). With *Connect on open* ticked (the default) the
   window connects both when it opens, each on its own: one that is off or
   held by another program is logged and the other still connects. The AWG
   is never connected on open (CH1 is often live from the ILC panel); it
   connects on first use. The scope and the AWG share one VISA resource
   manager on the default (NI) VISA that neither closes - EOM-ILC's fix for
   the 26 Aug traps (one close killing the other's session; Keysight's
   ktvisa32 half-loading once NI's is in the process). `Simulate both` runs
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
   `Spin echo 16.7 ms (2 legs)` is 5 ms/div from -12 to +38 ms (12 ms of
   locked light before the trigger, both legs, ~12 ms after leg 2, where the
   intensity lock switches off), HRES, trigger sweep NORMAL (AUTO would
   self-trigger in a 10 s gap), 8 shots at 10 s. The **Scope settings** window is laid out from the profile's own
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
   **Scan name**: a name already used counts up (`test-4` -> `test-5`,
   `scan` -> `scan-2`); a stopped scan of that name is offered for resuming
   first.
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
6. **Analyzer** tab, two parts. *Refine the shown scan's static nulls*:
   windows `auto` (rest, hold, after) or `t1-t2` in ms,
   offsets around crossed, the PD V/div at the null. It takes a background
   at that V/div first (block the beam when asked).
7. *Find the min / max transmission angle*: the analyzer angle of minimum (crossed) or maximum
   transmission for the light as it is in a window of the record (the rest
   before the ramp, `-10:-0.5`; a hold of the sequence), or with the AWG
   holding a bias. 4 angles give the azimuth, then the analyzer steps
   +-deg around crossed (at the most sensitive V/div that holds it) or
   aligned, the dip is fitted and the analyzer is left there. *Make it
   analyzer 0* sets the zero so crossed reads 0.

A scan that stops can be resumed: start a scan with the same name and answer
Yes.

**What is applied, up front.** Under the plot bar: switches for the dark /
background subtraction, the per-angle transmission and dropping missed-lock
shots (with the existing drift correction), and one line saying what each
correction did to the shown scan - what was subtracted and where it came
from, the drift, the per-angle gains, the shots dropped. The Corrections tab
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
(`rampol/awg.py`).

- **ramp**: idle -> the rotation -> idle, split between the crystals
  (`split` on X1), cosine or linear edges, lead / rise / hold / fall in a
  `record`-ms record on a `dt` grid. The default record is the ILC's (11 ms
  at 2 us, 5501 points, 90.893 Hz), so switching between an ILC drive and a
  ramp never needs the channel set up again. **Idle** blank = the ILC state
  files' first sample (the learned trim, X1 ~+20-26 mV, X2 ~+78-81 mV): the
  AWG holds the first sample between bursts, and file zero parks the EOMs at
  -9 / -41 V. Both ends are exactly idle.
- **ILC drives**: two `run/drive_<stem>_iNN.csv` files (AWG volts; a target
  file in EOM volts is refused), checked against their own state's target as
  the ILC does - a keeper checked as u x gain fails the 2 mA current limit.
- **Preview** draws it with the rotation the monitors' model gives and runs
  the checks: Trek limits, the 9.6 V cap, the 100 mV idle cap, <= 16384
  points (5501 proven), the record under 80 % of the trigger period, and a
  warning for long holds at kV (duty).
- **Load to AWG** sets a channel up only where it is not already right for
  the record length (FRQ = 1/(N dt), 20 Vpp, DDS, burst on EXT - a setting
  that does not take stops it), then puts the waveform on: a waveform's name
  is a hash of its samples, so the same one is selected again rather than
  stored again (the 4063B cannot delete over SCPI). The outputs go OFF for a
  change.
- **Outputs ON** asks first; both are switched one at a time and read back,
  and if either fails both go off. **Outputs OFF** works at any time, even
  while a measurement runs. A channel that is ON but was not switched on by
  this window (the ILC panel) is refused.
- **Find min / max** sweeps the analyzer in the hold, `settle` ms after it
  starts (scope overdrive recovery; the Trek's last 0.1 % takes 10-20 ms).
  A ramp scan with the waveform playing is the Ramp scan tab.
- **At the end** (window closed, Disconnect, a bias run or AWG-held Find):
  `off`, or `park` = an idle-level waveform with the outputs left ON - for
  when the drive goes through the X2 FPGA/buffer stage, whose output goes
  high on a floating input (the pull-down is not fitted). Closing the window
  first lets a running measurement finish its own cleanup, then ends the
  AWG, then closes the scope.

## Bias points (Bias points tab)

A ramp scan reads every angle at the V/div the brightest needs, so near a
null the scope resolves ~1 code: on test-4 the rest/hold minima (2.6-3 mV)
were one step at 1 V/div - ER ~1800 there is the scope's floor. Mid-ramp
(166-281 mV, ER 18-31) they were well resolved.

Bias points hold the EOMs at fixed rotations with the AWG (4063B; close its
GUI; CH1 -> X1, CH2 -> X2; plateaus on the bench trigger, EXT, each checked
with EOM-ILC's limit check before upload). They use the AWG tab's session,
idle levels and end policy; the plateaus are the ILC's record length (5501
points), so the ILC's FRQ check passes afterwards, and the outputs go off
for each change of bias. Per bias:

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

Next to the scan folders: `lab_log.csv`, one row per scan, bias run and found
angle.

The capture names split as Scope Grab expects (`prefix_NNN`), so its Compare
box opens any angle: key `<name>_a045.00`, runs `1-4`.

## Tabs

| tab | shows |
|---|---|
| Traces | PD at every analyzer angle (colour = angle), dark dashed; monitors below |
| Map | I(t, theta) / Imax(t), with the fitted null psi + 90 drawn over it; or the Malus-fit residual (mV, or per standard error) per angle and time, with the rms per angle beside it: a bad angle, clipping, a missed lock or drift shows as a row or a patch |
| Malus | I vs analyzer angle at the cursor time, the fit, residuals |
| Angle | rotation from rest with +-1 SD, the monitor prediction, their difference in mdeg |
| Extinction | ER_fit (smoothed by the plot bar's Smooth box), dip points (rising/falling), the direct points (crossings, lower bounds, static, offset-limited), refine points, the drift limit; each family switchable; x = time or rotation |
| Poincaré | the linear Stokes parameters in the rest frame on the sphere (coloured by time), |S3| and the ellipticity angle chi vs time assuming full polarization (handedness not measured), the ellipse at the cursor against the rest ellipse |
| Diagnostics | ref returns vs time, 1-theta and 4-theta amplitudes, residual vs block SEM, landing error and off-screen samples per step |
| Table | per-segment medians, every refine, dip and direct ER; Save CSV |
| Shots | the data behind every number: pick steps (several with ctrl/shift), a channel, and any of single shots straight from the files, the average the fit uses, +-1 SE, the min-max over the shots, the dropped (missed-lock) shots dashed, the analyzer 90 deg away; a shot list (`1, 3-5`) and a time window - zoom with the toolbar and the view re-reads the files at full resolution. *Crossed at cursor* picks the angle nearest crossed at the cursor time with its partner: the direct-ER view |
| Build | how the angles become the polarization: a time slider; top every angle's averaged trace as fitted, bottom left the points at that instant (and before corrections) with the Malus fit a0 + B cos 2(theta - psi), its maximum and null, bottom right the rotation with the instant marked |
| Corrections | what is subtracted (dark / background traces and levels, borrowed ones dashed), the reference drift, the per-angle transmission, shots kept and dropped per step |
| Compare | the shown scan against up to 6 others picked in a list: rotation, the difference from the shown scan (smoothed, with the shown scan's +-1 SD), ER_fit and the direct points; each with the window's Apply switches |
| Find angle | the last Find angle scan and its fit |
| AWG | the AWG tab's waveform per channel, the rotation it gives, the hold and the Find window |
| Bias points | static ER vs rotation (Imax/Imin and from the null curvature), light - monitors static (and the shown scan's ramp), Imin with the V/div it was read at, the last null scan |
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
- **Inter-channel skew** of ~97 ns between the CH1/CH2 and CH3/CH4 pairs:
  0.015 deg at the fastest ramp rate - negligible here, noted for anything
  finer.

## Tests

```
python tests/run_tests.py
```

| suite | covers |
|---|---|
| `test_ell14.py` | the driver against a fake serial port (the real mount's IN reply), the approach-from-below wrapper |
| `test_analysis.py` | the harmonic fit exact on noise-free data, its uncertainties checked by pulls (unit spread), lower bounds; a 72-angle simulated scan written and read back: rotation, rest azimuth, drift correction, 142 dip ERs against the model, direct ERs against the model and their Imax against the fit, the residual map at unit noise, the Stokes identities, segments, monitor prediction; provenance (this repository's commit, an ILC state's fingerprint) and the lab log's update-in-place |
| `test_checks.py` | the pre-run check against the simulator: a hand-changed scope put back by a preset and confirmed, a silently refused setting reported, and each failure it should catch (AUTO sweep, channel off, wait <= repetition, AC coupling, clipping, off screen, small signal, ramp cut off, no light, no pre-trigger) |
| `test_awg.py` | ramps (the ILC's record, ends at idle, the hold at the rotation), every check (length, trigger period, duty, idle and AWG caps), ILC drive files (header, a target refused, keepers against their targets pass and as u x gain fail), the session against the simulated AWG (names reused, a foreign ON refused, OFF for a change, park, a live FRQ change refused, CH2 refusing ON leaves nothing on) |
| `test_bias.py` | the plan (AWG volts, plateaus, the Trek limit check), the null fit and ER, and a whole bias run on the simulated bench (AWG plateaus into the bench model): ER at 0-90 deg against the model, a 2 deg static rotator error recovered, outputs off and scope restored at the end |
| `test_gui.py` | the window against the simulator: connect, dark, scan, every tab drawn, cursor, null refine of rest/hold/after against the model ER, a bias run from the Bias points tab, the direct points, residual map, Poincaré, Compare, Export brief, provenance and lab-log rows; config sandboxed, window off screen |

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

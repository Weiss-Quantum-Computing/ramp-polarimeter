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

**Which ER to believe.** `ER_fit` needs the light level to be the same at
every angle, so slow intensity drift between angles limits it: a 1e-3 gain
error between angles moves Imin by ~1e-3 x Imax, so ER_fit means little above
~1000 unless drift is controlled better than that. The scan therefore returns
to a reference angle every N angles (default 6), corrects the drift from
those returns, and reports how well they predict each other (leave-one-out);
the Extinction tab draws `1 / that scatter` as the limit. The dip and refine
measurements each come from a single angle's captures and do not have this
limit. The analyzer's own ER (LPVIS100 spec floor 1e4) caps everything; the
table reports `ER light` with it divided out.

## Using it

1. **Connect** the scope (VISA address blank = first MSO-X found) and the
   ELL14 (COM3 on this PC). `Simulate both` runs the whole window against a
   software bench instead, which is also what the tests use.
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
   `dark first` asks you to block the beam before the analyzer, takes the
   dark, then asks you to unblock it.
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
6. **Null refine**: windows `auto` (rest, hold, after) or `t1-t2` in ms,
   offsets around crossed, the PD V/div at the null. It takes a dark at that
   V/div first (block the beam when asked).

A scan that stops can be resumed: start a scan with the same name and answer
Yes.

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

## Bias points (Bias points tab)

A ramp scan reads every angle at the V/div the brightest needs, so near a
null the scope resolves ~1 code: on test-4 the rest/hold minima (2.6-3 mV)
were one step at 1 V/div - ER ~1800 there is the scope's floor. Mid-ramp
(166-281 mV, ER 18-31) they were well resolved.

Bias points hold the EOMs at fixed rotations with the AWG (4063B; close its
GUI; CH1 -> X1, CH2 -> X2; plateaus on the bench trigger, EXT, each checked
with EOM-ILC's limit check before upload) and, per bias:

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
| `<name>_scan.json` | manifest: plan, channel roles, scope and mount identity, the analyzer zero, every step with target, landed angle, clock, V/div and offset, file names; rewritten after every step |
| `<name>_a045.00_001.npz` + `.txt` | one dither block at analyzer 45.00 deg - Scope Grab's NPZ and sidecar, written by its own `write_capture` |
| `<name>_ref003_001...` | reference-angle returns |
| `<name>_dark_001...` | beam blocked |
| `<name>_n2_a137.25_001...` | null refine, window 2 |

The capture names split as Scope Grab expects (`prefix_NNN`), so its Compare
box opens any angle: key `<name>_a045.00`, runs `1-4`.

## Tabs

| tab | shows |
|---|---|
| Traces | PD at every analyzer angle (colour = angle), dark dashed; monitors below |
| Map | I(t, theta) / Imax(t), with the fitted null psi + 90 drawn over it |
| Malus | I vs analyzer angle at the cursor time, the fit, residuals |
| Angle | rotation from rest with +-1 SD, the monitor prediction, their difference in mdeg; compare scan overlaid |
| Extinction | ER_fit (smoothed by the plot bar's Smooth box), dip points (rising/falling), refine points, the drift and analyzer limits; x = time or rotation |
| Diagnostics | ref returns vs time, 1-theta and 4-theta amplitudes, residual vs block SEM, landing error and off-screen samples per step |
| Table | per-segment medians, every refine and dip ER; Save CSV |
| Bias points | static ER vs rotation (Imax/Imin and from the null curvature), light - monitors static (and the shown scan's ramp), Imin with the V/div it was read at, the last null scan |
| ILC target | the ILC comparison's figures for the shown scan |

Click a time on Map, Angle or Extinction to move the Malus cursor.

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
| `test_analysis.py` | the harmonic fit exact on noise-free data, its uncertainties checked by pulls (unit spread), lower bounds; a 72-angle simulated scan written and read back: rotation, rest azimuth, drift correction, 142 dip ERs against the model, segments, monitor prediction |
| `test_checks.py` | the pre-run check against the simulator: a hand-changed scope put back by a preset and confirmed, a silently refused setting reported, and each failure it should catch (AUTO sweep, channel off, wait <= repetition, AC coupling, clipping, off screen, small signal, ramp cut off, no light, no pre-trigger) |
| `test_bias.py` | the plan (AWG volts, plateaus, the Trek limit check), the null fit and ER, and a whole bias run on the simulated bench (AWG plateaus into the bench model): ER at 0-90 deg against the model, a 2 deg static rotator error recovered, outputs off and scope restored at the end |
| `test_gui.py` | the window against the simulator: connect, dark, scan, every tab drawn, cursor, null refine of rest/hold/after against the model ER, a bias run from the Bias points tab; config sandboxed, window off screen |

## What has and has not run on hardware

- **Verified, 5 Oct 2026:** the ELL14 (S/N 11400318, firmware 13, COM3)
  answered `in`, `gs` and `gp` through this driver: 143360 pulses/rev, status
  OK.
- **Run on hardware 5 Oct 2026:** ELL14 motion and full ramp scans
  (16-ms-spin-echo-test-1..5).
- **Not yet run on hardware:** bias points (AWG control from this program),
  the null refine at a sensitive V/div, and an optical correction applied
  through the ILC. These are tested against the simulator only. A first bias
  session: a short list (`0, 90`), 4 shots, watching the first plateau on
  the scope before the full 0-180.

## Provenance

`rampol/ell14.py` is the reviewed driver from EOM-ILC's
`eomilc_polarization_finetune` (a556aff); this repository now holds the
canonical copy.

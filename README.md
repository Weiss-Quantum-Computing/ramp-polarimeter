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
| `test_gui.py` | the window against the simulator: connect, dark, scan, every tab drawn, cursor, null refine of rest/hold/after against the model ER; config sandboxed, window off screen |

## What has and has not run on hardware

- **Verified, 5 Oct 2026:** the ELL14 (S/N 11400318, firmware 13, COM3)
  answered `in`, `gs` and `gp` through this driver: 143360 pulses/rev, status
  OK.
- **Not yet run on hardware:** ELL14 motion (home, absolute and relative
  moves, the approach), any scan, the null refine at a sensitive V/div, the
  preset writes. Everything else is tested against the simulator only. The
  first bench session should home the mount, check `Go to` lands within
  0.05 deg in both directions, then run a short scan (0-350 step 30, 16
  shots) before a full one.

## Provenance

`rampol/ell14.py` is the reviewed driver from EOM-ILC's
`eomilc_polarization_finetune` (a556aff); this repository now holds the
canonical copy.

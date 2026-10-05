# Ramp Polarimeter

Polarization and extinction ratio through an EOM ramp, measured with a
Thorlabs ELL14 rotating the analyzer (LPVIS100-MP2) in front of the PDA10A2,
and the MSO-X 2014A read through [Scope Grab](../scope-grab-multi). The
window follows Scope Grab's layout: controls on the left, plot tabs with a
plot bar on the right, log underneath.

```
Start Polarimeter.bat            (or: pythonw polarimeter.pyw / python -m rampol)
```

Needs the system Python 3.13 with numpy, matplotlib, pyvisa and pyserial (all
installed on this PC; pyserial sits in the per-user site-packages that both
Pythons share). Scope Grab is loaded by file path from `scope_grab_path` in
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
3. **Preset**: `AWG bench ramp` or `Spin-echo sequence`. `Apply to scope`
   writes the preset's timebase and trigger source; otherwise the scan runs on
   whatever the scope is set to. Set V/div so the PD stays on screen at every
   angle (the Diagnostics tab and the log flag off-screen samples).
4. **Scan**: angles, order (`bidirectional` or `shuffled` keep slow drift from
   lining up with angle), mode, shots, blocks, dither, ref returns.
   - `average`: the scope averages `shots / blocks` triggers per block
     (:DIGitize). An averaged MSO-X record reads out at **7680 points**.
   - `single`: one HRES shot per file, read at `Points (single)`; slower,
     but per-shot statistics and longer records.
   - The channel offsets step across `dither codes` ADC codes over the blocks,
     so the MSO-X per-code error pattern averages out instead of surviving
     every block identically.
   - Every angle is approached from below (`backoff` deg), so backlash always
     lands on the same side.
   `dark first` asks you to block the beam before the analyzer, takes the
   dark, then asks you to unblock it.
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

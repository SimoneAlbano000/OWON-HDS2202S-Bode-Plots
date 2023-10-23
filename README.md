# OWON HDS2202S Bode Plots
The following utility written in Python adds the Bode Plot/Frequency Response capability for both magnitude and phase to the handheld oscilloscope OWON HDS2202S, Dual channel, 200MHz Bandwith, 1GSa Max Sample Rate with integrated AWG (Should also work for all the handheld oscilloscope from OWON in the HDS200 Series with integrated AWG).

## Required packages:
```
pip install pyusb numpy matplotlib scipy
```
## Utility usage and oscilloscope setup
- Connect the oscilloscope to the PC using the included usb cable.
- Set the oscilloscope interface to HID in order to enable the SCPI protocol.
- Connect all the probes to the two input of the oscilloscope and one connector to the AWG output.
- After launching the utility, make sure all the oscilloscope and AWG settings match with the one
  reported by the instrument (specially the correct probe attenuation switch).
- Before the first measurement, adjust both the vertical and horizontal scale in order to have
  at least one entire period of the waveform visible on screen.

## Some examples of common Transfer Functions:

# RC Low Pass filter:
The value of the component are R = 470Ω and C = 0.1μF. 
The theoretical pole frequency is given by the formula: $`f_{p} =\frac{1}{2\pi RC} = 3.386kHz`$.

![BodePlot_low-pass-filter-470ohm-0 1uF_1X_1X_100Hz-1MHz](https://github.com/SimoneAlbano000/OWON-HDS2202S-Bode-Plots/assets/36369471/9b57cb8c-e8a0-4f9c-8df7-39931c7174d0)

# RC High Pass filter:
The value of the component are R = 470Ω and C = 0.1μF. 
The theoretical pole frequency is given by the formula: $`f_{p} =\frac{1}{2\pi RC} = 3.386kHz`$.

![BodePlot_high-pass-filter-470ohm-0 1uF_1X_1X_100Hz-1MHz](https://github.com/SimoneAlbano000/OWON-HDS2202S-Bode-Plots/assets/36369471/dc13d212-908a-488a-bada-2cbe9bd00fc5)

# RC BandPass filter:
In this particular example we can appreciate the difference between an LTSpice simulation of the circuit and a real Frequency Response Analysis. The difference lies in the behaviour of the capacitor C2 at high frequencies, where it's parasitic series inductance prevails over the desired capacitive proprierties.

LTSpice AC Analysis:

![Bodeplot_bandpass-filter_1X_1X_maybe_parasitic_sym](https://github.com/SimoneAlbano000/OWON-HDS2202S-Bode-Plots/assets/36369471/8114cea8-ef20-4e33-b444-006ed3f50fd8)

OWON HDS2202S Analysis:

![Bodeplot_bandpass-filter_1X_1X_maybe_parasitic](https://github.com/SimoneAlbano000/OWON-HDS2202S-Bode-Plots/assets/36369471/c1848be0-f084-42d0-ba1e-a47f3d93264e)

```
Executable build with PyInstaller 6.1.0
```

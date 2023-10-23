# Writte by Simone Albano on 15/10/2023
# Bode plot implementation for the handheld oscilloscope OWON HDS2202S

import usb.core
import usb.util
import numpy as np
import matplotlib.pyplot as plot
from scipy.interpolate import UnivariateSpline
import scipy.optimize as optimize
import time

# --- user variables --- ------------------------------------------
channel_out = 'CH2'
channel_in = 'CH1'
CH1_probe_attenuation_ratio = '1X' # 1X or 10X, remember to change the swich on the probe!
CH2_probe_attenuation_ratio = '1X' # 1X or 10X, remember to change the swich on the probe!
CH1_coupling = 'AC' # AC or DC coupling (for Bode plots use AC coupling!)
CH2_coupling = 'AC' # AC or DC coupling (for Bode plots use AC coupling!)
waveform_amplitude_V = 5 # pk-pk
AWG_output_impedance = 'ON' # OFF = 50 ohm = the output voltage will be doubled!, ON = High Z = the output voltage will not be modified
#    0       1      2      3      4      5      6      7      8
# [1E-1Hz, 1E0HZ, 1E1Hz, 1E2Hz, 1E3Hz, 1E4Hz, 1E5Hz, 1E6Hz, 1E7Hz]
points_per_decade = 30
start_decade = 2
stop_decade = 8
spline_points = 1000000
# --- end user variables --- --------------------------------------

# --- system variables --- ----------------------------------------
time_bases_commands = ['2.0ns', '5.0ns', '10.0ns', '20.0ns', '50.0ns', '100ns', '200ns', '500ns', '1.0us', '2.0us', '5.0us', '10us', '20us', '50us', '100us', '200us', '500us', '1.0ms', '2.0ms', '5.0ms', '10ms', '20ms', '50ms', '100ms', '200ms', '500ms', '1.0s', '2.0s', '5.0s', '10s', '20s', '50s', '100s', '200s', '500s', '1000s']
time_bases_values = [0.000000002, 0.000000005, 0.00000001, 0.00000002, 0.00000005, 0.0000001, 0.0000002, 0.0000005, 0.000001, 0.000002, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
amplitude_scales_commands = ['100V', '50V', '10.0V', '5.00V', '2.00V', '1.00V', '500mV', '200mV', '100mV', '50.0mV', '20.0mV', '10.0mV']
amplitude_scales_values = [100, 50, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
decades_list = [1E-1, 1E0, 1E1, 1E2, 1E3, 1E4, 1E5, 1E6, 1E7]

point_resize_factor = 5850
read_delay_ms = 250 # ocilloscope read timeout
read_buffer_size = 100000 # buffer size in bytes
sample_delay_s = 0.5 # delay between operations
vertical_scaling_factor = 0.25 # used for optimal vertical scale calibration
horizontal_scaling_factor = 0.25 # used for optimal horizontal scale calibration

horizontal_decades_n = 6 # 12 in total
vertical_decades_n = 4 # 8 in total

raw_frequencies_range = []
gain_linear = []
gain_dB = []
phase_radiant = []
phase_degree = []

# limit the amplitude_scales based on the choosen probe_attenuation_value (see datasheet for allowed values)
if CH1_probe_attenuation_ratio == '1X':
    CH1_amplitude_scales = amplitude_scales_commands[2:]
    CH1_amplitude_scales = amplitude_scales_values[2:]
elif CH1_probe_attenuation_ratio == '10X':
    CH1_amplitude_scales = amplitude_scales_commands[:9]
    CH1_amplitude_scales = amplitude_scales_values[:9]
if CH2_probe_attenuation_ratio == '1X':
    CH2_amplitude_scales = amplitude_scales_commands[2:]
    CH2_amplitude_scales = amplitude_scales_values[2:]
elif CH2_probe_attenuation_ratio == '10X':
    CH2_amplitude_scales = amplitude_scales_commands[:9]
    CH2_amplitude_scales = amplitude_scales_values[:9]
# --- end system variables --- ------------------------------------

# Specifications for OWON HDS2202S Handheld Oscilloscope
oscilloscope = usb.core.find(idVendor=0x5345, idProduct=0x1234)

if oscilloscope is None:
    input('HDS2202S not found, press any key to continue...')
    exit()
else:
    oscilloscope.set_configuration()
    # get an endpoint instance
    config = oscilloscope.get_active_configuration()
    intf = config[(0,0)]

    oscilloscope_OUT = usb.util.find_descriptor(
    intf,
    # match the first OUT endpoint
    custom_match = \
    lambda e: \
        usb.util.endpoint_direction(e.bEndpointAddress) == \
        usb.util.ENDPOINT_OUT)
    assert oscilloscope_OUT is not None

    oscilloscope_IN = usb.util.find_descriptor(
    intf,
    # match the first IN endpoint
    custom_match = \
    lambda e: \
        usb.util.endpoint_direction(e.bEndpointAddress) == \
        usb.util.ENDPOINT_IN)
    assert oscilloscope_IN is not None

    # print endpoint infomation for debug
    #print(oscilloscope_IN)
    #print(oscilloscope_OUT)

def oscilloscope_query(cmd):
    oscilloscope_OUT.write(cmd)
    result = (oscilloscope_IN.read(read_buffer_size, read_delay_ms))
    return result.tobytes().decode('utf-8').strip()

def oscilloscope_write(cmd):
    oscilloscope_OUT.write(cmd)

def oscilloscope_read(cmd):
    oscilloscope_OUT.write(cmd)
    result = (oscilloscope_IN.read(read_buffer_size, read_delay_ms))
    return result

def AWG_set_frequency(frequency):
    # set AWG frequency in Hz (0.1Hz to 25MHz for SINE waveform)
    oscilloscope_OUT.write(':FUNCtion:FREQuency {}'.format(frequency))

def set_time_base(period):
    # set the oscilloscope time-base
    oscilloscope_OUT.write(':HORizontal:SCALe {}'.format(period))

def set_amplitude_scale(channel, scale):
    # set the oscilloscope amplitude scale
    oscilloscope_OUT.write(':{CH}:SCALe {data}'.format(CH = channel, data = scale))

def get_pkpk_voltage(channel):
    return oscilloscope_query(':MEASurement:{CH}:PKPK?'.format(CH = channel)).lstrip('Vpp=')

def vertical_scale_to_float(voltage):
    # if the symbol 'm' for 'milli' is present in the reading, remove it and scale down the mesurement
    if 'mV' in voltage:
        voltage = voltage.rstrip('mV')
        return (float(voltage) / 1E3)
    else:
        voltage = voltage.rstrip('V')
        return float(voltage)
    
def horizontal_scale_to_float(timescale):
    # remove all the units, scale the value and convert it to float 
    if 'ns' in timescale:
        timescale = timescale.rstrip('ns')
        return (float(timescale) / 1E9)
    elif 'us' in timescale:
        timescale = timescale.rstrip('us')
        return (float(timescale) / 1E6)
    elif 'ms' in timescale:
        timescale = timescale.rstrip('ms')
        return (float(timescale) / 1E3)
    else:
        timescale = timescale.rstrip('s')
        return (float(timescale))

def get_waveform(channel, v_scale):
    # the first 4 bytes are discarted, in total there are 600 points
    rawdata = oscilloscope_read(':DATA:WAVE:SCREen:{}?'.format(channel))
    data_points = []
    for val in range(4, len(rawdata), 2):
        # take 2 bytes and convert them to signed integer using "little-endian"
        point = int().from_bytes([rawdata[val], rawdata[val + 1]],'little',signed=True)
        data_points.append((point*v_scale)/point_resize_factor)
    return data_points

def guess_phase(x, y, f, A, h_scale):
    # IMPORTANT! in the screen there must be at least half the waveform displayed
    # guessing the phase based on the value of the area under the curve between zero and T/2
    # approximately slice y values in half a sine period
    x_half_period_end = int((1/(2*f*h_scale))*(300/(horizontal_decades_n*2)))
    y_half_positive_period = (y.tolist())[150:(150 + x_half_period_end)]
    y_half_negative_period = (y.tolist())[(150 - x_half_period_end):150]
    # calculate the approximate phase inverting the integral formula for the sine function
    cos_phi_positive = ((np.pi*f)/A)*np.trapz(y = y_half_positive_period, dx = h_scale/(300/(horizontal_decades_n*2)))
    cos_phi_negative = ((np.pi*f)/A)*np.trapz(y = y_half_negative_period, dx = h_scale/(300/(horizontal_decades_n*2)))
    # for gaining precison, compute the average of the two
    cos_phi = (abs(cos_phi_positive) + abs(cos_phi_negative))/2
    if cos_phi >= 1:
        cos_phi = 1
        phi = np.arccos(cos_phi)
        return phi
    else:
        phi = np.arccos(cos_phi)
        # check for phase sign
        if (y.tolist())[150] <= 0:
            return (-phi)
        else:
            return phi

def fit_sine(x, y, frequency, h_scale):
    x = np.array(x)
    y = np.array(y)
    """ in this case we already know the frequency, no need to make guesses
    # extract the sample frequency from the FFT of the signal using x[1] - x[0] as timestep
    fft_sample_freq_array = np.fft.fftfreq(len(x), (x[1] - x[0]))
    # take the input waveform FFT
    waveform_fft = abs(np.fft.fft(y))
    # guess the sine frequency excluding the zero frequency "peak", which is related to offset
    guess_freq = abs(fft_sample_freq_array[np.argmax(waveform_fft[1:]) + 1])
    """
    guess_freq = frequency
    guess_amp = np.std(y) * 2.**0.5
    guess_offset = np.mean(y)
    guess_phase_rad = guess_phase(x, y, guess_freq, guess_amp, h_scale)
    guess = np.array([guess_amp, (2.*np.pi*guess_freq), guess_phase_rad, guess_offset])
    # define the test function that scipy will try to fit into the data points
    def sinfunc(t, A, w, p, c): return A * np.sin(w*t + p) + c
    # call the optimize.curve_fit function that try to fit the data points
    popt, _ = optimize.curve_fit(sinfunc, x, y, p0 = guess) # covariance matrix not used
    A, w, p, c = popt
    return {"amplitude": A, "omega": w, "phase": p, "offset": c}

# --- main program start ---
# general device config
print('\n --- Oscilloscope configurations --- \n')
print(oscilloscope_query('*IDN?') + '\n')
# set-up channels
oscilloscope_query(':{CH}:DISPlay OFF'.format(CH = channel_out))
print('Channel out status: ' + oscilloscope_query(':{CH}:DISPlay?'.format(CH = channel_out)).upper())
oscilloscope_query(':{CH}:DISPlay OFF'.format(CH = channel_in))
print('Channel in status: ' + oscilloscope_query(':{CH}:DISPlay?'.format(CH = channel_in)).upper())
# set AC coupling
oscilloscope_query(':{CH}:COUPling {data}'.format(CH = 'CH1', data = CH1_coupling))
print('Channel 1 coupling: ' + oscilloscope_query(':{CH}:COUPling?'.format(CH = 'CH1')).upper())
oscilloscope_query(':{CH}:COUPling {data}'.format(CH = 'CH2', data = CH2_coupling))
print('Channel 2 coupling: ' + oscilloscope_query(':{CH}:COUPling?'.format(CH = 'CH2')).upper())
# set attenuation mode
oscilloscope_query(':{CH}:PROBe {data}'.format(CH = 'CH1', data = CH1_probe_attenuation_ratio))
print('Channel 1 probe attenuation ratio: ' + oscilloscope_query(':{CH}:PROBe?'.format(CH = 'CH1')).upper())
oscilloscope_query(':{CH}:PROBe {data}'.format(CH = 'CH2', data = CH2_probe_attenuation_ratio))
print('Channel 2 probe attenuation ratio: ' + oscilloscope_query(':{CH}:PROBe?'.format(CH = 'CH2')).upper())
# turn on frequency and amplitude pk-pk mesurements
oscilloscope_query(':MEASurement:DISPlay ON')
# set acquire mode
oscilloscope_query(':ACQuire:MODE SAMPle')
print('Acquisition mode: ' + oscilloscope_query(':ACQuire:MODE?').upper())
# set memory depth
oscilloscope_query(':ACQuire:DEPMEM 8K')
print('Memory depth: ' + oscilloscope_query(':ACQuire:DEPMEM?').upper())
# set the trigger to rising edge, VERY IMPORTANT!

# setup the AWG
print('\n --- AWG configurations --- \n')
# turn off the AWG
oscilloscope_query(':CHANnel OFF')
print('Channel status: ' + oscilloscope_query(':CHANnel?').upper())
# set the output waveform: for bode plots the sine waveform is used
oscilloscope_query(':FUNCtion SINE')
print('Output waveform: ' + oscilloscope_query(':FUNCtion?').upper())
# set the waveform amplitude
oscilloscope_query(':FUNCtion:AMPLitude {}'.format(waveform_amplitude_V))
print('Waveform amplitude (pk-pk): ' + str(float(str(oscilloscope_query(':FUNCtion:AMPLitude?'))[0:8])) + 'V')
# set output impedance
oscilloscope_query(':FUNCtion:LOAD {}'.format(AWG_output_impedance))
print('High Output impedance?: ' + oscilloscope_query(':FUNCtion:LOAD?').upper() + ' Ω')
# set the waveform offset to zero
oscilloscope_query(':FUNCtion:OFFSet 0')
print('Waveform offset: ' + str(float(str(oscilloscope_query(':FUNCtion:OFFSet?'))[0:8])) + 'V\n')

# --- main loop ---
# turn on the device at correct initial range
oscilloscope_write(':{CH}:DISPlay ON'.format(CH = 'CH1'))
oscilloscope_write(':{CH}:DISPlay ON'.format(CH = 'CH2'))
oscilloscope_write(':CHANnel ON')
AWG_set_frequency(decades_list[start_decade]) # reduce start-up time
# wait for user input before starting the mesurements
input('Perform the AUTO reading and adjust initial scales, then press ENTER...')
# set the vertical offset of both channels to zero
oscilloscope_write(':CH1:OFFSet 0')
oscilloscope_write(':CH2:OFFSet 0')
time.sleep(sample_delay_s*4)

# generate the complete test frequency range
for indx in range(start_decade, stop_decade, 1):
    current_frequency_range = np.linspace(decades_list[indx], decades_list[indx + 1], points_per_decade)
    for indy in current_frequency_range:
        raw_frequencies_range.append(indy)
    # remove the last, duplicated value
    if indx != (stop_decade - 1):
        raw_frequencies_range.pop()

for index, frequency in enumerate(raw_frequencies_range):
    # ask for the current vertical and horizontal scale (only the first time)
    current_v_scale = vertical_scale_to_float(oscilloscope_query(':{}:SCALe?'.format(channel_out)))
    time.sleep(sample_delay_s)
    current_h_scale = horizontal_scale_to_float(oscilloscope_query(':HORIzontal:SCALe?'))
    time.sleep(sample_delay_s)
    # compute the time range for the waveform acquisition
    raw_frequencies_x = np.linspace(-(horizontal_decades_n*current_h_scale), (horizontal_decades_n*current_h_scale), 300) # constant x size of 300 points
    # set the current test frequency
    AWG_set_frequency(frequency)
    time.sleep(sample_delay_s)
    # ask for the complete datapoint array of the output channel
    raw_waveform_y = get_waveform(channel_out, current_v_scale)
    time.sleep(sample_delay_s)
    # process the data using the optimize.curve_fit and FFT functions
    fitted_waveform_param = fit_sine(raw_frequencies_x, raw_waveform_y, frequency, current_h_scale)
    # save the linear amplitude and phase in radiant information (mediate amplitude with the one calculated by the oscilloscope itself)
    Vpkpk_from_curve = 2*fitted_waveform_param["amplitude"]
    Vpkpk_measured = vertical_scale_to_float(get_pkpk_voltage(channel_out))
    Vpkpk = (Vpkpk_from_curve + Vpkpk_measured)/2
    gain_linear.append(Vpkpk)
    phase_radiant.append((fitted_waveform_param["phase"]))
    # adjust the vertical and horizontal scale based on previous waveform characteristic
    closest_v_scale_index = amplitude_scales_values.index(min(amplitude_scales_values, key=lambda x:abs(x-(Vpkpk*vertical_scaling_factor))))
    set_amplitude_scale(channel_out, amplitude_scales_commands[closest_v_scale_index])
    time.sleep(sample_delay_s)
    closest_h_scale_index = time_bases_values.index(min(time_bases_values, key=lambda x:abs(x-((1/frequency)*horizontal_scaling_factor))))
    set_time_base(time_bases_commands[closest_h_scale_index])
    time.sleep(sample_delay_s)

# --- start post-processing ---
# transform the linear gain in a logarithmic one
for linear_gain_value in range(0, len(raw_frequencies_range), 1):
    gain_dB.append(20*np.log10(gain_linear[linear_gain_value]/waveform_amplitude_V))
# transform the phase from radiant to degrees
for radiant_phase_value in range(0, len(raw_frequencies_range), 1):
    phase_degree.append(np.degrees(phase_radiant[radiant_phase_value]))

print(raw_frequencies_range)
print(gain_dB)
print(phase_degree)

figure, (magnitude, phase) = plot.subplots(2)
magnitude.set_xlim([decades_list[start_decade], decades_list[stop_decade]])
phase.set_xlim([decades_list[start_decade], decades_list[stop_decade]])
x_points = np.linspace(decades_list[start_decade], decades_list[stop_decade], spline_points)

# plotting the magnitude
magnitude.set_title('Magnitude Bode plot - Frequency response')
magnitude.set_xlabel('Frequency (Hz)')
magnitude.set_ylabel('Gain (dB)')
magnitude.set_xscale('log')
magnitude.grid(which='both')
magnitude.scatter(raw_frequencies_range, gain_dB, color='red', label='raw data:\n{} points per decade'.format(points_per_decade), s = 20)
magnitude_spline = UnivariateSpline(raw_frequencies_range, gain_dB, k = 3, s = 0)
magnitude.semilogx(x_points, magnitude_spline(x_points), color='blue', label='magnitude spline')

# plotting the phase
phase.set_title('Phase Bode plot - Frequency response')
phase.set_xlabel('Frequency (Hz)')
phase.set_ylabel('Degrees°')
phase.set_xscale('log')
phase.grid(which='both')
phase.scatter(raw_frequencies_range, phase_degree, color='orange',label='raw data:\n{} points per decade'.format(points_per_decade), s = 20)
phase_spline = UnivariateSpline(raw_frequencies_range, phase_degree, k = 3, s = len(raw_frequencies_range))
phase.semilogx(x_points, phase_spline(x_points), color='blue', label='phase spline')

# subtract 3 to the y values in order to solve for y = -3
spline_point = UnivariateSpline(raw_frequencies_range, [x + 3 for x in gain_dB], k = 3, s = 0)
try:
    roots = spline_point.roots()
    for j in range(len(roots)):
        magnitude.axhline(y = -3, color='k', linestyle='--')
        magnitude.axvline(x = roots[j], color='k', linestyle='--')
        magnitude.scatter(roots[j], -3, color='black', label='-3dB point: {}Hz'.format(np.round(roots[j], 2)))
        phase.axhline(y = phase_spline(roots[j]), color='k', linestyle='--')
        phase.axvline(x = roots[j], color='k', linestyle='--')
        phase.scatter(roots[j], phase_spline(roots[j]), color='black', label='Phase angle: {}°'.format(np.round(phase_spline(roots[j]), 2)))
except:
    print('-3dB root error!')

magnitude.legend(loc='upper right')
phase.legend(loc='upper right')
figure.tight_layout()
plot.show()
# --- end post-processing ---

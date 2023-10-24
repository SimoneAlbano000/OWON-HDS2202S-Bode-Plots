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
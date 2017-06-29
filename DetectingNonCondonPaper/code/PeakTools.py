import math
import warnings
import heapq

import numpy as np
import scipy.optimize
import scipy.integrate
import matplotlib.pyplot as plt

from scipy.special import wofz
DEFAULT_c_order = 2
regularization_constant = 0.50
NUM_LSTSQ_TRIES = 10
ZERO_TOLERANCE = 1E-4

def r_squared(data, prediction):
    data = np.array(data)
    prediction = np.array(prediction)
    data_average = np.average(data)
    sum_squares_total = np.sum((data - data_average)**2)
    sum_squares_regression = np.sum((prediction - data_average)**2)
    sum_squares_residuals = np.sum((prediction - data)**2)
    r2 = 1 - sum_squares_residuals / sum_squares_total
    return r2

def gaussian_spectra(x, centers, heights, widths):
    output = 0.0
    for i in range(len(centers)):
        c = centers[i]
        s = widths[i]
        h = heights[i] / (s * math.sqrt(2.0 * np.pi))
        output = output + h * np.exp(-(x - c)**2 / (2 * s**2))
    return output

def lorenztianian_spectra(x, centers, heights, widths):
    output = 0.0
    for i in range(len(centers)):
        c = centers[i]
        s = widths[i]
        h = heights[i]/ (s * np.pi)
        output = output + h / ( 1 + (x - c)**2/s**2)
    return  output

def voigt_spectra(x, centers, heights, widths):
    output = 0.0
    n_peaks = len(centers)
    for i in range(n_peaks):
        c = centers[i]
        h = heights[i]
        sigma = widths[i]
        gamma = widths[i + n_peaks ]

        z = ((x - c) + 1.0j * gamma) / (sigma * math.sqrt(2))
        new_peak = h * np.real(wofz(z))/ ( sigma * math.sqrt(2 * np.pi))
        output = output + new_peak

    return  output

def ordered_permutations(ordered_list_in, n_length):
    if n_length == 0:
        return []
    if len(ordered_list_in) < n_length:
        return []
    output = []
    for i, num in enumerate(ordered_list_in):
        if i == len(ordered_list_in) - 1:
            new_sublist = []
        else:
            new_sublist = ordered_list_in[i+1:]

        if n_length - 1 == 0:
            output.append([num])
        else:
            sub_permutations = ordered_permutations(new_sublist, n_length - 1)

            for sub_p in sub_permutations:
                new_p = [num] + sub_p
                output.append(new_p)
    return output


def pack_spectral_parameters(list_of_lists):
    output = []
    for l in list_of_lists:
        output += l
    return output
def unpack_spectral_parameters(number_values, list_in):
    return list_in[0:number_values], list_in[number_values:2*number_values], list_in[2*number_values:]

def left_zero_pad_spectra(x, y, ZERO_PAD=100):
    last_spectra_value = y[0]

    spectra_with_padding = np.ones(ZERO_PAD + y.shape[0]) * last_spectra_value
    spectra_padding = np.ones(ZERO_PAD) * last_spectra_value

    spectra_with_padding[0:ZERO_PAD] = spectra_padding
    spectra_with_padding[ZERO_PAD:] = y

    energy_with_padding = np.zeros(ZERO_PAD + y.shape[0])
    dE = x[1] - x[0]
    E_min = x[0]

    energy_padding = np.arange(E_min - dE * ZERO_PAD, E_min, dE)
    assert(energy_padding.shape[0] == ZERO_PAD)
    energy_with_padding[0:ZERO_PAD] = energy_padding
    energy_with_padding[ZERO_PAD:] = x

    return energy_with_padding, spectra_with_padding

def fit_spectral_peaks(x, y, approx_locations, approx_heights, function = "gaussian", width_guess = None):

    number_peaks = len(approx_locations)

    if function == "gaussian":
        fitting_function = gaussian_spectra
        number_widths = number_peaks

    elif function == "lorentzian":
        fitting_function = lorenztianian_spectra
        number_widths = number_peaks

    elif function == "voigt":
        fitting_function = voigt_spectra
        number_widths = number_peaks * 2

    else:
        raise Exception("Undefined Function to fit")

    def optimizer_function(args):
        centers, heights, widths = unpack_spectral_parameters(number_peaks, args)
        return y - fitting_function(x, centers, heights, widths)
        # return np.sqrt(np.abs(y - fitting_function(x, centers, heights, widths)))

    #make guesses for the width
    approx_widths = []
    if width_guess is None:
        width_guess = 10*(x[1] - x[0])

    try:
        n = len(width_guess)
        #this means we have a list
        assert(n == number_widths)
        approx_widths = width_guess
    except:
        width = width_guess
        for i in range(number_widths):
            approx_widths.append(width)

    #height normalization
    new_approx_heights = []
    for peak_i in range(number_peaks):
        h_0 = approx_heights[peak_i]
        c_0 = approx_locations[peak_i]
        w_0 = [approx_widths[peak_i]]
        if function == "voigt":
            w_0.append(approx_widths[peak_i + number_peaks])

        center_h = fitting_function(c_0, [c_0], [1.0], w_0)
        new_h = h_0 / center_h
        new_approx_heights.append(new_h)


    center_lower_bound = np.min(x)
    center_upper_bound = np.max(x)
    #let's try restricting the upper and lower bounds to be within a width_guess
    multiplier = 1.0
    c = np.array(approx_locations)[0:number_peaks]
    w = multiplier * np.array(approx_widths)[0:number_peaks]
    center_lower_bounds = list(c - w)
    center_upper_bounds = list(c + w)

    width_lower_bound = 0.0
    width_upper_bound = np.inf

    height_lower_bound = 0.0
    height_upper_bound = np.inf

    lower_bounds = []
    upper_bounds = []

    # lower_bounds = lower_bounds + number_peaks * [center_lower_bound]
    lower_bounds = lower_bounds + center_lower_bounds
    lower_bounds = lower_bounds + number_peaks * [height_lower_bound]
    lower_bounds = lower_bounds + number_widths * [width_lower_bound]

    # upper_bounds = upper_bounds + number_peaks * [center_upper_bound]
    upper_bounds = upper_bounds + center_upper_bounds
    upper_bounds = upper_bounds + number_peaks * [height_upper_bound]
    upper_bounds = upper_bounds + number_widths * [width_upper_bound]

    parameter_bounds = [lower_bounds, upper_bounds]

    initial_values = pack_spectral_parameters([approx_locations, new_approx_heights, approx_widths])

    # lst_sq_out = scipy.optimize.least_squares(optimizer_function, initial_values, method='lm')
    lst_sq_out = scipy.optimize.least_squares(optimizer_function, initial_values, bounds = parameter_bounds)

    return unpack_spectral_parameters(number_peaks, lst_sq_out["x"]), lst_sq_out

def HO_overlap(ground_index, excited_index, S):
    m = ground_index
    n = excited_index
    mn_factorial = math.factorial(m) * math.factorial(n)
    if S<0:
        warnings.warn("HO_overlap fed negative Huang Rhys factor!  performing absolute value operation")
        S = abs(S)
    outside_sum = (-1)**n * math.sqrt(math.exp(-S) * S**(m + n) / mn_factorial)
    sum_max = min([m,n])
    summation_total = 0
    for j in range(sum_max + 1):
        num = mn_factorial * (-1)**j * S**(-j)
        den = math.factorial(j) * math.factorial(m - j) * math.factorial(n - j)
        summation_total += num/den
    return outside_sum * summation_total

def vibrational_subspace_zero_operator(n):
    return np.zeros((n,n))

def vibrational_subspace_identity_operator(n):
    output = vibrational_subspace_zero_operator(n)
    np.fill_diagonal(output, 1.0)
    return output

def vibrational_subspace_creation_operator(n):
    vibrational_subspace = vibrational_subspace_zero_operator(n)
    for vibrational_index in range(n - 1):
        vibrational_subspace[vibrational_index, vibrational_index + 1] = math.sqrt(vibrational_index + 1)
    return vibrational_subspace.T

def vibrational_subspace_dimensionless_position_operator(n):
    creation = vibrational_subspace_creation_operator(n)
    annihilation = np.conj(creation.T)
    position = (creation + annihilation)
    return position

def vibrational_subspace_position_to_nth_power_operator(n, n_power):
    if n_power == 0:
        return vibrational_subspace_identity_operator(n)
    assert(n_power >=1)
    x = vibrational_subspace_dimensionless_position_operator(n)
    output = x
    for i in range(n_power-1):
        output = output.dot(x)
    return output

def harmonic_peak_height_ratio_klist_clist(k_list, S, c_list, c_order_list, initial_state=0):
    k_list = np.array(k_list)

    max_c_order = max(c_order_list)
    N_vib = 3 * max_c_order

    mu_operator = vibrational_subspace_identity_operator(N_vib)

    for i, c_value in enumerate(c_list):
        c_order = c_order_list[i]

        x_n_operator = vibrational_subspace_position_to_nth_power_operator(N_vib, c_order)

        # print("x_n_operator",x_n_operator)
        mu_operator = mu_operator + c_value * x_n_operator
    mu_vector = mu_operator[initial_state, 0:max_c_order+1]

    FCF_overlap_vector_k1 = np.array([[HO_overlap(i, k+1, S) for i in range(max_c_order+1)] for k in k_list])
    FCF_overlap_vector_k = np.array([[HO_overlap(i, k, S) for i in range(max_c_order+1)] for k in k_list]) #dimention of N_vib, num_k
    # print("FCF_overlap_vector_k", FCF_overlap_vector_k)
    numerator = np.abs(np.dot(FCF_overlap_vector_k1, mu_vector))**2
    denominator = np.abs(np.dot(FCF_overlap_vector_k, mu_vector))**2
    # raise Exception
    return numerator / denominator

def spectral_gaps(peak_energies):
    output = []
    for i in range(1, len(peak_energies)):
        output.append(peak_energies[i] - peak_energies[i-1])
    return output

def fit_condon(peak_height_list):
    n = len(peak_height_list)
    ratios = []
    for i in range(n-1):
        ratios.append(peak_height_list[i+1] / peak_height_list[i])
    def optimizer_function_handle(args):
        S = args[0]
        output = ratios - harmonic_peak_height_ratio_klist_clist(list(range(len(ratios))), S, [0.0], [1])
        return output

    initial_values = [ratios[0]]

    lst_sq_out = scipy.optimize.least_squares(optimizer_function_handle, initial_values, method='lm')


    return lst_sq_out["x"][0]



def best_fit_transition_parameters_clist(peak_height_list,
                                    c_order_list,
                                    number_lstSq_tries = NUM_LSTSQ_TRIES):

    number_data_points = len(peak_height_list)
    number_ratios = number_data_points - 1

    solution_constants = [] # the first term in the right hand side of the above equation

    #find the necessary ratios
    for ratio_index in range(number_ratios):
        new_peak_ratio = peak_height_list[ratio_index + 1] / peak_height_list[ratio_index ]
        solution_constants.append(new_peak_ratio)

    S_condon = fit_condon(peak_height_list)

    max_order = 0
    for order in c_order_list:
        if order > max_order:
            max_order = order
    n_c = len(c_order_list)
    best_cost = np.inf
    best_params = [S_condon] + n_c *[0.0]

    def optimizer_function_handle(args):
        S = args[0]
        c_list = args[1:]
        output = np.zeros((number_ratios + 1))

        output[1:] = solution_constants - harmonic_peak_height_ratio_klist_clist(list(range(number_ratios)), S, c_list, c_order_list)

        output[0] = regularization_constant * (S - S_condon)**2

        return output

    #Begin trying least squares
    switch_counter = 0
    for i in range(number_lstSq_tries):
        if i == 0:
            random_factor = 1.0
        else:
            random_factor = 0.0
        best_starting_guess_S = [S_condon]
        #new initial guesses for variables

        best_starting_guess_c = list(np.array(best_params[1:]) * ( 1 -  np.random.randn() * random_factor))

        initial_values = best_starting_guess_S + best_starting_guess_c

        # print("initial values", initial_values)

        lst_sq_out = scipy.optimize.least_squares(optimizer_function_handle, initial_values)#, method="lm")



        cost = lst_sq_out.cost

        if cost < best_cost:
            result = lst_sq_out
            best_params = lst_sq_out.x
            best_cost = cost
            switch_counter += 1
    print("Through running {} lstsq attempts, {} better models were found".format(number_lstSq_tries, switch_counter))

    jacobean = result["jac"]

    mean_squared_error = np.average((result["fun"])**2)

    number_fit_parameters = jacobean.shape[0]
    number_data_points = number_ratios
    df = number_data_points - number_fit_parameters
    try:
        cov_matrix = scipy.linalg.inv(jacobean.T.dot(jacobean)) * mean_squared_error
        relative_errors_for_CI = []
        for i, param in enumerate(result["x"]):
            try:
                err = 1.96 * math.sqrt(cov_matrix[i,i] / abs(df)) / param
            except:
                err = np.nan
            relative_errors_for_CI.append(err)

        print("\n\n relative parameter errors for order{}: {}\n\n".format(c_order_list, relative_errors_for_CI))
        print("check for big off-diagonal values: \n{} \n\n".format(cov_matrix))
    except scipy.linalg.LinAlgError:
        print("\n\n SINGULAR COV MATRIX FOR ORDER {}\n\n".format(c_order_list))

    #calculate best fit peaks
    best_S = result["x"][0]
    best_c_list = result["x"][1:]

    return (best_S, best_c_list), S_condon, best_cost, relative_errors_for_CI

def transition_dipole_functional_form(x_values, c_prime_values, c_orders):
    c_orders = list(c_orders)
    c_values = np.array(c_prime_values)

    output = 1
    for order_index, order in enumerate(c_orders):
        c = c_values[order_index]
        # print("{} * x^{}".format(c, order))
        output += c  * (x_values**order)
    return output


def predicted_peak_intensities_clist(indeces, S, c_list, c_orders, ratios = False):
    ratio_values = harmonic_peak_height_ratio_klist_clist(indeces, S, c_list, c_orders)
    if ratios:
        return ratio_values
    else:
        output = [1.0]
        for r in ratio_values:
            new_output = r * output[-1]
            output.append(new_output)

        return np.array(output)






methods_list = ["gaussian", "lorentzian", "voigt"]
methods_to_function = {"gaussian":gaussian_spectra, "lorentzian" : lorenztianian_spectra, "voigt":voigt_spectra}
class SpectraFit(object):
    def __init__(self, spectral_energies, spectral_amplitudes, approx_peak_energies, approx_peak_heights, width_guess):

        self.spectral_energies = spectral_energies
        self.spectral_amplitudes = spectral_amplitudes

        self.approx_peak_energies = approx_peak_energies
        self.approx_peak_heights = approx_peak_heights

        self.number_peaks = len(approx_peak_energies)

        self.width_guess = width_guess

        self.fit_storage = {}

        self.method_to_fit_params = {}
        self.methodOrderTuple_to_HO_params = {}
        # self.method_to_HO_params = {} deprecated

        self.method_to_anharmonic = {}


    def compare_spectral_fitting_methods(self):
        best_err_found = np.inf
        for method_string in methods_list:
            print("\n---------{}---------\n".format(method_string.upper()))
            spectra_func = methods_to_function[method_string]
            if method_string == "voigt":
                gaussian_fit = self.method_to_fit_params["gaussian"]
                lorentzian_fit = self.method_to_fit_params["lorentzian"]
                (a, b, gaussian_widths) = gaussian_fit
                (a, b, lorentzian_widths) = lorentzian_fit
                width_guess = list(gaussian_widths) + list(lorentzian_widths)

                #location should be the same
                location_guess = self.approx_peak_energies

                height_guess = self.approx_peak_heights
            else:
                height_guess = self.approx_peak_heights
                location_guess = self.approx_peak_energies
                width_guess = self.width_guess
            fit_params, fit = fit_spectral_peaks(x=self.spectral_energies,    y=self.spectral_amplitudes, approx_locations=location_guess , approx_heights = height_guess , function = method_string, width_guess = width_guess)

            (best_centers, best_heights, best_widths) = fit_params

            self.fit_storage[method_string] = fit

            if fit["cost"] < best_err_found:
                self.best_spectral_fitting_method = method_string
                best_err_found = fit["cost"]

            print("{} profile fit to spectra with cost {}".format(method_string, fit["cost"]))

            print("sqrt-diagonal valus ")

            peak_heights = best_heights


            self.method_to_fit_params[method_string] = (best_centers, peak_heights, best_widths)

            #determine the anharmonic parameters
            vibrational_numbers = np.array(range(0,len(best_centers)))
            total_energies = np.array(best_centers)
            gaps = np.gradient(total_energies)

            x = vibrational_numbers + .5
            y = total_energies
            fitted_poly_vec, residuals, rank, singular_values, rcond  = np.polyfit(x, y, 2, full=True)

            p = np.poly1d(fitted_poly_vec)
            self.method_to_anharmonic[method_string] = fitted_poly_vec

            #print info about anharmonic fit
            self.plot_anharmoic_spacing_fit(method_string, plot=False)


        # print("\n\nBEST METHOD WAS: %s" % self.best_spectral_fitting_method)

    def fit_non_condon_model(self, c_order, method_string_list = methods_list):
        best_cost_found = np.inf
        self.order = c_order
        for method_string in method_string_list:


            print("\n---------{}---------\n".format(method_string.upper()))

            (best_centers, peak_heights, best_widths) = self.method_to_fit_params[method_string]
            spectra_func = methods_to_function[method_string]

            peak_heights = peak_heights
            tuple_key = (method_string, tuple(c_order))
            if tuple_key in self.methodOrderTuple_to_HO_params:
                HO_params, condon_S, fit_cost, relative_errors_for_CI = self.methodOrderTuple_to_HO_params[tuple_key]
            else:
                HO_params, condon_S, fit_cost, relative_errors_for_CI = best_fit_transition_parameters_clist(peak_heights, c_order)
                self.methodOrderTuple_to_HO_params[tuple_key] = (HO_params, condon_S, fit_cost, relative_errors_for_CI)

            (best_S, best_c) = HO_params

            if fit_cost < best_cost_found:
                best_method = method_string
                best_cost_found = fit_cost

            print("HO/c model fit Found for {}!  S={}(1 +/- {}), mu={} * x^ {} * (1 +/- {}) @ 95% Confidence".format(method_string,  best_S, relative_errors_for_CI[0], best_c, c_order, relative_errors_for_CI[1:]))

            print("Naiive S_Condon={}".format(condon_S))

            #print info about anharmonic fit
            self.plot_peak_height_fit(method_string, c_order, plot=False)

        # print("\n\n\nBEST METHOD WAS: %s" % best_method)
        return best_cost_found
    def explore_non_condon_orders(self, max_order, method_str = None, order_lists_to_calculate = None):
        max_number_of_orders = int(math.floor((self.number_peaks - 1) / 2))

        if order_lists_to_calculate is None:
            order_lists_to_calculate = []
            possible_order_list = range(1, max_order + 1)
            for total_orders in range(1, max_number_of_orders+1):
                new_order_lists = ordered_permutations(possible_order_list, total_orders)
                order_lists_to_calculate += new_order_lists

        if method_str is None:
            method_str = self.best_spectral_fitting_method
        best_cost = np.inf
        bost_order = None
        cost_report = []

        for order_list in order_lists_to_calculate:
            print("{} order non-condon perturbation calculation".format(order_list))
            new_cost = self.fit_non_condon_model(order_list, method_string_list = [method_str])
            heapq.heappush(cost_report, (new_cost, order_list))
            if new_cost < best_cost:
                best_cost = new_cost
                best_order = order_list

        print("{} order non-condon perturbation fit the spectra best!".format(best_order))
        self.best_order = best_order
        self.cost_report = []
        print("cost report:")
        for i in range(len(cost_report)):
            cost = heapq.heappop(cost_report)
            self.cost_report.append(cost)
            print("{}".format(cost))

    def estimated_spectra(self, method_str):
        spectra_func = methods_to_function[method_str]
        y_guess = spectra_func(self.spectral_energies, *self.method_to_fit_params[method_str])

        return y_guess

    def plot_individual_peaks(self, method_str):
        spectra_func = methods_to_function[method_str]
        centers, heights, widths = self.method_to_fit_params[method_str]
        plt.title(method_str + " peaks")
        n = len(centers)
        for i, c in enumerate(centers):
            h = heights[i]
            if method_str == "voigt":
                w = [widths[i], widths[i + n]]
            else:
                w = [widths[i]]

            single_peak = spectra_func(self.spectral_energies , [c], [h], w)
            integral = scipy.integrate.simps(single_peak, x = self.spectral_energies)
            # print("integral of {} peak {} div. by h: {}".format(method_str, i, integral/h))
            plt.plot(self.spectral_energies, single_peak, label=i)
        plt.legend()

    def plot_fitted_heights(self):
        for method_str in methods_list:
            centers, heights, widths = self.method_to_fit_params[method_str]
            n = len(centers)
            plt.plot(centers, heights, "*-", label=method_str)
        plt.legend()

    def plot_spectral_fit(self, method_str_list = None):
        plt.plot(self.spectral_energies, self.spectral_amplitudes, label = "Spectra")
        if method_str_list == None:
            for m_str in methods_list:
                plt.plot(self.spectral_energies, self.estimated_spectra(m_str), label = "{} fit".format( m_str[0].upper() + m_str[1:]  ))
        else:
            for m_str in method_str_list:
                plt.plot(self.spectral_energies, self.estimated_spectra(m_str), label = "{} fit".format(m_str[0].upper() + m_str[1:] ))
    def plot_peak_height_fit(self,  method_str, c_order, plot_guess = False, plot=True):
        c_order = tuple(c_order)
        try:
            HO_params, s_condon, err, CI = self.methodOrderTuple_to_HO_params[(method_str, c_order)]
        except:
            print("plot_peak_height_fit {} not calculated yet!".format((method_str, c_order)))
            return None
        fit_params = self.method_to_fit_params[method_str]

        (best_S, best_c) = HO_params
        (best_centers, best_heights, best_widths) = fit_params

        indeces = list(range(len(best_centers)-1))
        best_fit_HO_peaks = predicted_peak_intensities_clist(indeces, best_S, best_c, c_order)

        condon_S = fit_condon(best_heights)

        best_fit_condon_peaks = predicted_peak_intensities_clist(indeces, condon_S, [0.0], [1])
        # print("best_fit_condon_peaks", best_fit_condon_peaks)

        actual_peaks = best_heights/best_heights[0]
        r2_condon = r_squared(data = actual_peaks, prediction = best_fit_condon_peaks)
        r2_model = r_squared(data = actual_peaks, prediction = best_fit_HO_peaks)
        improvement_factor = r2_model / r2_condon
        # print("peak height goddness of fit: \nr^2_condon = {} \t r^2_model = {} \t improvement_factor = {}".format(r2_condon, r2_model, improvement_factor))
        if plot:
            plt.title(method_str)
            plt.plot(best_centers, best_fit_HO_peaks, "x-", label="{} model".format(list(self.order)))
            plt.plot(best_centers, best_fit_condon_peaks, ".-", label="Condon")
            plt.plot(best_centers, actual_peaks, "*-", label="actual")
            if plot_guess:
                guess_e = self.approx_peak_energies
                guess_h = np.array(self.approx_peak_heights) / self.approx_peak_heights[0]
                plt.plot(guess_e, guess_h, "*-", label="guess")
            plt.legend(loc=0)
    def plot_peak_height_ratios_fit(self, method_str, c_order):
            c_order = tuple(c_order)
            try:
                HO_params, condon_S, fit_cost, CI = self.methodOrderTuple_to_HO_params[(method_str, c_order)]
            except:
                print("plot_peak_height_ratios_fit {} not calculated yet!".format((method_str, c_order)))
                return None

            fit_params = self.method_to_fit_params[method_str]

            (best_S, best_c) = HO_params
            (best_centers, best_heights, best_widths) = fit_params

            indeces = list(range(len(best_centers)-1))

            best_fit_HO_ratios = predicted_peak_intensities_clist(indeces, best_S, best_c, c_order, ratios=True)

            condon_S = fit_condon(best_heights)

            best_fit_condon_ratios = predicted_peak_intensities_clist(indeces, condon_S, [0.0],  [1], ratios=True)

            actual_peaks = best_heights/best_heights[0]
            actual_ratios = []
            for i in range(1, len(actual_peaks)):
                actual_ratios.append(actual_peaks[i] / actual_peaks[i - 1])


            plt.title("Model Fit of Peak Height Ratios")
            plt.plot(indeces, best_fit_HO_ratios, "x-", label="{} model".format(list(self.order)))
            plt.plot(indeces, best_fit_condon_ratios, ".-", label="Condon")
            plt.plot(indeces, actual_ratios, "*-", label="Actual")
            plt.legend(loc=0)
            plt.xlabel("kth peak")
            plt.ylabel(r"$H_{%r}(0,k)$" % self.order)
            r2_condon = r_squared(data = actual_ratios, prediction = best_fit_condon_ratios)
            r2_model = r_squared(data = actual_ratios, prediction = best_fit_HO_ratios)
            improvement_factor = r2_model / r2_condon
            print("peak ratio goodness of fit:\nr^2_condon = {} \t r^2_model = {} \t improvement_factor = {}".format(r2_condon, r2_model, improvement_factor))
    def plot_transition_dipole(self, x_values, c_order, method_str, plot_potentials = True, label_prefix = ""):

            c_order = tuple(c_order)
            c_orders = c_order
            try:
                HO_params, condon_S, fit_cost, CI = self.methodOrderTuple_to_HO_params[(method_str, c_order)]
            except:
                print("plot_transition_dipole {} not calculated yet!".format((method_str, c_order)))
                return None
            fit_params = self.method_to_fit_params[method_str]

            (best_S, best_c) = HO_params
            (best_centers, best_heights, best_widths) = fit_params

            dx = math.sqrt(2 * best_S)

            indeces = list(range(len(best_centers)-1))

            best_fit_HO_ratios = predicted_peak_intensities_clist(indeces, best_S, best_c, c_order, ratios=True)

            condon_S = fit_condon(best_heights)

            best_fit_condon_ratios = predicted_peak_intensities_clist(indeces, condon_S, [0.0], [1], ratios=True)

            actual_peaks = best_heights/best_heights[0]
            actual_ratios = []
            for i in range(1, len(actual_peaks)):
                actual_ratios.append(actual_peaks[i] / actual_peaks[i - 1])



            plt.title("Approximate Transition Dipole")
            y_values = transition_dipole_functional_form(x_values, best_c, c_orders)

            x_p = x_values + dx
            plt.plot(x_values, y_values, label=label_prefix + r" $\mu$")
            vlines_y = np.min(np.abs(y_values)), np.max(np.abs(y_values))
            plt.plot([dx, dx], vlines_y, label = label_prefix + r" $\langle x_e \rangle$")
            # plt.plot([0.0, 0.0], vlines_y, label = label_prefix + r" $\langle x_g \rangle$")
            plt.ylabel(r"$\mu(x) / \mu_0$")
            plt.xlabel(r"$x / x_0$")

    def plot_anharmoic_spacing_fit(self, method_str, plot=True):
        fit_params = self.method_to_fit_params[method_str]
        (best_centers, best_heights, best_widths) = fit_params

        vibrational_numbers = np.array(range(0,len(best_centers)))
        total_energies = np.array(best_centers)
        gaps = np.gradient(total_energies)
        # print("vib energy gaps", gaps)
        fitted_poly_vec = self.method_to_anharmonic[method_str]
        e_0 = fitted_poly_vec[2]
        omega = fitted_poly_vec[1]
        chi_omega = -fitted_poly_vec[0]

        dissociation_energy = omega**2 / (4 * chi_omega)
        max_bound_n = int(math.floor((2 * dissociation_energy - omega)/omega))
        x = vibrational_numbers + .5
        p = np.poly1d(fitted_poly_vec)

        print("anharmonic fit", fitted_poly_vec)
        print("chi", chi_omega / omega)
        print("max bound n={}".format(max_bound_n))
        predicted_energies = p(x)
        r2 = r_squared(data = total_energies, prediction = predicted_energies)
        print("r^2={}".format(r2))
        if plot:
            plt.title(method_str)
            plt.plot(vibrational_numbers, total_energies, "*", label="actual")
            plt.plot(vibrational_numbers, predicted_energies, "x", label="fit")
            plt.legend(loc=0)

    def fit_and_observe_spectra(self):
        print("PREPARING ANALYSIS FOR SPECTRA:\n")
        self.compare_spectral_fitting_methods()

        for method_str in methods_list:
            plt.figure()
            self.plot_spectral_fit([method_str])
            plt.legend(loc=0)
            plt.figure()
            self.plot_individual_peaks(method_str)

        plt.figure()
        self.plot_fitted_heights()


    def fit_and_observe_non_condon_models(self, order=None, plot_ratio = True, plot_spectra = True, my_methods_list = None):
        if order == None:
            order = self.best_order
        print("PREPARING ANALYSIS FOR A {}-ORDER CORRECTION TO TRANSITION DIPOLE:\n".format(order))

        if my_methods_list is None:
            my_methods_list = methods_list

        plt.figure()
        self.fit_non_condon_model(order, method_string_list = my_methods_list)

        for method_str in my_methods_list:
            if plot_spectra:
                plt.figure()
                self.plot_peak_height_fit(method_str, order)
                plt.legend(loc=0)
            if plot_ratio:
                plt.figure()
                self.plot_peak_height_ratios_fit(method_str, order)
                plt.legend(loc=0)



def table_generator(order, spectra_fit_1, label_1, spectra_fit_2, label_2, method_str="voigt"):
    tuple_key = (method_str, tuple(order))
    (HO_params_1, condon_S_1, fit_cost_1, CI) = spectra_fit_1.methodOrderTuple_to_HO_params[tuple_key]
    (HO_params_2, condon_S_2, fit_cost_2, CI) = spectra_fit_2.methodOrderTuple_to_HO_params[tuple_key]
    (best_S_1, best_c_1) = HO_params_1
    (best_S_2, best_c_2) = HO_params_2

    fit_params_1 = spectra_fit_1.method_to_fit_params[method_str]
    (best_centers_1, best_heights_1, best_widths_1) = fit_params_1
    actual_ratios = []
    for i in range(1, len(best_heights_1)):
        actual_ratios.append(best_heights_1[i] / best_heights_1[i - 1])
    predicted_ratios = predicted_peak_intensities_clist(list(range(len(actual_ratios))), best_S_1, best_c_1, order, ratios=True)
    r2_1 = r_squared(actual_ratios, predicted_ratios)


    fit_params_2 = spectra_fit_2.method_to_fit_params[method_str]
    (best_centers_2, best_heights_2, best_widths_2) = fit_params_2
    actual_ratios = []
    for i in range(1, len(best_heights_2)):
        actual_ratios.append(best_heights_2[i] / best_heights_2[i - 1])
    predicted_ratios = predicted_peak_intensities_clist(list(range(len(actual_ratios))), best_S_2, best_c_2, order, ratios=True)
    r2_2 = r_squared(actual_ratios, predicted_ratios)

    print("  & {}  &  {} \\\\".format(label_1, label_2))
    print("$S_{{ {} }}$ & {:.3f}  &  {:.3f} \\\\".format(tuple(order), best_S_1, best_S_2))
    for i, o in enumerate(order):
        print("$c_{{ {} }}$ & {:.2e}  &  {:.2e} \\\\".format(o, best_c_1[i], best_c_2[i]))
    print("$r^2_{{ {} }}$ & {:.3f}  &  {:.3f} \\\\".format(tuple(order), r2_1, r2_2))

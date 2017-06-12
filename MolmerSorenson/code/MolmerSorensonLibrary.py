import numpy as np
import math
import cmath
import scipy
import scipy.linalg
import scipy.integrate
import sys
import tqdm
import warnings


h = 1.0
hbar = h / (2.0 * np.pi)

ZERO_TOLERANCE = 10**-4


MAX_VIBRATIONAL_STATES = 75
MAX_FREQ_TIME_SCALE_DT = 18
STARTING_NUMBER_VIBRATIONAL_STATES = 12

NUMBER_ELECTRONIC_STATES = 4
class VibrationalStateOverFlowException(Exception):
    def __init__(self):
        pass


class MolmerSorenson():

    def __init__(self,
                number_vibrational_states,
                energy_vibrational,
                eta_values_list,
                transition_dipole_c_values,
                oscillator_mass = 1.0,
                electronic_energies = [0.0, 0.0, 0.0, 0.0]):
        self.number_vibrational_states = number_vibrational_states
        self.energy_vibrational = energy_vibrational

        self.oscillator_mass = oscillator_mass
        self.electronic_energies = electronic_energies

        vibrational_frequency = energy_vibrational / h
        self.nu = energy_vibrational / hbar
        self.one_vibrational_time = 1.0 / vibrational_frequency

        self.eta_values = eta_values_list
        self.c_values = transition_dipole_c_values

        self.propagated_results = None

    ##VIBRATIONAL HELPER FUNCTIONS
    def vibrational_subspace_zero_operator(self):
        return np.zeros((self.number_vibrational_states, self.number_vibrational_states))

    def vibrational_subspace_identity_operator(self):
        output = self.vibrational_subspace_zero_operator()
        np.fill_diagonal(output, 1.0)
        return output

    def vibrational_subspace_creation_operator(self):
        vibrational_subspace = self.vibrational_subspace_zero_operator()
        for vibrational_index in range(self.number_vibrational_states - 1):
            vibrational_subspace[vibrational_index, vibrational_index + 1] = math.sqrt(vibrational_index + 1)
        return vibrational_subspace.T


    def vibrational_subspace_dimensionless_position_operator(self):
        creation = self.vibrational_subspace_creation_operator()
        annihilation = np.conj(creation.T)
        position = (creation + annihilation)
        return position

    # def vibrational_subspace_position_operator(self):
    #     warnings.warn("MAY BE OFF BY 2pi^n!!!")
    #     dimensionless_position = self.vibrational_subspace_dimensionless_position_operator()
    #     return dimensionless_position * math.sqrt(hbar / (1.0 * self.oscillator_mass * (self.energy_vibrational / hbar)) )


    def vibrational_subspace_hamiltonian_operator(self):
        vibrational_subspace = self.vibrational_subspace_zero_operator()
        for vibrational_index in range(self.number_vibrational_states):
            vibrational_subspace[vibrational_index, vibrational_index ] = self.energy_vibrational * (vibrational_index + .5)
        return vibrational_subspace

    def vibrational_subspace_laser_recoil_operator(self, eta_value):
        exp_arg = eta_value * self.vibrational_subspace_dimensionless_position_operator()
        return scipy.linalg.expm(1.0j * exp_arg)

    def vibrational_subspace_position_to_nth_power_operator(self, n_power, dimensionless=False):
        if n_power == 0:
            return self.vibrational_subspace_identity_operator()
        assert(n_power >=1)
        if dimensionless:
            x = self.vibrational_subspace_dimensionless_position_operator()
        else:
            x = self.vibrational_subspace_position_operator()
        output = x
        for i in range(n_power-1):
            output = output.dot(x)
        return output


    def vibrational_transition_polynomial_operator(self, c_list):
        output = self.vibrational_subspace_zero_operator()
        for poly_order, poly_coefficient in enumerate(c_list):
            x_nthPower = poly_coefficient * self.vibrational_subspace_position_to_nth_power_operator(poly_order, dimensionless=True)
            output =  output + x_nthPower
        return output

    ##VIBRATIONAL HELPER FUNCTIONS
    def blank_wavefunction(self):
        return np.zeros((self.number_vibrational_states * NUMBER_ELECTRONIC_STATES), dtype=np.complex)

    def test_wf(self, normalized = True):
        shape = (self.number_vibrational_states * NUMBER_ELECTRONIC_STATES)
        output = np.random.rand(shape)
        if normalized:
            normalizer = 1.0
        else:
            normalizer = output.T.dot(np.conj(output))
        return output / np.sqrt(normalizer)

    def ground_state_wavefunction(self):
        output = self.blank_wavefunction()
        output[0] = 1.0
        return output

    def access_wavefunction_entry(self, wf_vector, electronic_index, vibrational_index):
        assert(vibrational_index < self.number_vibrational_states)
        index = electronic_index * self.number_vibrational_states + vibrational_index
        return  wf_vector[index]

    def coherent_vibrational_state_ground_electronic(self, alpha_value):
        creation_operator = self.vibrational_subspace_creation_operator()
        coherent_state_operator = np.exp(-np.abs(alpha_value)**2 / 2.0) * scipy.linalg.expm(alpha_value * creation_operator)
        vibrational_subspace_wf = np.zeros(self.number_vibrational_states, dtype=np.complex)
        vibrational_subspace_wf[0] = 1.0
        coherent_vibrational_subspace = coherent_state_operator.dot(vibrational_subspace_wf)
        output = self.blank_wavefunction()
        for i, amp in enumerate(coherent_vibrational_subspace):
            output[i] = amp
        return output

    ##FULL OPERATOR HELPER FUNCTIONS:
    def total_zero_operator(self):
        return np.zeros((self.number_vibrational_states * NUMBER_ELECTRONIC_STATES, self.number_vibrational_states * NUMBER_ELECTRONIC_STATES), dtype=np.complex)

    def place_vibrational_subspace_into_electronic_indeces(self, electronic_operator, vibrational_subspace,electronic_index1, electronic_index2):
        i_0_start = electronic_index1 * self.number_vibrational_states
        i_0_end = i_0_start + self.number_vibrational_states
        i_1_start = electronic_index2 * self.number_vibrational_states
        i_1_end = i_1_start + self.number_vibrational_states
        electronic_operator[i_0_start:i_0_end, i_1_start:i_1_end] = vibrational_subspace

    def total_time_independent_hamiltonian(self):
        output = self.total_zero_operator()
        vib_ham = self.vibrational_subspace_hamiltonian_operator()
        for electronic_index in range(NUMBER_ELECTRONIC_STATES):
            starting_index = electronic_index * self.number_vibrational_states
            ending_index = starting_index + self.number_vibrational_states
            output[starting_index:ending_index, starting_index:ending_index] = vib_ham + self.electronic_energies[electronic_index]
        return output

    def total_time_independent_transition_dipole_helper(self,
                                                 eta_value,
                                                 electronic_index_pair_list):
        output = self.total_zero_operator()
        recoil_operator = self.vibrational_subspace_laser_recoil_operator(eta_value)

        #create the vibrational transition dipole
        vibrational_part_transition_dipole = self.vibrational_transition_polynomial_operator(self.c_values)

        total_vibrational_off_diagonal = vibrational_part_transition_dipole.dot(recoil_operator)
        #place the vibrational part into the total output operator
        for indexpair in electronic_index_pair_list:
            index_0, index_1 = indexpair
            self.place_vibrational_subspace_into_electronic_indeces(output,
                                                              total_vibrational_off_diagonal,
                                                              index_0, index_1)
        return output

    def total_time_independent_transition_dipole_1_excitation(self):
        return self.total_time_independent_transition_dipole_helper(
                                           self.eta_values[0],
                                          [(1,0), (3,2)])
    def total_time_independent_transition_dipole_2_excitation(self):
        return self.total_time_independent_transition_dipole_helper(
                                           self.eta_values[1],
                                          [(2,0), (3,1)])


    def IR_transition_dipole():
        output = self.total_zero_operator()
        creation = self.vibrational_subspace_creation_operator()
        destruction = np.conj(creation.T)
        ir_dipole = creation + destruction
        for electronic_i in range(NUMBER_ELECTRONIC_STATES):
            self.place_vibrational_subspace_into_electronic_indeces(output,
                                                              ir_dipole,
                                                              electronic_i, electronic_i)
        return output

    ## TIME DEPENDENT HELPER FUNCTIONS
    def ode_diagonal_matrix(self):
        return -1.0j * self.total_time_independent_hamiltonian() / hbar

    def propagate(self, laser_energy_list_of_lists,
                  rabi_energy_list_of_lists,
                  initial_state_generator,
                  max_time_vibrational_cycles,
                  use_activation_function=False,
                  time_scale_set = MAX_FREQ_TIME_SCALE_DT):

        while self.number_vibrational_states < MAX_VIBRATIONAL_STATES:

            #define time scales
            max_energy = self.electronic_energies[-1] + self.energy_vibrational * (.5 + self.number_vibrational_states - 1)
            max_frequency = max_energy / h
            dt = 1.0 / (time_scale_set *  max_frequency)
            self.dt = dt
            max_time = max_time_vibrational_cycles * self.one_vibrational_time

            if use_activation_function:
                activation_width = 2.0 * dt
                def activation_function(t):
                    return 1.0/(1.0 + np.exp(-(t/activation_width - 10.0)))
            else:
                def activation_function(t):
                    return 1.0
            ODE_DIAGONAL = self.ode_diagonal_matrix()

            MU_1_EXCITATION = self.total_time_independent_transition_dipole_1_excitation()
            MU_2_EXCITATION = self.total_time_independent_transition_dipole_2_excitation()

            ion_1_energies = laser_energy_list_of_lists[0]
            ion_2_energies = laser_energy_list_of_lists[1]

            ion_1_amplitudes = rabi_energy_list_of_lists[0]
            ion_2_amplitudes = rabi_energy_list_of_lists[1]
            #Added /2 in the amplitudes...?
            def time_function_1_excitation(time):
                output = 0
                for beam_index, beam_energy in enumerate(ion_1_energies):
                    amp = ion_1_amplitudes[beam_index] / 2.0
                    f = beam_energy / hbar
                    output += amp*np.exp(-1.0j * f * time)
                return output
            def time_function_2_excitation(time):
                output = 0
                for beam_index, beam_energy in enumerate(ion_2_energies):
                    amp = ion_2_amplitudes[beam_index] / 2.0
                    f = beam_energy / hbar
                    output += amp*np.exp(-1.0j * f * time)
                return output


            def ODE_integrable_function(time, wf_coefficient_vector):
                mu_1_perturber_excitation = MU_1_EXCITATION * time_function_1_excitation(time)
                mu_2_perturber_excitation = MU_2_EXCITATION * time_function_2_excitation(time)

                mu_1_total = mu_1_perturber_excitation + np.conj(mu_1_perturber_excitation.T)
                mu_2_total = mu_2_perturber_excitation + np.conj(mu_2_perturber_excitation.T)
                ODE_OFF_DIAGONAL = -1.0j *activation_function(time) * (mu_1_total + mu_2_total) / hbar
                ODE_TOTAL_MATRIX = ODE_OFF_DIAGONAL  + ODE_DIAGONAL
                return np.dot(ODE_TOTAL_MATRIX, wf_coefficient_vector)

            #define the starting wavefuntion
            initial_conditions = initial_state_generator()
            #create ode solver
            current_time = 0.0
            ode_solver = scipy.integrate.complex_ode(ODE_integrable_function)
            ode_solver.set_initial_value(initial_conditions, current_time)

            #Run it
            results = [initial_conditions]
            time_values = [current_time]
            n_time = int(math.ceil(max_time / dt))
            try:  #this block catches an overflow into the highest vibrational state
                for time_index in tqdm.tqdm(range(n_time)):
                    #update time, perform solution
                    current_time = ode_solver.t+dt
                    new_result = ode_solver.integrate(current_time)
                    results.append(new_result)
                    #make sure solver was successful
                    if not ode_solver.successful():
                        raise Exception("ODE Solve Failed!")
                    #make sure that there hasn't been substantial leakage to the highest excited states

                    time_values.append(current_time)
                    re_start_calculation = False

                    for electronic_index in range(NUMBER_ELECTRONIC_STATES):
                        max_vibrational_amp = self.access_wavefunction_entry(new_result, electronic_index, self.number_vibrational_states-1)
                        p = np.abs(max_vibrational_amp)**2

                        if p >= ZERO_TOLERANCE:
                            self.number_vibrational_states+=1
                            print("\nIncreasing Number of vibrational states to %i " % self.number_vibrational_states)
                            print("Time reached:" + str(current_time))
                            print("electronic quantum number" + str(electronic_index))
                            raise VibrationalStateOverFlowException()

            except VibrationalStateOverFlowException:
                #Move on and re-start the calculation
                continue

            #Finish calculating
            results = np.array(results)
            time_values = np.array(time_values)

            self.propagated_results = results
            self.time_values = time_values
            return time_values, results
        raise Exception("NEEDED TOO MANY VIBRATIONAL STATES!  RE-RUN WITH DIFFERENT PARAMETERS!")

    ## POST-PROPAGATION CALCULATIONS

    def reduced_electronic_density_matrix(self):
        if self.propagated_results is None:
            raise Exception("No reusults generated yet!")
        results = self.propagated_results
        output = np.zeros((results.shape[0], NUMBER_ELECTRONIC_STATES, NUMBER_ELECTRONIC_STATES), dtype=np.complex)
        for electronic_index_1 in range(NUMBER_ELECTRONIC_STATES):
            for electronic_index_2 in range(NUMBER_ELECTRONIC_STATES):
                new_entry = 0.0
                for vibrational_index in range(self.number_vibrational_states):
                    total_index_1 = self.number_vibrational_states*electronic_index_1 + vibrational_index
                    total_index_2 = self.number_vibrational_states*electronic_index_2 + vibrational_index
                    new_entry += results[:, total_index_1] * np.conj(results[:, total_index_2])
                output[:, electronic_index_1, electronic_index_2] = new_entry
        return output

    def effective_rabi_energy(self, eta, laser_detuning, laser_rabi_energy):
        return -(eta * laser_rabi_energy)**2 / (2  * (self.energy_vibrational - laser_detuning))
    def expected_unitary_dynamics(self,
                                expected_rabi_energy,
                                initial_density_matrix,
                                time_values):
        rho_0 = initial_density_matrix

        #THis is difficult for me but the argument of this funciton in the
        #original paper by Molmer and Sorenson says the argument should be:
        #expected_rabi_energy * time_values / ( 2.0 * hbar)
        #but I seem to need it to be what it is below.
        #there are some odd conventions of 2 in the pauli matrices so I'm
        #guessing it's from that, but I'd be more comfortable if I could
        #precisely chase down the issue....
        cos_func = np.cos(expected_rabi_energy * time_values / ( hbar))
        sin_func = 1.0j * np.sin(expected_rabi_energy * time_values / ( hbar))

        time_evolution_operator = np.zeros((time_values.shape[0], NUMBER_ELECTRONIC_STATES, NUMBER_ELECTRONIC_STATES), dtype = np.complex)

        for electronic_i in range(NUMBER_ELECTRONIC_STATES):
            time_evolution_operator[:, electronic_i, electronic_i] = cos_func

        time_evolution_operator[:, 0, 3] = sin_func
        time_evolution_operator[:, 3, 0] = sin_func
        time_evolution_operator[:, 1, 2] = -sin_func
        time_evolution_operator[:, 2, 1] = -sin_func

        output = np.zeros((time_values.shape[0], NUMBER_ELECTRONIC_STATES, NUMBER_ELECTRONIC_STATES), dtype = np.complex)

        for time_i, time in enumerate(time_values):
            U = time_evolution_operator[time_i]
            U_dagger = np.conj(U)
            np.matmul(rho_0, U_dagger)
            output[time_i,:,:] = np.dot(U, np.dot(rho_0, U_dagger))

        return output

    def trace(self, matrix_in):
        out = 0.0
        for i in range(matrix_in.shape[0]):
            out += matrix_in[i,i]
        return out

    def moving_average(self, a, window_size) :
        ret = np.cumsum(a, dtype=float)
        ret[window_size:] = ret[window_size:] - ret[:-window_size]
        return ret[window_size - 1:] / window_size

    def fidelity(self, density_matrix_series1, density_matrix_series2):
        assert(density_matrix_series1.shape == density_matrix_series2.shape)

        fidelity_series = []
        for i in range(density_matrix_series2.shape[0]):
            rho_1 = density_matrix_series1[i]
            rho_2 = density_matrix_series2[i]

            rho_product = np.dot(rho_1, rho_2)

            rho_1_det = np.linalg.det(rho_1)
            rho_2_det = np.linalg.det(rho_2)

            new_fidelity = math.sqrt(self.trace(rho_product) + 2 * math.sqrt(rho_1_det * rho_2_det))
            fidelity_series.append(new_fidelity)
        return np.array(fidelity_series)

    def reduced_vibrational_density_matrix(self):
        if self.propagated_results is None:
            raise Exception("No results generated yet!")
        results = self.propagated_results
        output = np.zeros((results.shape[0], self.number_vibrational_states, self.number_vibrational_states), dtype=np.complex)
        for vibrational_index_1 in range(self.number_vibrational_states):
            for vibrational_index_2 in range(self.number_vibrational_states):
                new_entry = 0.0
                for electronic_index in range(NUMBER_ELECTRONIC_STATES):
                    total_index_1 = self.number_vibrational_states*electronic_index + vibrational_index_1
                    total_index_2 = self.number_vibrational_states*electronic_index + vibrational_index_2
                    new_entry += results[:, total_index_1] * np.conj(results[:, total_index_2])
                output[:, vibrational_index_1, vibrational_index_2] = new_entry
        return output

    def average_vibrational_quanta(self):
        if self.propagated_results == None:
            raise Exception("No results generated yet!")
        results = self.propagated_results

        time_output_shape = results.shape[0]
        output = np.zeros(time_output_shape, dtype=np.complex)
        for electronic_index in range(NUMBER_ELECTRONIC_STATES):
            for vibrational_index in range(self.number_vibrational_states):
                total_index = self.number_vibrational_states*electronic_index + vibrational_index
                output += vibrational_index * results[:, total_index] * np.conj(results[:, total_index])
        return output

    def ir_spectrum():
        if self.propagated_results == None:
            raise Exception("No reusults generated yet!")
        results = self.propagated_results

        operator = self.IR_transition_dipole()
        time_trace = []
        for time_index in range(results.shape[0]):
            wf = results[time_index]
            new_amp = np.conj(wf.T).dot(operator.dot(wf))
            time_trace.append(new_amp)
        time_trace = np.array(time_trace)
        frequency_amplitude = np.fft.fftshift(np.fft.fft(time_trace))
        frequency_values = np.fft.fftshift(np.fft.fftfreq(time_trace.shape[0], d = self.dt))

        return frequency_values, frequency_amplitude

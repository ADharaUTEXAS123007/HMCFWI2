#include "../../ext/emojicpp-master/emoji.h"
#include "../../ext/inih/INIReader.h"
#include "omp.h"
#include <utility>

//
// Created by Lars Gebraad on 03/04/18.
// Main sampler for Full Waveform Inversion using
// Hamiltonian Monte Carlo
//
#include "hmc_sampler.h"

using namespace std;

hmc_sampler::hmc_sampler(const char *settings_configuration_file) {

    // Select the stream to output to. We could divert the output to a file this way.
    logging_stream = &cout;

    *logging_stream << endl
                    << "========================================================================================"
                    << endl
                    << "|               Hamiltonian Monte Carlo Full-waveform Inversion sampler                |"
                    << endl
                    << "|                          Copyright Lars Gebraad, ETHZ, 2019                          |"
                    << endl
                    << "========================================================================================"
                    << endl
                    << endl;

    // Load .ini file into object -------------------------------------------------------------------------------------

    *logging_stream << "Loading sampler configuration file: '" << settings_configuration_file << "'." << endl;
    INIReader parsed_configuration(settings_configuration_file);
    if (parsed_configuration.ParseError() < 0) {
        throw invalid_argument("Can't load .ini file");
    }
    *logging_stream << emojicpp::emojize(" :heavy_check_mark: ") << " Done parsing HMC settings." << endl << endl;

    // Create fdWaveModel attribute from settings found in hmc .ini file ----------------------------------------------

    string model_configuration = parsed_configuration.Get("fd-setup", "file");
    model = new fdWaveModel(model_configuration.c_str());
    *logging_stream << emojicpp::emojize(" :heavy_check_mark: ") << " Done parsing FD settings." << endl << endl;

    // Parse settings from .ini object into attributes ----------------------------------------------------------------

    // Diagnostics settings
    verbosity = parsed_configuration.GetInteger("diagnostics", "verbosity");

    // Tuning settings
    dt_hmc = parsed_configuration.GetReal("tuning", "timestep");
    n_proposals = parsed_configuration.GetInteger("tuning", "proposals");
    data_variance = parsed_configuration.GetReal("tuning", "data_variance");
    min_hmc_steps = parsed_configuration.GetInteger("tuning", "min_steps");
    max_hmc_steps = parsed_configuration.GetInteger("tuning", "max_steps");
    try {
        mass_matrix_filename = parsed_configuration.Get("tuning", "mass_matrix_from_file");
        custom_mass = true;
    } catch (const invalid_argument &e) {
        *logging_stream << e.what() << " Defaulting to expected_*_standard_deviation for the mass matrix." << endl
                        << endl;
        expected_vp_std = parsed_configuration.GetReal("tuning", "expected_vp_standard_deviation");
        expected_vs_std = parsed_configuration.GetReal("tuning", "expected_vs_standard_deviation");
        expected_de_std = parsed_configuration.GetReal("tuning", "expected_de_standard_deviation");
        custom_mass = false;
    }

    // Prior settings
    upper_bound_vp = parsed_configuration.GetReal("prior", "upper_bound_vp");
    lower_bound_vp = parsed_configuration.GetReal("prior", "lower_bound_vp");
    upper_bound_vs = parsed_configuration.GetReal("prior", "upper_bound_vs");
    lower_bound_vs = parsed_configuration.GetReal("prior", "lower_bound_vs");
    upper_bound_de = parsed_configuration.GetReal("prior", "upper_bound_de");
    lower_bound_de = parsed_configuration.GetReal("prior", "lower_bound_de");

    // Sanity checks / assertions -------------------------------------------------------------------------------------

    // Check that the priors make sense
    assert(upper_bound_vp > lower_bound_vp and upper_bound_vs > lower_bound_vs and upper_bound_de > lower_bound_de);

    // Check that the time-steps distribution makes sense (equal is allowed; just a single number)
    assert(min_hmc_steps <= max_hmc_steps and min_hmc_steps > 0);

    // Assert that data variance is positive
    assert(data_variance > 0);

    // Assert that integration time-step is positive
    assert(dt_hmc > 0);

    // Assert that the number of proposals is positive
    assert(n_proposals > 0);

    // Computing / setting attributes ---------------------------------------------------------------------------------

    // Initialize parametrization dependent fields
    free_parameters = model->free_parameters;
    number_basis_functions_per_par = model->free_parameters / 3;

    // Calculate u_prior
    real u_prior = -(number_basis_functions_per_par * log(1.0 / (upper_bound_de - lower_bound_de)) +
                     number_basis_functions_per_par * log(1.0 / (upper_bound_vp - lower_bound_vp)) +
                     number_basis_functions_per_par * log(1.0 / (upper_bound_vs - lower_bound_vs)));
    // Mass matrix
    if (custom_mass) {
        set_diagonal_mass_matrix(mass_matrix_filename);
    } else {
        set_diagonal_mass_matrix(1.0 / (expected_vp_std * expected_vp_std), 1.0 / (expected_vs_std * expected_vs_std),
                                 1.0 / (expected_de_std * expected_de_std));
    }

    // Initialize HMC attributes
    current_m = dynamic_vector(free_parameters, 1);
    propagated_m = dynamic_vector(free_parameters, 1);
    current_momentum = dynamic_vector(free_parameters, 1);
    propagated_momentum = dynamic_vector(free_parameters, 1);
    gradient = dynamic_vector(free_parameters, 1);

    // Create random number generators for random
    // filenames, momentum generation and
    // acceptance rules.
    random_device rand_dev;
    generator = mt19937_64(rand_dev()); // The Mersenne Twister

    // Create unique filename suffix based on datetime and rng.
    time_t t = time(nullptr);
    struct tm tm = *localtime(&t);
    string timestamp = "d" + to_string(tm.tm_year + 1900) + "." + to_string(tm.tm_mon + 1) + "." +
                       to_string(tm.tm_mday) + ".t" + to_string(tm.tm_hour) + "." + to_string(tm.tm_min) + "." +
                       to_string(tm.tm_sec);
    const int range_from = 1000;
    const int range_to = 9999;
    uniform_int_distribution<int> random_filename_uniform_distribution(range_from, range_to);
    int random_variable = random_filename_uniform_distribution(generator);

    string suffix = "." + timestamp + ".r" + to_string(random_variable);

    samples_filename = "samples" + suffix + ".txt";
    // rejected_filename = "rejected" + suffix + ".txt";

    // Create distribution for drawing from MVN.
    standard_normal = normal_distribution<real>(0, 1);

    stepsize_distribution = uniform_int_distribution<int>(min_hmc_steps, max_hmc_steps);
}

void hmc_sampler::sample() {
    // Create files for samples
    samples_file = ofstream(samples_filename, ofstream::out);
    // rejected_file = ofstream(rejected_filename, ofstream::out);

    if (!samples_file) {
        if (!samples_file) {
            cerr << "The sample file could not be created." << endl;
        }
        // if (!rejected_file) {
        //     cerr << "The rejected sample file could not be created." << endl;
        // }
        cerr << "Exiting.";
        exit(1);
    }

    // Sample function, propagates model through
    // phase space and accepts based on the
    // Hamiltonian reset_velocity_fields(reset_de,
    // reset_vp, reset_vs); // todo make sure this
    // is replaced by setting the velocity fields
    // to the start sample
    run_model();
    *logging_stream << setw(terminal_output_width) << setfill('-') << left << "Sampling " << endl
                    << "Initial misfit: " << misfit << endl
                    << endl;
    current_m = model->get_model_vector();

    // Set up random number generator with a
    // random seed used in the acceptance step
    const int range_from = 0, range_to = 1;
    uniform_real_distribution<real> acceptance_uniform_distribution(range_from, range_to);

    // Update model
    propagated_m = current_m;

    // Initialize momenta
    current_momentum = propose_momentum();
    propagated_momentum = current_momentum;

    int accepted = 0;
    int rejected = 0;

    samples_file << fixed << setprecision(5);

    real start_H, start_X, start_K, end_H, end_X, end_K;

    for (int i = 0; i < n_proposals; ++i) {
        real time_start_sample = omp_get_wtime();
        // Note; the important thing is that
        // run_model() is always called before
        // using X or dXdm, as in the Hamiltonian
        // or Leapfrog propagation.

        *logging_stream << "Proposal: " << i + 1 << endl;

        // Compute hamiltonian before trajectory
        hamiltonian(current_m, current_momentum);
        start_H = H;
        start_K = K;
        start_X = X;

        propagate_leap_frog();
        model->set_model_vector(propagated_m);
        hamiltonian(propagated_m, propagated_momentum);
        end_H = H;
        end_K = K;
        end_X = X;

        if (verbosity > 1) {
            auto a = model->get_model_vector();
            *logging_stream << "Ending parameters" << endl
                            << "vp: " << a[100] << " vs: " << a[100 + number_basis_functions_per_par]
                            << " de: " << a[100 + number_basis_functions_per_par * 2] << endl
                            << flush;
        }
        if (verbosity > 0) {
            *logging_stream << "Old H: " << start_H << ", old X: " << start_X << ", old K: " << start_K << endl
                            << "New H: " << end_H << ", new X: " << end_X << ", new K: " << end_K << endl;
        } else {
            *logging_stream << "Proposed misfit: " << end_X << endl;
        }
        *logging_stream << "Acceptance probability: " << exp((-end_H + start_H)) << endl;

        if ((end_H < start_H or acceptance_uniform_distribution(generator) < exp((-end_H + start_H))) and
            !isnan(end_H) and !isinf(end_H) and !isnan(-end_H) and !isinf(-end_H)) {
            // If accepted (extra clauses are to
            // prevent numerical instability)
            *logging_stream << "\033[1;32m accepted\033[0m" << endl;
            current_m = propagated_m; // Set new sample
            write_sample(samples_file, end_X, current_m, false);
            accepted++;
        } else {
            *logging_stream << "\033[1;31m rejected\033[0m" << endl;
            // Diagnostics
            {
                // rejected_file << "Rejected sample:" << endl
                //               << propagated_m.transpose() << endl
                //               << "Misfit (X): " << scientific << end_X << fixed << endl
                //               << "Kinetic (K): " << scientific << end_K << fixed << endl
                //               << "Hamiltonian: " << scientific << end_H << fixed << endl;
            }
            // Reset
            propagated_m = current_m;
            model->set_model_vector(current_m);
            run_model();
            write_sample(samples_file, start_X, current_m, false);
            rejected++;
        }
        *logging_stream << "Computation time for sample: " << omp_get_wtime() - time_start_sample << " seconds. "
                        << accepted << " accepted samples. " << rejected << " rejected samples." << endl;
        *logging_stream << endl << flush;
        // Propose new momentum
        current_momentum = propose_momentum();
        propagated_momentum = current_momentum;
    }

    *logging_stream << accepted << " accepted" << endl;
    *logging_stream << rejected << " rejected" << endl;
}

void hmc_sampler::propagate_leap_frog() {
    // Leapfrog propagator. Assumes X and dXdm are
    // correct w.r.t. to starting position. Does
    // update X and dXdm at the end of the
    // trajectory.

    // Half step
    hmc_sampler::dynamic_vector localGradient;
    localGradient = gradient;

    real initial_H = misfit + kinetic_energy(propagated_momentum);

    propagated_momentum -= (dt_hmc / 2.) * localGradient;

    int local_steps = stepsize_distribution(generator);
    *logging_stream << "steps: " << local_steps << endl;

    for (int istep = 0; istep < local_steps; ++istep) {

        propagated_m += dt_hmc * propagated_momentum.cwiseProduct(mass_matrix_diagonal.cwiseInverse());

        if (verbosity > 1) {
            *logging_stream << "Leapfrog step " << istep + 1 << endl
                            << "  | vp: " << propagated_m[100]
                            << " vs: " << propagated_m[100 + number_basis_functions_per_par]
                            << " de: " << propagated_m[100 + number_basis_functions_per_par * 2] << endl
                            << flush;
        }

        correct_momentum();
        model->set_model_vector(propagated_m);
        run_model(); // updates X & dXdm
        localGradient = gradient;

        dynamic_vector delta_momentum = dt_hmc * localGradient;

        real temp_K = kinetic_energy(propagated_momentum - 0.5 * delta_momentum);
        real final_H = misfit + temp_K;
        propagated_momentum -= ((istep == local_steps - 1) ? 0.5 : 1.0) * delta_momentum;
    }
}

hmc_sampler::dynamic_vector hmc_sampler::propose_momentum() { return propose_momentum_vector(); }

hmc_sampler::dynamic_vector hmc_sampler::propose_momentum_vector() {
    hmc_sampler::dynamic_vector p = dynamic_vector(free_parameters, 1);
    for (int i_dim = 0; i_dim < free_parameters; ++i_dim) {
        p(i_dim) = standard_normal(generator);
    }
    // For this logic, see http://eigen.tuxfamily.org/dox/group__TutorialArrayClass.html#title6
    return mass_matrix_diagonal.array().sqrt() * p.array();
}

real hmc_sampler::kinetic_energy(hmc_sampler::dynamic_vector momentum) { return kinetic_energy_vector(momentum); }

real hmc_sampler::kinetic_energy_vector(hmc_sampler::dynamic_vector momentum) {
    return 0.5 * (momentum.array().square() / mass_matrix_diagonal.array()).sum();
}

void hmc_sampler::run_model() {

    // Run the forward model
    model->run_model(verbosity > 2, true);

    misfit = model->misfit / data_variance;

    // Get the model gradient vector and store in object
    gradient = model->get_gradient_vector() / data_variance;

    // Sanity check on the size of the gradient.
    assert(gradient.size() == free_parameters);
}

void hmc_sampler::write_sample(ofstream &outfile, real misfit, hmc_sampler::dynamic_vector m, bool from_accepted_move) {
    outfile << m.transpose() << scientific << " " << misfit << fixed << " " << data_variance << " "
            << from_accepted_move << endl;
}

void hmc_sampler::hamiltonian(hmc_sampler::dynamic_vector m, hmc_sampler::dynamic_vector momentum) {
    X = misfit; //+ u_prior;
    K = kinetic_energy(momentum);
    H = (X + K);
}

void hmc_sampler::set_diagonal_mass_matrix(const string filename) {
    mass_matrix_diagonal = load_last_vector(filename);
    assert(mass_matrix_diagonal.size() == free_parameters);
}

void hmc_sampler::set_diagonal_mass_matrix(real mass_vp, real mass_vs, real mass_de) {

    assert(mass_vp > 0.0 and mass_vs > 0.0 and mass_de > 0.0);

    dynamic_vector p = mass_vp * dynamic_vector::Ones(number_basis_functions_per_par, 1);
    dynamic_vector s = mass_vs * dynamic_vector::Ones(number_basis_functions_per_par, 1);
    dynamic_vector d = mass_de * dynamic_vector::Ones(number_basis_functions_per_par, 1);

    mass_matrix_diagonal = dynamic_vector(free_parameters, 1);

    mass_matrix_diagonal << p, s, d;
}

void hmc_sampler::create_data() {
    // Assumes that true model is loaded
    *logging_stream << setw(terminal_output_width) << setfill('-') << left << "Creating true data " << endl;
    real startTime = real(omp_get_wtime());
    for (int is = 0; is < model->n_shots; ++is) {
        model->forward_simulate(is, true, verbosity > 2);
    }
    model->write_receivers();
    model->write_sources();
    real endTime = real(omp_get_wtime());
    *logging_stream << "Elapsed time: " << endTime - startTime
                    << " seconds.  Multiply by ~2.5 to estimate cost per gradient." << endl
                    << emojicpp::emojize(" :heavy_check_mark: ") << "Created true data." << endl
                    << endl;
}

void hmc_sampler::write_data(const string prefix) {
    *logging_stream << setw(terminal_output_width) << setfill('-') << left << "Writing out data for current model "
                    << endl;
    for (int is = 0; is < model->n_shots; ++is) {
        model->forward_simulate(is, true, verbosity > 2);
    }
    model->write_receivers(prefix);
    *logging_stream << emojicpp::emojize(" :heavy_check_mark: ") << "Wrote out data." << endl << endl;
}

void hmc_sampler::load_data() {
    *logging_stream << setw(terminal_output_width) << setfill('-') << left << "Loading observed data " << endl;

    *logging_stream << "Data files" << endl;
    for (int i_shot = 0; i_shot < model->n_shots; ++i_shot) {
        *logging_stream << "Shot " << i_shot << ":" << endl
                        << "\t" << model->observed_data_folder + "/rtf_ux" + to_string(i_shot) + ".txt" << endl
                        << "\t" << model->observed_data_folder + "/rtf_uz" + to_string(i_shot) + ".txt" << endl;
    }

    model->load_receivers(verbosity > 0);
    *logging_stream << emojicpp::emojize(" :heavy_check_mark: ") << "Read in the data." << endl << endl << flush;
}

void hmc_sampler::add_noise() { add_noise(sqrt(data_variance)); }

void hmc_sampler::add_noise(real data_standard_deviation) {
    
    *logging_stream << "Adding noise with standard deviation to data " << data_standard_deviation << endl;

    for (int i_shot = 0; i_shot < model->n_shots; ++i_shot) {
        for (int i_receiver = 0; i_receiver < model->nr; ++i_receiver) {
            for (int it = 0; it < model->nt; ++it) {
                model->rtf_ux_true[i_shot][i_receiver][it] += data_standard_deviation * standard_normal(generator);
                model->rtf_uz_true[i_shot][i_receiver][it] += data_standard_deviation * standard_normal(generator);
            }
        }
    }
}

void hmc_sampler::load_model(const string de_starting_filename, const string vp_starting_filename,
                             const string vs_starting_filename) {
    *logging_stream << setw(terminal_output_width) << setfill('-') << left << "Loading models into fd object " << endl;
    *logging_stream << "Using the following models: " << endl
                    << "P-wave velocity:                           " << vp_starting_filename << endl
                    << "S-wave velocity:                           " << vs_starting_filename << endl
                    << "Density:                                   " << de_starting_filename << endl;
    model->load_model(de_starting_filename, vp_starting_filename, vs_starting_filename, verbosity > 0);
    *logging_stream << emojicpp::emojize(" :heavy_check_mark: ") << "Loaded models into fd class." << endl << endl;
}

dynamic_vector hmc_sampler::load_last_vector(const string filename, bool strict /* = true */) {
    // Method to load the last row from a file as a vector related to the inverse problem. The line should contain as
    // many entries as relevant for the inverse problem (free_parameters).

    // Say what we are doing
    *logging_stream << setw(terminal_output_width) << setfill('-') << left << "Loading vector from file " << endl;
    *logging_stream << "File:                                   " << filename << endl;

    // Create vector
    hmc_sampler::dynamic_vector m = dynamic_vector(free_parameters, 1);

    // Open file
    ifstream fs;
    fs.open(filename.c_str(), fstream::in);
    if (!fs.is_open()) {
        throw invalid_argument("Could not open file");
    }

    // Get last line from the file
    string last_line = getLastLineInFile(fs);

    // Start at position 0
    size_t pos = 0;

    // Read-in value from file
    string token;

    // File delimiter
    string delimiter = " ";

    // Current place to write to in m
    int vector_index = 0;

    // Temporary place to store floats
    real next_value;

    // Loop over entries between delimiters
    while ((pos = last_line.find(delimiter)) != string::npos) {
        // Find new entry
        token = last_line.substr(0, pos);
        if (!token.empty()) { // Check if not empty, this happens if two delimiters follow each other
            // Compute next value
            next_value = real(stod(token));
            m[vector_index] = next_value;
            vector_index++;
        }
        if (vector_index == m.size()) {
            if (strict) {
                // Our vector is populated, but there is more data in the file! Oh no!
                throw invalid_argument("Too much data on the last line of the file for the vector!");
            } else {
                // There is more in the file, but we don't care (e.g. when loading the last sample)
                break;
            }
        }
        last_line.erase(0, pos + delimiter.length());
    }
    *logging_stream << endl;

    // Check if we populated the entire vector
    if (vector_index < free_parameters) {
        throw invalid_argument(
            "The supplied file did not contain enough values in the last row for a complete vector.");
    }
    *logging_stream << emojicpp::emojize(" :heavy_check_mark: ") << "Read in the vector." << endl << endl;

    return m;
}

void hmc_sampler::load_sample(const string samples_to_load_filename) {
    model->set_model_vector(load_last_vector(samples_to_load_filename, false));
}

bool moveToStartOfLine(ifstream &fs) {
    fs.seekg(-1, ios_base::cur);
    for (long i = fs.tellg(); i > 0; i--) {
        if (fs.peek() == '\n') {
            fs.get();
            return true;
        }
        fs.seekg(i, ios_base::beg);
    }
    return false;
}

string getLastLineInFile(ifstream &fs) {
    // Go to the last character before EOF
    fs.seekg(-1, ios_base::end);
    if (!moveToStartOfLine(fs))
        return "";

    string lastline = "";
    getline(fs, lastline);
    return lastline;
}

void hmc_sampler::correct_momentum() {
    for (int i_parameter = 0; i_parameter < number_basis_functions_per_par; ++i_parameter) {
        // Check if out of lower bound
        if (propagated_m[i_parameter] < lower_bound_vp) {
            // Reflect ...
            propagated_m[i_parameter] = (lower_bound_vp - propagated_m[i_parameter]) + lower_bound_vp;
            // ... and reverse momentum
            propagated_momentum[i_parameter] *= -1;
        } else if (propagated_m[i_parameter] > upper_bound_vp) {
            propagated_m[i_parameter] = upper_bound_vp - (propagated_m[i_parameter] - upper_bound_vp);
            propagated_momentum[i_parameter] *= -1;
        }

        if (propagated_m[i_parameter + number_basis_functions_per_par] < lower_bound_vs) {
            propagated_m[i_parameter + number_basis_functions_per_par] =
                (lower_bound_vs - propagated_m[i_parameter + number_basis_functions_per_par]) + lower_bound_vs;
            propagated_momentum[i_parameter + number_basis_functions_per_par] *= -1;
        } else if (propagated_m[i_parameter + number_basis_functions_per_par] > upper_bound_vs) {
            propagated_m[i_parameter + number_basis_functions_per_par] =
                upper_bound_vs - (propagated_m[i_parameter + number_basis_functions_per_par] - upper_bound_vs);
            propagated_momentum[i_parameter + number_basis_functions_per_par] *= -1;
        }

        if (propagated_m[i_parameter + 2 * number_basis_functions_per_par] < lower_bound_de) {
            propagated_m[i_parameter + 2 * number_basis_functions_per_par] =
                (lower_bound_de - propagated_m[i_parameter + 2 * number_basis_functions_per_par]) + lower_bound_de;
            propagated_momentum[i_parameter + 2 * number_basis_functions_per_par] *= -1;
        } else if (propagated_m[i_parameter + 2 * number_basis_functions_per_par] > upper_bound_de) {
            propagated_m[i_parameter + 2 * number_basis_functions_per_par] =
                upper_bound_de - (propagated_m[i_parameter + 2 * number_basis_functions_per_par] - upper_bound_de);
            propagated_momentum[i_parameter + 2 * number_basis_functions_per_par] *= -1;
        }
    }
}

void hmc_sampler::dump_gradient(string output_filename) {
    ofstream outfile(output_filename);
    outfile << gradient.transpose() << endl;
    outfile.close();
}

void hmc_sampler::print_configuration() {
    // Prints the settings of the sampler

    ostringstream confstr;

    confstr << setw(terminal_output_width) << setfill('-') << left << "Sampler settings " << endl;
    confstr << "Max threads for wavefield simulations: " << omp_get_max_threads() << endl;

    // Tuning
    confstr << setw(terminal_output_width) << setfill('.') << left << "Tuning: " << endl;
    confstr << setw(81) << setfill(' ') << left << "\tLeapfrog integration " << endl;
    confstr << "\tLeapfrog time step:                " << dt_hmc << endl;
    confstr << "\tLeapfrog minimum steps:            " << min_hmc_steps << endl;
    confstr << "\tLeapfrog maximum steps:            " << max_hmc_steps << endl;
    confstr << setw(81) << setfill(' ') << left << "\tMass matrix " << endl;
    confstr << "\tUsing high detail diagonal matrix: " << (custom_mass ? "yes" : "no") << endl;
    if (custom_mass) {
        confstr << "\tMass matrix file:                  " << mass_matrix_filename << endl;
    } else {
        confstr << "\tExpected std. dev. / mass vp:      " << expected_vp_std << " / "
                << 1.0 / (expected_vp_std * expected_vp_std) << endl;
        confstr << "\tExpected std. dev. / mass vs:      " << expected_vs_std << " / "
                << 1.0 / (expected_vs_std * expected_vs_std) << endl;
        confstr << "\tExpected std. dev. / mass de:      " << expected_de_std << " / "
                << 1.0 / (expected_de_std * expected_de_std) << endl;
    }
    confstr << setw(81) << setfill(' ') << left << "\tTempering " << endl;
    confstr << "\tData variance (or temperature):        " << data_variance << endl;

    // Output
    confstr << setw(terminal_output_width) << setfill('.') << left << "Output: " << endl;
    confstr << "\tFree parameters:                   " << free_parameters << endl;
    confstr << "\tProposals:                         " << n_proposals << endl;
    confstr << "\tSamples output file:               " << samples_filename << endl;
    // confstr << "\tRejected output file:              " << rejected_filename << endl;
    confstr << "\tVerbosity level:                   " << verbosity << endl;

    // Prior
    confstr << setw(terminal_output_width) << setfill('.') << left << "Prior: " << endl;
    confstr << "\tP-wave velocity:                   " << lower_bound_vp << " - " << upper_bound_vp << " m/s" << endl;
    confstr << "\tS-wave velocity:                   " << lower_bound_vs << " - " << upper_bound_vs << " m/s" << endl;
    confstr << "\tDensity:                           " << lower_bound_de << " - " << upper_bound_de << " kg/m³" << endl;

    // Simulation
    confstr << setw(terminal_output_width) << setfill('.') << left << "Numerical set-up: " << endl;
    confstr << "\tTemporal samples:                  " << model->nt << endl;
    confstr << "\tAbsorbing at max z (top):          " << (model->free_surface_maxz ? "no" : "yes") << endl;
    confstr << "\tA Spatial samples x:               " << model->nx << endl;
    confstr << "\tB Spatial samples z:               " << model->nz << endl;
    confstr << "\tC Absorbing boundary width:        " << model->np_boundary << endl;
    confstr << "\tD Spacing target - boundary x:     " << model->nx_inner_boundary << endl;
    confstr << "\tE Spacing target - boundary z:     " << model->nz_inner_boundary << endl;
    confstr << "\tF Resulting free nodes x:          " << model->nx_free_parameters << endl;
    confstr << "\tG Resulting free nodes z:          " << model->nz_free_parameters << endl;
    confstr << endl
            << "\t/////////// Free surface //////////////////////" << endl
            << "\t----------------------→A←---------↓------------" << endl
            << "\t|   |                             E       |   |" << endl
            << "\t|   |                             ↑       |   |" << endl
            << "\t|→C←|     ------------→F←------------     |   |" << endl
            << "\t|   |     |                         |     |   |" << endl
            << "\t|   |     |                         ↓     |   |" << endl
            << "\t↓   |     | Free parameters         G     |   |" << endl
            << "\tB   | →D← |                         ↑     |   |" << endl
            << "\t↑   |     |                         | →D← |   |" << endl
            << "\t|   |     ------------------------↓--     |   |" << endl
            << "\t|   |       Fixed parameters      E       |   |" << endl
            << "\t|   |                             ↑       |   |" << endl
            << "\t|   ------------------------------↓--------   |" << endl
            << "\t|           Absorbing boundary    C           |" << endl
            << "\t----------------------------------↑------------" << endl
            << endl;

    confstr << "\tTemporal resolution:               " << model->dt << " s" << endl;
    confstr << "\tSpatial simulation resolution x:   " << model->dx << " m" << endl;
    confstr << "\tSpatial simulation resolution z:   " << model->dz << " m" << endl;
    confstr << "\tAbsorbing boundary factor:         " << model->np_factor << endl;
    confstr << "\tGrid points x per basis function:  " << model->basis_gridpoints_x << endl;
    confstr << "\tGrid points z per basis function:  " << model->basis_gridpoints_z << endl;
    confstr << "\tResulting inversion grid:          " << model->nx_free_parameters / model->basis_gridpoints_x
            << " (x) × " << model->nz_free_parameters / model->basis_gridpoints_z << " (z)" << endl;

    // Sources and receivers
    confstr << setw(terminal_output_width) << setfill('.') << left << "Source-receiver geometry: " << endl;
    confstr << "\tNumber of sources:                 " << model->n_sources << endl;
    confstr << "\tShooting pattern:                  ";
    for (vector<int>::size_type i = 0; i != model->which_source_to_fire_in_which_shot.size(); i++) {
        confstr << "Shot " << i << ": [";
        for (auto i_source : model->which_source_to_fire_in_which_shot[i]) {
            confstr << i_source << " ";
        }
        confstr << "]  ";
    }
    confstr << endl;
    confstr << "\tCycle delay per source:            " << model->delay_cycles_per_shot << endl;
    confstr << "\tPeak frequency:                    " << model->peak_frequency << " Hz" << endl;
    confstr << "\tNumber of receivers:               " << model->nr << endl;

    // Adjoint-related
    confstr << setw(terminal_output_width) << setfill('.') << left << "Sensitivity kernel computation: " << endl;
    confstr << "\tStore wavefield every n-th sample: " << model->snapshot_interval << endl;
    confstr << "\tTotal snapshots for singel shot:   " << model->snapshots << endl;

    // Waveforms
    confstr << setw(terminal_output_width) << setfill('.') << left
            << "Input/output waveforms (a dot represents current folder): " << endl;
    confstr << "\tObserved data folder:              " << model->observed_data_folder << endl;
    confstr << "\tGenerated source wavelet output:   " << model->stf_folder << endl;

    confstr << endl;

    *logging_stream << confstr.str();
}

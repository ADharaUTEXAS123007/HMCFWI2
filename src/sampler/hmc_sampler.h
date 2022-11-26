//
// Created by Lars Gebraad on 03/04/18.
//
#ifndef HMC_FULL_WAVEFORM_INVERSION_SAMPLER_H
#define HMC_FULL_WAVEFORM_INVERSION_SAMPLER_H

#include "../../ext/eigen/Eigen/Dense"
#include "../../ext/eigen/Eigen/Sparse"
#include "../../ext/forward-virieux/src/fdWaveModel.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <time.h>

#define real double

class hmc_sampler {
  public:
    explicit hmc_sampler(const char *settings_configuration_file);

    fdWaveModel *model;

    // Number of basis functions, initialized in constructor
    int free_parameters;
    int number_basis_functions_per_par;

    // Some typedefs for everyone
    typedef Eigen::Matrix<real, Eigen::Dynamic, 1> dynamic_vector;
    typedef Eigen::Triplet<real> triplet;

    int verbosity = 0;

    bool custom_mass;

    std::string mass_matrix_filename;

    int n_proposals = 500000;

    int min_hmc_steps, max_hmc_steps;
    std::uniform_int_distribution<int> stepsize_distribution;

    real dt_hmc;

    real expected_vp_std, expected_vs_std, expected_de_std;

    std::ostream *logging_stream;

    dynamic_vector current_m, propagated_m;

    dynamic_vector current_momentum, propagated_momentum;

    // Householding methods -------------------------------------------------------------------------------------------
    void print_configuration();
    dynamic_vector load_last_vector(std::string filename, bool strict = true);

    // Householding attributes ----------------------------------------------------------------------------------------
    int terminal_output_width = 88;

    // Simulation methods ---------------------------------------------------------------------------------------------
    void create_data();
    void write_data(std::string prefix);
    void load_data();
    void add_noise();
    void add_noise(real data_standard_deviation);
    void load_model(std::string de_starting_filename, std::string vp_starting_filename,
                    std::string vs_starting_filename);
    void load_sample(std::string samples_filename);
    void dump_gradient(std::string output_filename);
    void run_model();

    // Mass matrix and kinetic energy methods -------------------------------------------------------------------------
    void set_diagonal_mass_matrix(const std::string);
    void set_diagonal_mass_matrix(real mass_vp, real mass_vs, real mass_de);

    real kinetic_energy(dynamic_vector momentum); // delegator
    real kinetic_energy_vector(dynamic_vector momentum);
    dynamic_vector propose_momentum(); // delegator
    dynamic_vector propose_momentum_vector();

    // Mass matrix and kinetic energy attributes ----------------------------------------------------------------------
    Eigen::SparseMatrix<real> m_sparseMassMatrix;
    dynamic_vector mass_matrix_diagonal;

    // HMC sampling methods -------------------------------------------------------------------------------------------
    void sample();
    void propagate_leap_frog();
    void correct_momentum();
    void hamiltonian(dynamic_vector m, dynamic_vector momentum);
    void write_sample(std::ofstream &outfile, real misfit, dynamic_vector m, bool from_accepted_move);

    // HMC sampling attributes ----------------------------------------------------------------------------------------
    real H, K, X;
    real misfit;
    dynamic_vector gradient;
    std::ofstream samples_file;
    // std::ofstream rejected_file;

    std::string samples_filename;
    // std::string rejected_filename;

    // Random number generator attributes -----------------------------------------------------------------------------
    std::normal_distribution<real> standard_normal;
    std::mt19937_64 generator;

    // Prior & likelihood attributes ----------------------------------------------------------------------------------
    real upper_bound_vp, lower_bound_vp, upper_bound_vs, lower_bound_vs, upper_bound_de, lower_bound_de;
    real u_prior;
    real data_variance;
};

// Miscellaneous functions

bool moveToStartOfLine(std::ifstream &fs);
std::string getLastLineInFile(std::ifstream &fs);

#endif // HMC_FULL_WAVEFORM_INVERSION_SAMPLER_H

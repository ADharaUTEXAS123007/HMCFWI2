//
// Created by Lars Gebraad on 16/04/19.
//

// Includes
#include <iostream>
#include <iomanip>
#include <fstream>
#include <tgmath.h>
#include <omp.h>
#include "../src/fdWaveModel.h"

int main(int argc, char **argv) {
    std::cout << "Maximum amount of OpenMP threads:" << omp_get_max_threads() << std::endl << std::endl;

    auto configuration_file = argv[1];

    auto *model = new fdWaveModel(configuration_file);

    model->load_model("../tests/test_setup/de_target.txt", "../tests/test_setup/vp_target.txt", "../tests/test_setup/vs_target.txt", true);

    auto startTime = omp_get_wtime();
    int n_tests = 5;
    std::cout << "Running forward simulation " << n_tests << " times." << std::endl;
    auto deterministic_sum = new real_simulation[n_tests];

    for (int i_test = 0; i_test < n_tests; ++i_test) {
        for (int is = 0; is < model->n_shots; ++is) {
            model->forward_simulate(is, false, true);
        }

        deterministic_sum[i_test] = 0;

        for (int i_shot = 0; i_shot < model->n_shots; ++i_shot) {
            for (int i_receiver = 0; i_receiver < model->nr; ++i_receiver) {
                for (int it = 0; it < model->nt; ++it) {
                    deterministic_sum[i_test] += model->rtf_ux[i_shot][i_receiver][it];
                }
            }
        }
    }
    auto endTime = omp_get_wtime();
    std::cout << "Elapsed time for all forward wave simulations: " << endTime - startTime << std::endl;

    model->write_receivers();
    model->write_sources();

    // Check deterministic output
    for (unsigned i = 0; i < n_tests; i++) {
        if (deterministic_sum[i] != deterministic_sum[0]) {
            exit(1);
        }
    }
    std::cout << "All simulations produced the same seismograms. The test succeeded." << std::endl;
    exit(0);
}


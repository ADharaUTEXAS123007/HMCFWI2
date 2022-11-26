//#include <mpi.h>
#include "../ext/inih/INIReader.h"
#include "sampler/hmc_sampler.h"

using namespace std;

int main(int argc, char *argv[]) {
    Eigen::initParallel();

    const char *configuration = "configuration_structural_target_hmc.ini";

    // Create sampler
    hmc_sampler *sampler = new hmc_sampler(configuration);

    // Print configuration
    sampler->print_configuration();

    sampler->set_diagonal_mass_matrix(1.0 / (80.0 * 80.0), 1.0 / (20.0 * 20.0), 1.0 / (45.0 * 45.0));

    // Create true data
    sampler->load_model("target_models/de_target.txt", "target_models/vp_target.txt", "target_models/vs_target.txt");
    sampler->create_data();

    // Load previous sample (the sampler uses the current model as the starting model).
    // Use one of these two statements.
    sampler->load_model("starting_models/de_starting_after_burnin.txt", 
                        "starting_models/vp_starting_after_burnin.txt", 
                        "starting_models/vs_starting_after_burnin.txt");
    // sampler->load_sample("<previous_samples_file.txt>");

    // Load observed data
    sampler->load_data();
    sampler->model->write_receivers();

    // Start sampling
    sampler->sample();

    return 0;
}

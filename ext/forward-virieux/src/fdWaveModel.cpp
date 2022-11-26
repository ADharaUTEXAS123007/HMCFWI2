//
// Created by Lars Gebraad on 25.01.19.
//

#include "fdWaveModel.h"
#include "../../../ext/inih/INIReader.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <stdexcept>

#define PI 3.14159265

fdWaveModel::fdWaveModel(const char *configuration_file_relative_path) {
  // --- Initialization section ---

  // Parse configuration from supplied file
  parse_configuration(configuration_file_relative_path);

  // Allocate all fields dynamically (grouped by type)
  allocate_2d_array(vx, nx, nz);
  allocate_2d_array(vz, nx, nz);
  allocate_2d_array(txx, nx, nz);
  allocate_2d_array(tzz, nx, nz);
  allocate_2d_array(txz, nx, nz);

  allocate_2d_array(lm, nx, nz);
  allocate_2d_array(la, nx, nz);
  allocate_2d_array(mu, nx, nz);
  allocate_2d_array(b_vx, nx, nz);
  allocate_2d_array(b_vz, nx, nz);
  allocate_2d_array(rho, nx, nz);
  allocate_2d_array(vp, nx, nz);
  allocate_2d_array(vs, nx, nz);

  allocate_2d_array(density_l_kernel, nx, nz);
  allocate_2d_array(lambda_kernel, nx, nz);
  allocate_2d_array(mu_kernel, nx, nz);

  allocate_2d_array(vp_kernel, nx, nz);
  allocate_2d_array(vs_kernel, nx, nz);
  allocate_2d_array(density_v_kernel, nx, nz);

  allocate_2d_array(starting_rho, nx, nz);
  allocate_2d_array(starting_vp, nx, nz);
  allocate_2d_array(starting_vs, nx, nz);

  allocate_2d_array(taper, nx, nz);

  allocate_1d_array(t, nt);
  allocate_2d_array(stf, n_sources, nt);
  allocate_3d_array(moment, n_sources, 2, 2);
  allocate_3d_array(rtf_ux, n_shots, nr, nt);
  allocate_3d_array(rtf_uz, n_shots, nr, nt);
  allocate_3d_array(rtf_ux_true, n_shots, nr, nt);
  allocate_3d_array(rtf_uz_true, n_shots, nr, nt);
  allocate_3d_array(a_stf_ux, n_shots, nr, nt);
  allocate_3d_array(a_stf_uz, n_shots, nr, nt);
  allocate_4d_array(accu_vx, n_shots, snapshots, nx, nz);
  allocate_4d_array(accu_vz, n_shots, snapshots, nx, nz);
  allocate_4d_array(accu_txx, n_shots, snapshots, nx, nz);
  allocate_4d_array(accu_tzz, n_shots, snapshots, nx, nz);
  allocate_4d_array(accu_txz, n_shots, snapshots, nx, nz);

  // Place sources/receivers inside the domain if required
  if (add_np_to_receiver_location) {
    for (int ir = 0; ir < nr; ++ir) {
      ix_receivers[ir] += np_boundary;
      iz_receivers[ir] += np_boundary;
    }
  }
  if (add_np_to_source_location) {
    for (int is = 0; is < n_sources; ++is) {
      ix_sources[is] += np_boundary;
      iz_sources[is] += np_boundary;
    }
  }

  // Assign source time function and time axis based on set-up
  for (int i_shot = 0; i_shot < n_shots; ++i_shot) {
    for (int i_source = 0;
         i_source < which_source_to_fire_in_which_shot[i_shot].size();
         ++i_source) {
      for (unsigned int it = 0; it < nt; ++it) {
        t[it] = it * dt;
        auto f = static_cast<real_simulation>(1.0 / alpha);
        auto shiftedTime = static_cast<real_simulation>(
            t[it] - 1.4 / f - delay_cycles_per_shot * i_source / f);
        stf[which_source_to_fire_in_which_shot[i_shot][i_source]][it] =
            real_simulation((1 - 2 * pow(M_PI * f * shiftedTime, 2)) *
                            exp(-pow(M_PI * f * shiftedTime, 2)));
      }
    }
  }

  // Assign moment tensors based on rotation.
  for (int i_source = 0; i_source < n_sources;
       ++i_source) { // todo allow for more complex moment tensors
    moment[i_source][0][0] = static_cast<real_simulation>(
        cos(moment_angles[i_source] * PI / 180.0) * 1e15);
    moment[i_source][0][1] = static_cast<real_simulation>(
        -sin(moment_angles[i_source] * PI / 180.0) * 1e15);
    moment[i_source][1][0] = static_cast<real_simulation>(
        -sin(moment_angles[i_source] * PI / 180.0) * 1e15);
    moment[i_source][1][1] = static_cast<real_simulation>(
        -cos(moment_angles[i_source] * PI / 180.0) * 1e15);
  }

// Set all fields to background value so as to at least initialize.
#pragma omp parallel for collapse(2)
  for (int ix = 0; ix < nx; ++ix) {
    for (int iz = 0; iz < nz; ++iz) {
      vp[ix][iz] = scalar_vp;
      vs[ix][iz] = scalar_vs;
      rho[ix][iz] = scalar_rho;
    }
  }

  // Update Lamé's fields from velocity fields.
  update_from_velocity();

// Initialize Gaussian taper by ...
#pragma omp parallel for collapse(2)
  for (int ix = 0; ix < nx;
       ++ix) { // ... starting with zero taper in every point, ...
    for (int iz = 0; iz < nz; ++iz) {
      taper[ix][iz] = 0.0;
    }
  }
  for (int id = 0; id < np_boundary;
       ++id) { // ... subsequently, move from outside inwards over the np,
               // adding one to every point ...

    int max_z = free_surface_maxz ? nz : nz - id;

#pragma omp parallel for collapse(2)
    for (int ix = id; ix < nx - id; ++ix) {
      for (int iz = id; iz < max_z; ++iz) {
        taper[ix][iz]++;
      }
    }
  }
#pragma omp parallel for collapse(2)
  for (int ix = 0; ix < nx; ++ix) {
    // ... and finally setting the maximum taper value to taper 1
    // using exponential function, decaying outwards.
    for (int iz = 0; iz < nz; ++iz) {
      taper[ix][iz] = static_cast<real_simulation>(
          exp(-pow(np_factor * (np_boundary - taper[ix][iz]), 2)));
    }
  }

  // Check if the amount of snapshots satisfies the other parameters.
  if (floor(double(nt) / snapshot_interval) != snapshots) {
    throw std::length_error(
        "Snapshot interval and size of accumulator don't match!");
  }
}

fdWaveModel::~fdWaveModel() {
  deallocate_2d_array(vx, nx);
  deallocate_2d_array(vz, nx);
  deallocate_2d_array(txx, nx);
  deallocate_2d_array(tzz, nx);
  deallocate_2d_array(txz, nx);

  deallocate_2d_array(lm, nx);
  deallocate_2d_array(la, nx);
  deallocate_2d_array(mu, nx);
  deallocate_2d_array(b_vx, nx);
  deallocate_2d_array(b_vz, nx);
  deallocate_2d_array(rho, nx);
  deallocate_2d_array(vp, nx);
  deallocate_2d_array(vs, nx);

  deallocate_2d_array(density_l_kernel, nx);
  deallocate_2d_array(lambda_kernel, nx);
  deallocate_2d_array(mu_kernel, nx);

  deallocate_2d_array(vp_kernel, nx);
  deallocate_2d_array(vs_kernel, nx);
  deallocate_2d_array(density_v_kernel, nx);

  deallocate_2d_array(starting_rho, nx);
  deallocate_2d_array(starting_vp, nx);
  deallocate_2d_array(starting_vs, nx);

  deallocate_2d_array(taper, nx);

  deallocate_1d_array(t);
  deallocate_2d_array(stf, n_sources);
  deallocate_3d_array(moment, n_sources, 2);
  deallocate_3d_array(rtf_ux, n_shots, nr);
  deallocate_3d_array(rtf_uz, n_shots, nr);
  deallocate_3d_array(rtf_ux_true, n_shots, nr);
  deallocate_3d_array(rtf_uz_true, n_shots, nr);
  deallocate_3d_array(a_stf_ux, n_shots, nr);
  deallocate_3d_array(a_stf_uz, n_shots, nr);
  deallocate_4d_array(accu_vx, n_shots, snapshots, nx);
  deallocate_4d_array(accu_vz, n_shots, snapshots, nx);
  deallocate_4d_array(accu_txx, n_shots, snapshots, nx);
  deallocate_4d_array(accu_tzz, n_shots, snapshots, nx);
  deallocate_4d_array(accu_txz, n_shots, snapshots, nx);
}

void fdWaveModel::parse_configuration(
    const char *configuration_file_relative_path) {
  std::cout << "Loading configuration file: '"
            << configuration_file_relative_path << "'." << std::endl;

  INIReader reader(configuration_file_relative_path);
  if (reader.ParseError() < 0) {
    std::cout << "Can't load .ini file\n";
    exit(1);
  }

  free_surface_maxz = reader.GetBoolean("domain", "free_surface_maxz", false);

  // Domain
  nt = reader.GetInteger("domain", "nt", 1000);
  nx_inner = reader.GetInteger("domain", "nx_inner", 200);
  nz_inner = reader.GetInteger("domain", "nz_inner", 100);
  nx_inner_boundary = reader.GetInteger("domain", "nx_inner_boundary", 10);
  nz_inner_boundary = reader.GetInteger("domain", "nz_inner_boundary", 20);
  dx = reader.GetReal("domain", "dx", 1.249);
  dz = reader.GetReal("domain", "dz", 1.249);
  dt = reader.GetReal("domain", "dt", 0.00025);

  // Boundary
  np_boundary = reader.GetInteger("boundary", "np_boundary", 10);
  np_factor = reader.GetReal("boundary", "np_factor", 0.075);

  if (free_surface_maxz) {
    // In the case that we have a free surface at z = z2
    nx = nx_inner + np_boundary * 2;
    nz = nz_inner + np_boundary;
    nx_free_parameters = nx_inner - nx_inner_boundary * 2;
    nz_free_parameters = nz_inner - nz_inner_boundary * 2;
  } else {
    // In the case that we have only abosrbing boundaries
    nx = nx_inner + np_boundary * 2;
    nz = nz_inner + np_boundary * 2;
    nx_free_parameters = nx_inner - nx_inner_boundary * 2;
    nz_free_parameters = nz_inner - nz_inner_boundary * 2;
  }

  // Default medium
  scalar_rho = reader.GetReal("medium", "scalar_rho", 1500.0);
  scalar_vp = reader.GetReal("medium", "scalar_vp", 2000.0);
  scalar_vs = reader.GetReal("medium", "scalar_vs", 800.0);

  // Basis functions
  basis_gridpoints_x = reader.GetInteger("basis", "npx", 1);
  basis_gridpoints_z = reader.GetInteger("basis", "npz", 1);
  assert(nx_free_parameters % basis_gridpoints_x == 0 and
         nz_free_parameters % basis_gridpoints_z == 0);
  free_parameters = 3 * nx_free_parameters * nz_free_parameters /
                    (basis_gridpoints_x * basis_gridpoints_z);

  // Sources
  peak_frequency = reader.GetReal("sources", "peak_frequency", 50.0);
  t0 = reader.GetReal("sources", "source_timeshift", 0.005);
  delay_cycles_per_shot =
      reader.GetReal("sources", "delay_cycles_per_shot", 12);
  n_sources = reader.GetInteger("sources", "n_sources", 7);
  n_shots = reader.GetInteger("sources", "n_shots", 1);
  // Parse source setup.
  ix_sources = new int[n_sources];
  iz_sources = new int[n_sources];
  moment_angles = new real_simulation[n_sources];
  std::vector<int> ix_sources_vector;
  std::vector<int> iz_sources_vector;
  std::vector<real_simulation> moment_angles_vector;
  parse_string_to_vector(
      reader.Get("sources", "ix_sources", "{25, 50, 75, 100, 125, 150, 175};"),
      &ix_sources_vector);
  parse_string_to_vector(
      reader.Get("sources", "iz_sources", "{10, 10, 10, 10, 10, 10, 10};"),
      &iz_sources_vector);
  parse_string_to_vector(reader.Get("sources", "moment_angles",
                                    "{90, 81, 41, 300, 147, 252, 327};"),
                         &moment_angles_vector);
  if (ix_sources_vector.size() != n_sources or
      iz_sources_vector.size() != n_sources or
      moment_angles_vector.size() != n_sources) {
    std::cout << "Dimension mismatch between n_sources and sources.ix_sources, "
                 "sources.iz_sources or sources.moment_angles"
              << std::endl;
    exit(1);
  }
  for (int i_source = 0; i_source < n_sources; ++i_source) {
    ix_sources[i_source] = ix_sources_vector[i_source];
    iz_sources[i_source] = iz_sources_vector[i_source];
    moment_angles[i_source] = moment_angles_vector[i_source];
  }
  // Parse source stacking
  parse_string_to_nested_int_vector(
      reader.Get("sources", "which_source_to_fire_in_which_shot",
                 "{{0, 1, 2, 3, 4, 5, 6}};"),
      &which_source_to_fire_in_which_shot);
  if (which_source_to_fire_in_which_shot.size() != n_shots) {
    std::cout << "Mismatch between n_shots and "
                 "sources.which_source_to_fire_in_which_shot"
              << std::endl;
    exit(1);
  }
  int total_sources = 0;
  for (const auto &shot_sources : which_source_to_fire_in_which_shot) {
    total_sources += shot_sources.size();
  }
  if (total_sources != n_sources) {
    std::cout << "Mismatch between n_sources and "
                 "sources.which_source_to_fire_in_which_shot"
              << std::endl;
    exit(1);
  }

  // Receivers
  nr = reader.GetInteger("receivers", "nr", 19);
  ix_receivers = new int[nr];
  iz_receivers = new int[nr];
  std::vector<int> ix_receivers_vector;
  std::vector<int> iz_receivers_vector;
  parse_string_to_vector(
      reader.Get("receivers", "ix_receivers",
                 "{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, "
                 "140, 150, 160, 170, 180}; !!"),
      &ix_receivers_vector);
  parse_string_to_vector(
      reader.Get("receivers", "iz_receivers",
                 "{90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, "
                 "90, 90, 90, 90}; !!"),
      &iz_receivers_vector);

  if (ix_receivers_vector.size() != nr or iz_receivers_vector.size() != nr) {
    std::cout << "Mismatch between nr and receivers.ix_receivers or "
                 "receivers.iz_receivers"
              << std::endl;
    exit(1);
  }
  for (int i_receiver = 0; i_receiver < nr; ++i_receiver) {
    ix_receivers[i_receiver] = ix_receivers_vector[i_receiver];
    iz_receivers[i_receiver] = iz_receivers_vector[i_receiver];
  }

  // Inversion
  snapshot_interval = reader.GetInteger("inversion", "snapshot_interval", 10);

  // Output
  observed_data_folder = reader.Get("output", "observed_data_folder", ".");
  stf_folder = reader.Get("output", "stf_folder", ".");

  // Final calculations
  alpha = static_cast<real_simulation>(1.0 / peak_frequency);
  snapshots = ceil(nt / snapshot_interval);
}

// Forward modeller
void fdWaveModel::forward_simulate(int i_shot, bool store_fields, bool verbose,
                                   bool output_wavefields /* = false   */) {
// Set dynamic physical fields to zero to reflect initial conditions.
#pragma omp parallel for collapse(2)
  for (int ix = 0; ix < nx; ++ix) {
    for (int iz = 0; iz < nz; ++iz) {
      vx[ix][iz] = 0.0;
      vz[ix][iz] = 0.0;
      txx[ix][iz] = 0.0;
      tzz[ix][iz] = 0.0;
      txz[ix][iz] = 0.0;
    }
  }

  // If verbose, clock time of modelling.
  double startTime = 0, stopTime = 0, secsElapsed = 0;
  if (verbose) {
    startTime = real_simulation(omp_get_wtime());
  }

  // Time-loop starts here
  for (int it = 0; it < nt; ++it) {
    // Take wavefield snapshot at required intervals.
    if (it % snapshot_interval == 0 and store_fields) {
#pragma omp parallel for collapse(2)
      for (int ix = np_boundary; ix < nx_inner + np_boundary; ++ix) {
        for (int iz = np_boundary; iz < nz_inner + np_boundary; ++iz) {
          accu_vx[i_shot][it / snapshot_interval][ix][iz] = vx[ix][iz];
          accu_vz[i_shot][it / snapshot_interval][ix][iz] = vz[ix][iz];
          accu_txx[i_shot][it / snapshot_interval][ix][iz] = txx[ix][iz];
          accu_txz[i_shot][it / snapshot_interval][ix][iz] = txz[ix][iz];
          accu_tzz[i_shot][it / snapshot_interval][ix][iz] = tzz[ix][iz];
        }
      }
    }

// Record seismograms by integrating velocity into displacement for every
// time-step.
#pragma omp parallel for collapse(1)
    for (int i_receiver = 0; i_receiver < nr; ++i_receiver) {
      if (it == 0) {
        rtf_ux[i_shot][i_receiver][it] =
            dt * vx[ix_receivers[i_receiver]][iz_receivers[i_receiver]] /
            (dx * dz);
        rtf_uz[i_shot][i_receiver][it] =
            dt * vz[ix_receivers[i_receiver]][iz_receivers[i_receiver]] /
            (dx * dz);
      } else {
        rtf_ux[i_shot][i_receiver][it] =
            rtf_ux[i_shot][i_receiver][it - 1] +
            dt * vx[ix_receivers[i_receiver]][iz_receivers[i_receiver]] /
                (dx * dz);
        rtf_uz[i_shot][i_receiver][it] =
            rtf_uz[i_shot][i_receiver][it - 1] +
            dt * vz[ix_receivers[i_receiver]][iz_receivers[i_receiver]] /
                (dx * dz);
      }
    }

// Time integrate dynamic fields for stress ...
#pragma omp parallel for collapse(2)
    for (int ix = 2; ix < nx - 2; ++ix) {
      for (int iz = 2; iz < nz - 2; ++iz) {
        txx[ix][iz] =
            taper[ix][iz] *
            (txx[ix][iz] + dt * (lm[ix][iz] *
                                     (c1 * (vx[ix + 1][iz] - vx[ix][iz]) +
                                      c2 * (vx[ix - 1][iz] - vx[ix + 2][iz])) /
                                     dx +
                                 la[ix][iz] *
                                     (c1 * (vz[ix][iz] - vz[ix][iz - 1]) +
                                      c2 * (vz[ix][iz - 2] - vz[ix][iz + 1])) /
                                     dz));
        tzz[ix][iz] =
            taper[ix][iz] *
            (tzz[ix][iz] + dt * (la[ix][iz] *
                                     (c1 * (vx[ix + 1][iz] - vx[ix][iz]) +
                                      c2 * (vx[ix - 1][iz] - vx[ix + 2][iz])) /
                                     dx +
                                 (lm[ix][iz]) *
                                     (c1 * (vz[ix][iz] - vz[ix][iz - 1]) +
                                      c2 * (vz[ix][iz - 2] - vz[ix][iz + 1])) /
                                     dz));
        txz[ix][iz] =
            taper[ix][iz] *
            (txz[ix][iz] + dt * mu[ix][iz] *
                               ((c1 * (vx[ix][iz + 1] - vx[ix][iz]) +
                                 c2 * (vx[ix][iz - 1] - vx[ix][iz + 2])) /
                                    dz +
                                (c1 * (vz[ix][iz] - vz[ix - 1][iz]) +
                                 c2 * (vz[ix - 2][iz] - vz[ix + 1][iz])) /
                                    dx));
      }
    }
// ... and time integrate dynamic fields for velocity.
#pragma omp parallel for collapse(2)
    for (int ix = 2; ix < nx - 2; ++ix) {
      for (int iz = 2; iz < nz - 2; ++iz) {
        vx[ix][iz] =
            taper[ix][iz] *
            (vx[ix][iz] + b_vx[ix][iz] * dt *
                              ((c1 * (txx[ix][iz] - txx[ix - 1][iz]) +
                                c2 * (txx[ix - 2][iz] - txx[ix + 1][iz])) /
                                   dx +
                               (c1 * (txz[ix][iz] - txz[ix][iz - 1]) +
                                c2 * (txz[ix][iz - 2] - txz[ix][iz + 1])) /
                                   dz));
        vz[ix][iz] =
            taper[ix][iz] *
            (vz[ix][iz] + b_vz[ix][iz] * dt *
                              ((c1 * (txz[ix + 1][iz] - txz[ix][iz]) +
                                c2 * (txz[ix - 1][iz] - txz[ix + 2][iz])) /
                                   dx +
                               (c1 * (tzz[ix][iz + 1] - tzz[ix][iz]) +
                                c2 * (tzz[ix][iz - 1] - tzz[ix][iz + 2])) /
                                   dz));
      }
    }

    // Inject sources at appropriate location and times.
    for (const auto &i_source :
         which_source_to_fire_in_which_shot[i_shot]) { // Don't parallelize
                                                       // in assignment!
      if (it < 1 and verbose) {
        std::cout << "Firing source " << i_source << " in shot " << i_shot
                  << std::endl;
      }
      // |-inject source
      // | (x,x)-couple
      vx[ix_sources[i_source] - 1][iz_sources[i_source]] -=
          moment[i_source][0][0] * stf[i_source][it] * dt *
          b_vz[ix_sources[i_source] - 1][iz_sources[i_source]] /
          (dx * dx * dx * dx);
      vx[ix_sources[i_source]][iz_sources[i_source]] +=
          moment[i_source][0][0] * stf[i_source][it] * dt *
          b_vz[ix_sources[i_source]][iz_sources[i_source]] /
          (dx * dx * dx * dx);
      // | (z,z)-couple
      vz[ix_sources[i_source]][iz_sources[i_source] - 1] -=
          moment[i_source][1][1] * stf[i_source][it] * dt *
          b_vz[ix_sources[i_source]][iz_sources[i_source] - 1] /
          (dz * dz * dz * dz);
      vz[ix_sources[i_source]][iz_sources[i_source]] +=
          moment[i_source][1][1] * stf[i_source][it] * dt *
          b_vz[ix_sources[i_source]][iz_sources[i_source]] /
          (dz * dz * dz * dz);
      // | (x,z)-couple
      vx[ix_sources[i_source] - 1][iz_sources[i_source] + 1] +=
          0.25 * moment[i_source][0][1] * stf[i_source][it] * dt *
          b_vz[ix_sources[i_source] - 1][iz_sources[i_source] + 1] /
          (dx * dx * dx * dx);
      vx[ix_sources[i_source]][iz_sources[i_source] + 1] +=
          0.25 * moment[i_source][0][1] * stf[i_source][it] * dt *
          b_vz[ix_sources[i_source]][iz_sources[i_source] + 1] /
          (dx * dx * dx * dx);
      vx[ix_sources[i_source] - 1][iz_sources[i_source] - 1] -=
          0.25 * moment[i_source][0][1] * stf[i_source][it] * dt *
          b_vz[ix_sources[i_source] - 1][iz_sources[i_source] - 1] /
          (dx * dx * dx * dx);
      vx[ix_sources[i_source]][iz_sources[i_source] - 1] -=
          0.25 * moment[i_source][0][1] * stf[i_source][it] * dt *
          b_vz[ix_sources[i_source]][iz_sources[i_source] - 1] /
          (dx * dx * dx * dx);
      // | (z,x)-couple
      vz[ix_sources[i_source] + 1][iz_sources[i_source] - 1] +=
          0.25 * moment[i_source][1][0] * stf[i_source][it] * dt *
          b_vz[ix_sources[i_source] + 1][iz_sources[i_source] - 1] /
          (dz * dz * dz * dz);
      vz[ix_sources[i_source] + 1][iz_sources[i_source]] +=
          0.25 * moment[i_source][1][0] * stf[i_source][it] * dt *
          b_vz[ix_sources[i_source] + 1][iz_sources[i_source]] /
          (dz * dz * dz * dz);
      vz[ix_sources[i_source] - 1][iz_sources[i_source] - 1] -=
          0.25 * moment[i_source][1][0] * stf[i_source][it] * dt *
          b_vz[ix_sources[i_source] - 1][iz_sources[i_source] - 1] /
          (dz * dz * dz * dz);
      vz[ix_sources[i_source] - 1][iz_sources[i_source]] -=
          0.25 * moment[i_source][1][0] * stf[i_source][it] * dt *
          b_vz[ix_sources[i_source] - 1][iz_sources[i_source]] /
          (dz * dz * dz * dz);
    }

    if (it % 10 == 0 and output_wavefields) {
      // Write wavefields
      std::string filename_vp =
          "snapshots/vx" + zero_pad_number(it, 5) + ".txt";
      std::string filename_vs =
          "snapshots/vz" + zero_pad_number(it, 5) + ".txt";

      std::ofstream file_vx(filename_vp);
      std::ofstream file_vz(filename_vs);

      for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
          file_vx << vx[ix][iz] << " ";
          file_vz << vz[ix][iz] << " ";
        }
        file_vx << std::endl;
        file_vz << std::endl;
      }
      file_vx.close();
      file_vz.close();
    }
  }

  // Output timing if verbose.
  if (verbose) {
    stopTime = omp_get_wtime();
    secsElapsed = stopTime - startTime;
    std::cout << "Seconds elapsed for forward wave simulation: " << secsElapsed
              << std::endl;
  }
}

void fdWaveModel::adjoint_simulate(int i_shot, bool verbose) {
  // Reset dynamical fields
  for (int ix = 0; ix < nx; ++ix) {
    for (int iz = 0; iz < nz; ++iz) {
      vx[ix][iz] = 0.0;
      vz[ix][iz] = 0.0;
      txx[ix][iz] = 0.0;
      tzz[ix][iz] = 0.0;
      txz[ix][iz] = 0.0;
    }
  }

  // If verbose, count time
  double startTime = 0, stopTime = 0, secsElapsed = 0;
  if (verbose) {
    startTime = real_simulation(omp_get_wtime());
  }

  for (int it = nt - 1; it >= 0; --it) {
    // Correlate wavefields
    if (it % snapshot_interval == 0) { // Todo, [X] rewrite for only relevant
                                       // parameters [ ] Check if done properly
#pragma omp parallel for collapse(2)
      for (int ix = np_boundary + nx_inner_boundary;
           ix < np_boundary + nx_inner - nx_inner_boundary; ++ix) {
        for (int iz = np_boundary + nz_inner_boundary;
             iz < np_boundary + nz_inner - nz_inner_boundary; ++iz) {
          density_l_kernel[ix][iz] -=
              snapshot_interval * dt *
              (accu_vx[i_shot][it / snapshot_interval][ix][iz] * vx[ix][iz] +
               accu_vz[i_shot][it / snapshot_interval][ix][iz] * vz[ix][iz]);

          lambda_kernel[ix][iz] +=
              snapshot_interval * dt *
              (((accu_txx[i_shot][it / snapshot_interval][ix][iz] -
                 (accu_tzz[i_shot][it / snapshot_interval][ix][iz] *
                  la[ix][iz]) /
                     lm[ix][iz]) +
                (accu_tzz[i_shot][it / snapshot_interval][ix][iz] -
                 (accu_txx[i_shot][it / snapshot_interval][ix][iz] *
                  la[ix][iz]) /
                     lm[ix][iz])) *
               ((txx[ix][iz] - (tzz[ix][iz] * la[ix][iz]) / lm[ix][iz]) +
                (tzz[ix][iz] - (txx[ix][iz] * la[ix][iz]) / lm[ix][iz]))) /
              ((lm[ix][iz] - ((la[ix][iz] * la[ix][iz]) / (lm[ix][iz]))) *
               (lm[ix][iz] - ((la[ix][iz] * la[ix][iz]) / (lm[ix][iz]))));

          mu_kernel[ix][iz] +=
              snapshot_interval * dt * 2 *
              ((((txx[ix][iz] - (tzz[ix][iz] * la[ix][iz]) / lm[ix][iz]) *
                 (accu_txx[i_shot][it / snapshot_interval][ix][iz] -
                  (accu_tzz[i_shot][it / snapshot_interval][ix][iz] *
                   la[ix][iz]) /
                      lm[ix][iz])) +
                ((tzz[ix][iz] - (txx[ix][iz] * la[ix][iz]) / lm[ix][iz]) *
                 (accu_tzz[i_shot][it / snapshot_interval][ix][iz] -
                  (accu_txx[i_shot][it / snapshot_interval][ix][iz] *
                   la[ix][iz]) /
                      lm[ix][iz]))) /
                   ((lm[ix][iz] - ((la[ix][iz] * la[ix][iz]) / (lm[ix][iz]))) *
                    (lm[ix][iz] - ((la[ix][iz] * la[ix][iz]) / (lm[ix][iz])))) +
               2 * (txz[ix][iz] *
                    accu_txz[i_shot][it / snapshot_interval][ix][iz] /
                    (4 * mu[ix][iz] * mu[ix][iz])));
        }
      }
    }

// Reverse time integrate dynamic fields for stress
#pragma omp parallel for collapse(2)
    for (int ix = 2; ix < nx - 2; ++ix) {
      for (int iz = 2; iz < nz - 2; ++iz) {
        txx[ix][iz] =
            taper[ix][iz] *
            (txx[ix][iz] - dt * (lm[ix][iz] *
                                     (c1 * (vx[ix + 1][iz] - vx[ix][iz]) +
                                      c2 * (vx[ix - 1][iz] - vx[ix + 2][iz])) /
                                     dx +
                                 la[ix][iz] *
                                     (c1 * (vz[ix][iz] - vz[ix][iz - 1]) +
                                      c2 * (vz[ix][iz - 2] - vz[ix][iz + 1])) /
                                     dz));
        tzz[ix][iz] =
            taper[ix][iz] *
            (tzz[ix][iz] - dt * (la[ix][iz] *
                                     (c1 * (vx[ix + 1][iz] - vx[ix][iz]) +
                                      c2 * (vx[ix - 1][iz] - vx[ix + 2][iz])) /
                                     dx +
                                 (lm[ix][iz]) *
                                     (c1 * (vz[ix][iz] - vz[ix][iz - 1]) +
                                      c2 * (vz[ix][iz - 2] - vz[ix][iz + 1])) /
                                     dz));
        txz[ix][iz] =
            taper[ix][iz] *
            (txz[ix][iz] - dt * mu[ix][iz] *
                               ((c1 * (vx[ix][iz + 1] - vx[ix][iz]) +
                                 c2 * (vx[ix][iz - 1] - vx[ix][iz + 2])) /
                                    dz +
                                (c1 * (vz[ix][iz] - vz[ix - 1][iz]) +
                                 c2 * (vz[ix - 2][iz] - vz[ix + 1][iz])) /
                                    dx));
      }
    }
// Reverse time integrate dynamic fields for velocity
#pragma omp parallel for collapse(2)
    for (int ix = 2; ix < nx - 2; ++ix) {
      for (int iz = 2; iz < nz - 2; ++iz) {
        vx[ix][iz] =
            taper[ix][iz] *
            (vx[ix][iz] - b_vx[ix][iz] * dt *
                              ((c1 * (txx[ix][iz] - txx[ix - 1][iz]) +
                                c2 * (txx[ix - 2][iz] - txx[ix + 1][iz])) /
                                   dx +
                               (c1 * (txz[ix][iz] - txz[ix][iz - 1]) +
                                c2 * (txz[ix][iz - 2] - txz[ix][iz + 1])) /
                                   dz));
        vz[ix][iz] =
            taper[ix][iz] *
            (vz[ix][iz] - b_vz[ix][iz] * dt *
                              ((c1 * (txz[ix + 1][iz] - txz[ix][iz]) +
                                c2 * (txz[ix - 1][iz] - txz[ix + 2][iz])) /
                                   dx +
                               (c1 * (tzz[ix][iz + 1] - tzz[ix][iz]) +
                                c2 * (tzz[ix][iz - 1] - tzz[ix][iz + 2])) /
                                   dz));
      }
    }

    // Inject adjoint sources
    for (int ir = 0; ir < nr; ++ir) {
      vx[ix_receivers[ir]][iz_receivers[ir]] +=
          dt * b_vx[ix_receivers[ir]][iz_receivers[ir]] *
          a_stf_ux[i_shot][ir][it] / (dx * dz);
      vz[ix_receivers[ir]][iz_receivers[ir]] +=
          dt * b_vz[ix_receivers[ir]][iz_receivers[ir]] *
          a_stf_uz[i_shot][ir][it] / (dx * dz);
    }
  }

  // Output timing
  if (verbose) {
    stopTime = omp_get_wtime();
    secsElapsed = stopTime - startTime;
    std::cout << "Seconds elapsed for adjoint wave simulation: " << secsElapsed
              << std::endl;
  }
}

void fdWaveModel::write_receivers() { write_receivers(std::string("")); }

void fdWaveModel::write_receivers(const std::string prefix) {
  std::string filename_ux;
  std::string filename_uz;

  std::ofstream receiver_file_ux;
  std::ofstream receiver_file_uz;

  for (int i_shot = 0; i_shot < n_shots; ++i_shot) {
    filename_ux =
        observed_data_folder + "/rtf_ux" + std::to_string(i_shot) + ".txt";
    filename_uz =
        observed_data_folder + "/rtf_uz" + std::to_string(i_shot) + ".txt";

    receiver_file_ux.open(filename_ux);
    receiver_file_uz.open(filename_uz);

    receiver_file_ux.precision(std::numeric_limits<real_simulation>::digits10 +
                               10);
    receiver_file_uz.precision(std::numeric_limits<real_simulation>::digits10 +
                               10);

    for (int i_receiver = 0; i_receiver < nr; ++i_receiver) {
      receiver_file_ux << std::endl;
      receiver_file_uz << std::endl;
      for (int it = 0; it < nt; ++it) {
        receiver_file_ux << rtf_ux[i_shot][i_receiver][it] << " ";
        receiver_file_uz << rtf_uz[i_shot][i_receiver][it] << " ";
      }
    }
    receiver_file_ux.close();
    receiver_file_uz.close();
  }
}

void fdWaveModel::write_sources() {
  std::string filename_sources;
  std::ofstream shot_file;

  for (int i_shot = 0; i_shot < n_shots; ++i_shot) {
    filename_sources =
        stf_folder + "/sources_shot_" + std::to_string(i_shot) + ".txt";

    shot_file.open(filename_sources);

    shot_file.precision(std::numeric_limits<real_simulation>::digits10 + 10);

    for (int i_source : which_source_to_fire_in_which_shot[i_shot]) {
      shot_file << std::endl;
      for (int it = 0; it < nt; ++it) {
        shot_file << stf[i_source][it] << " ";
      }
    }
    shot_file.close();
  }
}

void fdWaveModel::update_from_velocity() {
#pragma omp parallel for collapse(2)
  for (int ix = 0; ix < nx; ++ix) {
    for (int iz = 0; iz < nz; ++iz) {
      mu[ix][iz] = real_simulation(pow(vs[ix][iz], 2) * rho[ix][iz]);
      lm[ix][iz] = real_simulation(pow(vp[ix][iz], 2) * rho[ix][iz]);
      la[ix][iz] = lm[ix][iz] - 2 * mu[ix][iz];
      b_vx[ix][iz] = real_simulation(1.0 / rho[ix][iz]);
      b_vz[ix][iz] = b_vx[ix][iz];
    }
  }
}

void fdWaveModel::load_receivers(bool verbose) {
  std::string filename_ux;
  std::string filename_uz;

  std::ifstream receiver_file_ux;
  std::ifstream receiver_file_uz;

  for (int i_shot = 0; i_shot < n_shots; ++i_shot) {
    // Create filename from folder and shot.
    filename_ux =
        observed_data_folder + "/rtf_ux" + std::to_string(i_shot) + ".txt";
    filename_uz =
        observed_data_folder + "/rtf_uz" + std::to_string(i_shot) + ".txt";

    // Attempt to open the file by its filename.
    receiver_file_ux.open(filename_ux);
    receiver_file_uz.open(filename_uz);

    // Check if the file actually exists
    if (verbose) {
      std::cout << "File for ux data at shot " << i_shot << " is "
                << (receiver_file_ux.good() ? "good (exists at least)."
                                            : "ungood.")
                << std::endl;
      std::cout << "File for uz data at shot " << i_shot << " is "
                << (receiver_file_uz.good() ? "good (exists at least)."
                                            : "ungood.")
                << std::endl;
    }
    if (!receiver_file_ux.good() or !receiver_file_uz.good()) {
      throw std::invalid_argument("Not all data is present!");
    }

    real_simulation placeholder_ux;
    real_simulation placeholder_uz;

    for (int i_receiver = 0; i_receiver < nr; ++i_receiver) {
      for (int it = 0; it < nt; ++it) {
        receiver_file_ux >> placeholder_ux;
        receiver_file_uz >> placeholder_uz;

        rtf_ux_true[i_shot][i_receiver][it] = placeholder_ux;
        rtf_uz_true[i_shot][i_receiver][it] = placeholder_uz;
      }
    }

    // Check data was large enough for set up
    if (!receiver_file_ux.good() or !receiver_file_uz.good()) {
      std::cout << "Received bad state of file at end of reading! Does "
                   "the data match the set up?"
                << std::endl;
      throw std::invalid_argument("Not enough data is present!");
    }
    // Try to load more data ...
    receiver_file_ux >> placeholder_ux;
    receiver_file_uz >> placeholder_uz;
    // ... which shouldn't be possible
    if (receiver_file_ux.good() or receiver_file_uz.good()) {
      std::cout << "Received good state of file past reading! Does the "
                   "data match the set up?"
                << std::endl;
      throw std::invalid_argument("Too much data is present!");
    }

    receiver_file_uz.close();
    receiver_file_ux.close();
  }
}

void fdWaveModel::calculate_l2_misfit() {
  misfit = 0;
  for (int i_shot = 0; i_shot < n_shots; ++i_shot) {
    for (int i_receiver = 0; i_receiver < nr; ++i_receiver) {
      for (int it = 0; it < nt; ++it) {
        misfit += 0.5 * dt *
                  pow(rtf_ux_true[i_shot][i_receiver][it] -
                          rtf_ux[i_shot][i_receiver][it],
                      2); // /
        // data_variance_ux[i_shot][i_receiver][it];
        misfit += 0.5 * dt *
                  pow(rtf_uz_true[i_shot][i_receiver][it] -
                          rtf_uz[i_shot][i_receiver][it],
                      2); // /
                          // data_variance_uz[i_shot][i_receiver][it];
      }
    }
  }
}

void fdWaveModel::calculate_l2_adjoint_sources() {
#pragma omp parallel for collapse(3)
  for (int is = 0; is < n_shots; ++is) {
    for (int ir = 0; ir < nr; ++ir) {
      for (int it = 0; it < nt; ++it) {
        a_stf_ux[is][ir][it] = rtf_ux[is][ir][it] - rtf_ux_true[is][ir][it];
        a_stf_uz[is][ir][it] = rtf_uz[is][ir][it] - rtf_uz_true[is][ir][it];
      }
    }
  }
}

void fdWaveModel::map_kernels_to_velocity() {
#pragma omp parallel for collapse(2)
  for (int ix = 0; ix < nx; ++ix) {
    for (int iz = 0; iz < nz; ++iz) {
      vp_kernel[ix][iz] = 2 * vp[ix][iz] * lambda_kernel[ix][iz] / b_vx[ix][iz];
      vs_kernel[ix][iz] = (2 * vs[ix][iz] * mu_kernel[ix][iz] -
                           4 * vs[ix][iz] * lambda_kernel[ix][iz]) /
                          b_vx[ix][iz];
      density_v_kernel[ix][iz] =
          density_l_kernel[ix][iz] +
          (vp[ix][iz] * vp[ix][iz] - 2 * vs[ix][iz] * vs[ix][iz]) *
              lambda_kernel[ix][iz] +
          vs[ix][iz] * vs[ix][iz] * mu_kernel[ix][iz];
    }
  }
}

void fdWaveModel::load_model(const std::string &de_path,
                             const std::string &vp_path,
                             const std::string &vs_path, bool verbose) {
  std::ifstream de_file;
  std::ifstream vp_file;
  std::ifstream vs_file;

  de_file.open(de_path);
  vp_file.open(vp_path);
  vs_file.open(vs_path);

  // Check if the file actually exists
  if (verbose) {
    std::cout << "Loading models." << std::endl;
    std::cout << "File: " << de_path << std::endl;
    std::cout << "File for density is "
              << (de_file.good() ? "good (exists at least)." : "ungood.")
              << std::endl;
    std::cout << "File: " << vp_path << std::endl;
    std::cout << "File for P-wave velocity is "
              << (vp_file.good() ? "good (exists at least)." : "ungood.")
              << std::endl;
    std::cout << "File: " << vs_path << std::endl;
    std::cout << "File for S-wave velocity is "
              << (vs_file.good() ? "good (exists at least)." : "ungood.")
              << std::endl;
  }
  if (!de_file.good() or !vp_file.good() or !vs_file.good()) {
    throw std::invalid_argument(
        "The files for target models don't seem to exist.\r\n"
        "Paths:\r\n"
        "\tDensity target: " +
        de_path +
        "\r\n"
        "\tP-wave target : " +
        vp_path +
        "\r\n"
        "\tS-wave target : " +
        vs_path + "\r\n");
  }

  real_simulation placeholder_de;
  real_simulation placeholder_vp;
  real_simulation placeholder_vs;
  for (int ix = 0; ix < nx; ++ix) {
    for (int iz = 0; iz < nz; ++iz) {
      de_file >> placeholder_de;
      vp_file >> placeholder_vp;
      vs_file >> placeholder_vs;

      rho[ix][iz] = placeholder_de;
      vp[ix][iz] = placeholder_vp;
      vs[ix][iz] = placeholder_vs;
    }
  }

  // Check data was large enough for set up
  if (!de_file.good() or !vp_file.good() or !vs_file.good()) {
    std::cout << "Received bad state of one of the files at end of "
                 "reading. Does the data match the domain?"
              << std::endl;
    throw std::invalid_argument("Not enough data is present!");
  }
  // Try to load more data ...
  de_file >> placeholder_de;
  vp_file >> placeholder_vp;
  vs_file >> placeholder_vs;
  // ... which shouldn't be possible
  if (de_file.good() or vp_file.good() or vs_file.good()) {
    std::cout << "Received good state of file past reading. Does the data "
                 "match the domain?"
              << std::endl;
    throw std::invalid_argument("Too much data is present!");
  }

  de_file.close();
  vp_file.close();
  vs_file.close();

  update_from_velocity();
  if (verbose)
    std::cout << std::endl;
}

void fdWaveModel::run_model(bool verbose, bool simulate_adjoint) {
  for (int i_shot = 0; i_shot < n_shots; ++i_shot) {
    forward_simulate(i_shot, true, verbose);
  }
  calculate_l2_misfit();
  if (simulate_adjoint) {
    calculate_l2_adjoint_sources();
    reset_kernels();
    for (int is = 0; is < n_shots; ++is) {
      adjoint_simulate(is, verbose);
    }
    map_kernels_to_velocity();
  }
}

void fdWaveModel::reset_kernels() {
  for (int ix = 0; ix < nx; ++ix) {
    for (int iz = 0; iz < nz; ++iz) {
      lambda_kernel[ix][iz] = 0.0;
      mu_kernel[ix][iz] = 0.0;
      density_l_kernel[ix][iz] = 0.0;
    }
  }
}

void fdWaveModel::write_kernels() {
  std::string filename_kernel_vp = "kernel_vp.txt";
  ;
  std::string filename_kernel_vs = "kernel_vs.txt";
  ;
  std::string filename_kernel_density = "kernel_density.txt";
  ;

  std::ofstream file_kernel_vp(filename_kernel_vp);
  std::ofstream file_kernel_vs(filename_kernel_vs);
  std::ofstream file_kernel_density(filename_kernel_density);

  for (int ix = 0; ix < nx; ++ix) {
    for (int iz = 0; iz < nz; ++iz) {
      file_kernel_vp << vp_kernel[ix][iz] << " ";
      file_kernel_vs << vs_kernel[ix][iz] << " ";
      file_kernel_density << density_v_kernel[ix][iz] << " ";
    }
    file_kernel_vp << std::endl;
    file_kernel_vs << std::endl;
    file_kernel_density << std::endl;
  }
  file_kernel_vp.close();
  file_kernel_vs.close();
  file_kernel_density.close();
}

dynamic_vector fdWaveModel::get_model_vector() {
  // Assert that the chosen parametrization makes full blocks (i.e. that we
  // don't have 3 horizontal subdivisions for 20 points, which would result in a
  // non-rounded number for points per cell)
  assert(nx_free_parameters % basis_gridpoints_x == 0 and
         nz_free_parameters % basis_gridpoints_z == 0);

  int n_free_per_par = nx_free_parameters * nz_free_parameters /
                       (basis_gridpoints_x * basis_gridpoints_z);

  dynamic_vector m = dynamic_vector(n_free_per_par * 3, 1);

  // Loop over points within the free zone, so excluding boundary and non-free
  // parameters
  for (int ix = 0; ix < nx_free_parameters / basis_gridpoints_x; ++ix) {
    for (int iz = 0; iz < nz_free_parameters / basis_gridpoints_z; ++iz) {
      // Model vector index
      int i_parameter = ix + iz * nx_free_parameters / basis_gridpoints_x;

      // Global x index
      int gix = ix * basis_gridpoints_x + nx_inner_boundary + np_boundary;
      // Global z index
      int giz = iz * basis_gridpoints_z + nz_inner_boundary + np_boundary;

      m[i_parameter] = 0;
      m[i_parameter + n_free_per_par] = 0;
      m[i_parameter + 2 * n_free_per_par] = 0;

      // Average over the points in the cell
      // We could easily just retrieve one of the values in the block, but when
      // loading a model onto the grid, all the gridpoindts might not be the
      // same value.
      for (int sub_ix = 0; sub_ix < basis_gridpoints_x; sub_ix++) {
        for (int sub_iz = 0; sub_iz < basis_gridpoints_z; sub_iz++) {
          m[i_parameter] += vp[gix + sub_ix][giz + sub_iz] /
                            (basis_gridpoints_x * basis_gridpoints_z);
          m[i_parameter + n_free_per_par] +=
              vs[gix + sub_ix][giz + sub_iz] /
              (basis_gridpoints_x * basis_gridpoints_z);
          m[i_parameter + 2 * n_free_per_par] +=
              rho[gix + sub_ix][giz + sub_iz] /
              (basis_gridpoints_x * basis_gridpoints_z);
        }
      }
    }
  }
  return m;
}

void fdWaveModel::set_model_vector(dynamic_vector m) {
  assert(nx_free_parameters % basis_gridpoints_x == 0 and
         nz_free_parameters % basis_gridpoints_z == 0);

  int n_free_per_par = nx_free_parameters * nz_free_parameters /
                       (basis_gridpoints_x * basis_gridpoints_z);

  assert(m.size() == n_free_per_par * 3);

  // Loop over points within the free zone, so excluding boundary and non-free
  // parameters
  for (int ix = 0; ix < nx_free_parameters / basis_gridpoints_x; ++ix) {
    for (int iz = 0; iz < nz_free_parameters / basis_gridpoints_z; ++iz) {
      // Model vector index
      int i_parameter = ix + iz * nx_free_parameters / basis_gridpoints_x;

      // Global x index
      int gix = ix * basis_gridpoints_x + nx_inner_boundary + np_boundary;
      // Global z index
      int giz = iz * basis_gridpoints_z + nz_inner_boundary + np_boundary;

      // Average over the points in the cell
      // We could easily just retrieve one of the values in the block, but when
      // loading a model onto the grid, all the gridpoindts might not be the
      // same value.
      for (int sub_ix = 0; sub_ix < basis_gridpoints_x; sub_ix++) {
        for (int sub_iz = 0; sub_iz < basis_gridpoints_z; sub_iz++) {
          vp[gix + sub_ix][giz + sub_iz] = m[i_parameter];
          vs[gix + sub_ix][giz + sub_iz] = m[i_parameter + n_free_per_par];
          rho[gix + sub_ix][giz + sub_iz] = m[i_parameter + 2 * n_free_per_par];
        }
      }
    }
  }
  update_from_velocity();
}

dynamic_vector fdWaveModel::get_gradient_vector() {
  // Assert that the chosen parametrization makes full blocks (i.e. that we
  // don't have 3 horizontal subdivisions for 20 points, which would result in a
  // non-rounded number for points per cell)
  assert(nx_free_parameters % basis_gridpoints_x == 0 and
         nz_free_parameters % basis_gridpoints_z == 0);

  int n_free_per_par = nx_free_parameters * nz_free_parameters /
                       (basis_gridpoints_x * basis_gridpoints_z);

  dynamic_vector g = dynamic_vector(n_free_per_par * 3, 1);

  // Loop over points within the free zone, so excluding boundary and non-free
  // parameters
  for (int ix = 0; ix < nx_free_parameters / basis_gridpoints_x; ++ix) {
    for (int iz = 0; iz < nz_free_parameters / basis_gridpoints_z; ++iz) {
      // Model vector index
      int i_parameter = ix + iz * nx_free_parameters / basis_gridpoints_x;

      // Global x index
      int gix = ix * basis_gridpoints_x + nx_inner_boundary + np_boundary;
      // Global z index
      int giz = iz * basis_gridpoints_z + nz_inner_boundary + np_boundary;

      g[i_parameter] = 0;
      g[i_parameter + n_free_per_par] = 0;
      g[i_parameter + 2 * n_free_per_par] = 0;

      // Average over the points in the cell
      // We could easily just retrieve one of the values in the block, but when
      // loading a model onto the grid, all the gridpoindts might not be the
      // same value.
      for (int sub_ix = 0; sub_ix < basis_gridpoints_x; sub_ix++) {
        for (int sub_iz = 0; sub_iz < basis_gridpoints_z; sub_iz++) {
          g[i_parameter] +=
              vp_kernel[gix + sub_ix][giz + sub_iz]; // / (basis_gridpoints_x *
                                                     // basis_gridpoints_z);
          g[i_parameter + n_free_per_par] +=
              vs_kernel[gix + sub_ix][giz + sub_iz]; // / (basis_gridpoints_x *
                                                     // basis_gridpoints_z);
          g[i_parameter + 2 * n_free_per_par] +=
              density_v_kernel[gix + sub_ix]
                              [giz + sub_iz]; //  / (basis_gridpoints_x *
                                              //  basis_gridpoints_z);
        }
      }
    }
  }
  return g;
}

dynamic_vector fdWaveModel::load_vector(const std::string &model_vector_path,
                                        bool verbose) {
  int n_free_per_par = nx_free_parameters * nz_free_parameters;

  dynamic_vector m = dynamic_vector(n_free_per_par * 3, 1);

  std::ifstream model_vector_file;
  model_vector_file.open(model_vector_path);

  // Check if the file actually exists
  if (verbose) {
    std::cout << "Loading model vector." << std::endl;
    std::cout << "File: " << model_vector_path << std::endl;
    std::cout << "File is "
              << (model_vector_file.good() ? "good (exists at least)."
                                           : "ungood.")
              << std::endl;
  }
  if (!model_vector_file.good()) {
    throw std::invalid_argument(
        "The files for target models don't seem to exist.\r\n"
        "Paths:\r\n"
        "\tDensity target: " +
        model_vector_path + "\r\n");
  }

  real_simulation placeholder_model_vector;
  for (int i = 0; i < n_free_per_par * 3; ++i) {
    model_vector_file >> placeholder_model_vector;
    m[i] = placeholder_model_vector;
  }

  // Check data was large enough for set up
  if (!model_vector_file.good()) {
    std::cout << "Received bad state of one of the files at end of "
                 "reading. Does the data match the domain?"
              << std::endl;
    throw std::invalid_argument("Not enough data is present!");
  }
  // Try to load more data ...
  model_vector_file >> placeholder_model_vector;
  // ... which shouldn't be possible
  if (model_vector_file.good()) {
    std::cout << "Received good state of file past reading. Does the data "
                 "match the domain?"
              << std::endl;
    throw std::invalid_argument("Too much data is present!");
  }

  model_vector_file.close();
  return m;
}

// Allocation and deallocation

void allocate_1d_array(real_simulation *&pDouble, int dim1) {
  pDouble = new real_simulation[dim1];
}

void allocate_2d_array(real_simulation **&pDouble, const int dim1,
                       const int dim2) {
  pDouble = new real_simulation *[dim1];
  for (int i = 0; i < dim1; ++i)
    allocate_1d_array(pDouble[i], dim2);
}

void allocate_3d_array(real_simulation ***&pDouble, int dim1, int dim2,
                       int dim3) {
  pDouble = new real_simulation **[dim1];
  for (int i = 0; i < dim1; ++i)
    allocate_2d_array(pDouble[i], dim2, dim3);
}

void allocate_4d_array(real_simulation ****&pDouble, int dim1, int dim2,
                       int dim3, int dim4) {
  pDouble = new real_simulation ***[dim1];
  for (int i = 0; i < dim1; ++i)
    allocate_3d_array(pDouble[i], dim2, dim3, dim4);
}

void deallocate_1d_array(real_simulation *&pDouble) {
  delete[] pDouble;
  pDouble = nullptr;
}

void deallocate_2d_array(real_simulation **&pDouble, const int dim1) {
  for (int i = 0; i < dim1; i++) {
    deallocate_1d_array(pDouble[i]);
  }
  delete[] pDouble;
  pDouble = nullptr;
}

void deallocate_3d_array(real_simulation ***&pDouble, const int dim1,
                         const int dim2) {
  for (int i = 0; i < dim1; i++) {
    deallocate_2d_array(pDouble[i], dim2);
  }
  delete[] pDouble;
  pDouble = nullptr;
}

void deallocate_4d_array(real_simulation ****&pDouble, const int dim1,
                         const int dim2, const int dim3) {
  for (int i = 0; i < dim1; i++) {
    deallocate_3d_array(pDouble[i], dim2, dim3);
  }
  delete[] pDouble;
  pDouble = nullptr;
}

template <class T>
void parse_string_to_vector(std::basic_string<char> string_to_parse,
                            std::vector<T> *destination_vector) {
  // Erase all spaces
  string_to_parse.erase(
      remove_if(string_to_parse.begin(), string_to_parse.end(), isspace),
      string_to_parse.end());
  // Find end of data and cut afterwards
  size_t pos = string_to_parse.find("}");
  string_to_parse.erase(pos, string_to_parse.length());
  // Cut leading curly brace
  string_to_parse.erase(0, 1);
  // Split up string
  std::string delimiter = ",";
  pos = 0;
  std::string token;
  while ((pos = string_to_parse.find(delimiter)) != std::string::npos) {
    token = string_to_parse.substr(0, pos);
    destination_vector->emplace_back(atof(token.c_str()));
    string_to_parse.erase(0, pos + delimiter.length());
  }
  token = string_to_parse.substr(0, pos);
  destination_vector->emplace_back(atof(token.c_str()));
}

void parse_string_to_nested_int_vector(
    std::basic_string<char> string_to_parse,
    std::vector<std::vector<int>> *destination_vector) { // todo clean up
  // Erase all spaces
  string_to_parse.erase(
      remove_if(string_to_parse.begin(), string_to_parse.end(), isspace),
      string_to_parse.end());

  std::string delimiter_outer = "},{";
  string_to_parse.erase(0, 2);

  size_t pos_outer = 0;
  std::string token_outer;

  while ((pos_outer = string_to_parse.find(delimiter_outer)) !=
         std::string::npos) {
    std::vector<int> sub_vec;

    token_outer = string_to_parse.substr(0, pos_outer);

    std::string delimiter_inner = ",";
    size_t pos_inner = 0;
    std::string token_inner;
    while ((pos_inner = token_outer.find(delimiter_inner)) !=
           std::string::npos) {
      token_inner = token_outer.substr(0, pos_inner);
      sub_vec.emplace_back(atof(token_inner.c_str()));
      token_outer.erase(0, pos_inner + delimiter_inner.length());
    }
    token_inner = token_outer.substr(0, pos_inner);
    sub_vec.emplace_back(atof(token_inner.c_str()));

    destination_vector->emplace_back(sub_vec);

    string_to_parse.erase(0, pos_outer + delimiter_outer.length());
  }

  // Process last vector
  std::vector<int> sub_vec;
  pos_outer = string_to_parse.find("}};");
  token_outer = string_to_parse.substr(0, pos_outer);
  //    std::cout << token_outer << std::endl;
  std::string delimiter_inner = ",";
  size_t pos_inner = 0;
  std::string token_inner;
  while ((pos_inner = token_outer.find(delimiter_inner)) != std::string::npos) {
    token_inner = token_outer.substr(0, pos_inner);
    sub_vec.emplace_back(atof(token_inner.c_str()));
    //        std::cout << token_inner << std::endl;
    token_outer.erase(0, pos_inner + delimiter_inner.length());
  }
  token_inner = token_outer.substr(0, pos_inner);
  sub_vec.emplace_back(atof(token_inner.c_str()));
  //    std::cout << token_inner << std::endl;
  destination_vector->emplace_back(sub_vec);
  //    destination_vector->emplace_back(atof(token_outer.c_str()));
}

void cross_correlate(const real_simulation *signal1,
                     const real_simulation *signal2, real_simulation *r,
                     int signal_length, int max_delay) {
  // Calculate means
  real_simulation mean1 = 0;
  real_simulation mean2 = 0;
#pragma omp parallel for reduction(+ : mean1, mean2)
  for (int i = 0; i < signal_length; i++) {
    mean1 += signal1[i];
    mean2 += signal2[i];
  }
  mean1 /= signal_length;
  mean2 /= signal_length;

  //    std::cout << "mean1: " << mean1 << " mean2: " << mean2 << std::endl;

  // Calculate the denominator from standard deviations.
  real_simulation std1 = 0;
  real_simulation std2 = 0;
#pragma omp parallel for reduction(+ : std1, std2)
  for (int i = 0; i < signal_length; i++) {
    std1 += (signal1[i] - mean1) * (signal1[i] - mean1);
    std2 += (signal2[i] - mean2) * (signal2[i] - mean2);
  }
  real_simulation denominator = sqrt(std1 * std2);

// Calculate cross-correlation
#pragma omp parallel for
  for (int delay = -max_delay; delay < max_delay; delay++) {
    real_simulation sxy = 0;
    //#pragma omp parallel for reduction(+: sxy) // not really needed if
    // more delays are present than physical cores (pretty much always).
    for (int i = 0; i < signal_length; i++) {
      int j = i + delay;
      if (j < 0 || j >= signal_length)
        continue;
      else
        sxy += (signal1[i] - mean1) * (signal2[j] - mean2);
    }
    r[delay + max_delay] = sxy / denominator;
  }
}

std::string zero_pad_number(int num, int pad) {
  std::ostringstream ss;
  ss << std::setw(pad) << std::setfill('0') << num;
  return ss.str();
}

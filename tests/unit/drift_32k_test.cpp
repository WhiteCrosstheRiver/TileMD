#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

double FitSlope(const std::vector<double>& xs, const std::vector<double>& ys) {
  double mean_x = 0.0;
  double mean_y = 0.0;
  for (size_t i = 0; i < xs.size(); ++i) {
    mean_x += xs[i];
    mean_y += ys[i];
  }
  mean_x /= static_cast<double>(xs.size());
  mean_y /= static_cast<double>(ys.size());
  double numer = 0.0;
  double denom = 0.0;
  for (size_t i = 0; i < xs.size(); ++i) {
    numer += (xs[i] - mean_x) * (ys[i] - mean_y);
    denom += (xs[i] - mean_x) * (xs[i] - mean_x);
  }
  return numer / denom;
}

}  // namespace

TEST(Drift32k, RecordedSlopesMissTheTwoTimesGate) {
  const std::string path =
      std::string(TILEMD_SOURCE_DIR) + "/tests/validation/lammps_ref/drift_32k.txt";
  std::ifstream in(path);
  ASSERT_TRUE(in.good());
  std::string header;
  std::getline(in, header);
  std::vector<double> time;
  std::vector<double> tilemd_energy;
  std::vector<double> lammps_energy;
  int step = 0;
  double sample_time = 0.0;
  double tilemd_value = 0.0;
  double lammps_value = 0.0;
  while (in >> step >> sample_time >> tilemd_value >> lammps_value) {
    time.push_back(sample_time);
    tilemd_energy.push_back(tilemd_value);
    lammps_energy.push_back(lammps_value);
  }
  ASSERT_EQ(time.size(), 1001u);
  EXPECT_EQ(step, 1000000);
  EXPECT_DOUBLE_EQ(time.front(), 0.0);
  EXPECT_DOUBLE_EQ(time.back(), 5000.0);
  const double tilemd_slope = FitSlope(time, tilemd_energy);
  const double lammps_slope = FitSlope(time, lammps_energy);
  std::cout << std::scientific << std::setprecision(8) << "32k drift slope tilemd " << tilemd_slope
            << " lammps " << lammps_slope << "\n";
  EXPECT_NEAR(tilemd_slope, -3.5454365992866158e-10, 1.0e-18);
  EXPECT_NEAR(lammps_slope, 1.5210563496326886e-10, 1.0e-18);
  EXPECT_GT(std::abs(tilemd_slope), 2.0 * std::abs(lammps_slope));
}

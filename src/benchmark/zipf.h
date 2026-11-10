// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include <iomanip>
#include <fstream>
#include <iostream>
#include <stack>
#include <type_traits>
#include <random>
// Zipf generator, inspired by
// https://github.com/brianfrankcooper/YCSB/blob/master/core/src/main/java/site/ycsb/generator/ScrambledZipfianGenerator.java
// https://github.com/brianfrankcooper/YCSB/blob/master/core/src/main/java/site/ycsb/generator/ZipfianGenerator.java

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

class ScrambledZipfianGenerator {
 public:
  int num_keys_;
  double zipfian_constant_;
  double zeta_n_;

  double alpha_;
  double eta_;

  std::mt19937_64 gen_;
  std::uniform_real_distribution<double> dis_;

  std::vector<double> cdf_;

  explicit ScrambledZipfianGenerator(int num_keys,
                                     size_t *seed = nullptr,
                                     double skew = 0.99)
      : num_keys_(num_keys),
        zipfian_constant_(skew),
        zeta_n_(0.0),
        alpha_(0.0),
        eta_(0.0),
        gen_(std::random_device{}()),
        dis_(0.0, 1.0) {

    if (seed) {
      gen_.seed(*seed);
    }

    if (zipfian_constant_ < 0.0) {
      throw std::invalid_argument("Zipf alpha must be >= 0");
    }

    buildCDF();
  }

  int nextValue() {
    double u = dis_(gen_);

    auto it = std::lower_bound(cdf_.begin(), cdf_.end(), u);
    int rank = std::distance(cdf_.begin(), it);

    if (rank >= num_keys_) {
      rank = num_keys_ - 1;
    }

    int ret = fnv1a(rank) % num_keys_;
    return ret;
  }

 private:

  void buildCDF() {
    cdf_.resize(num_keys_);

    if (zipfian_constant_ == 0.0) {
      for (int i = 0; i < num_keys_; ++i) {
        cdf_[i] = double(i + 1) / num_keys_;
      }
      zeta_n_ = 1.0;
      return;
    }

    zeta_n_ = 0.0;
    for (int i = 1; i <= num_keys_; ++i) {
      zeta_n_ += 1.0 / std::pow(i, zipfian_constant_);
    }

    double cumulative = 0.0;
    for (int i = 1; i <= num_keys_; ++i) {
      cumulative += (1.0 / std::pow(i, zipfian_constant_)) / zeta_n_;
      cdf_[i - 1] = cumulative;
    }

    cdf_[num_keys_ - 1] = 1.0;
  }

 public:

  static const uint32_t PRIME = 0x01000193;
  static const uint32_t SEED  = 0x811C9DC5;

  inline uint32_t fnv1a(unsigned char oneByte, uint32_t hash = SEED) {
    return (oneByte ^ hash) * PRIME;
  }

  inline uint32_t fnv1a(int fourBytes, uint32_t hash = SEED) {
    const unsigned char* ptr =
        reinterpret_cast<const unsigned char*>(&fourBytes);
    hash = fnv1a(*ptr++, hash);
    hash = fnv1a(*ptr++, hash);
    hash = fnv1a(*ptr++, hash);
    return fnv1a(*ptr, hash);
  }
};

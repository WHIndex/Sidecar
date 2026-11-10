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

// class ScrambledZipfianGenerator {
//  public:
//   int num_keys_;
//   double zipfian_constant_;
//   double zeta_n_;
//   double alpha_;
//   double eta_;
//   std::mt19937_64 gen_;
//   std::uniform_real_distribution<double> dis_;

//   explicit ScrambledZipfianGenerator(int num_keys, size_t *seed = nullptr, double skew=0.99)
//       : num_keys_(num_keys), zipfian_constant_(skew), gen_(std::random_device{}()), dis_(0, 1) {
//     if(seed) {
//       gen_.seed(*seed);
//     }
//     zeta_n_ = zeta(num_keys_, zipfian_constant_);
//     double zeta2theta = zeta(2, zipfian_constant_);
//     alpha_ = 1. / (1. - zipfian_constant_);

//     eta_ = (1 - std::pow(2. / num_keys_, 1 - zipfian_constant_)) /
//            (1 - zeta2theta / zeta_n_);
//   }

//   int nextValue() {
//     double u = dis_(gen_);
//     double uz = u * zeta_n_;

//     int ret;
//     if (uz < 1.0) {
//       ret = 0;
//     } else if (uz < 1.0 + std::pow(0.5, zipfian_constant_)) {
//       ret = 1;
//     } else {
//       ret = (int)(num_keys_ * std::pow(eta_ * u - eta_ + 1, alpha_));
//     }

//     ret = fnv1a(ret) % num_keys_;
//     return ret;
//   }

//   double zeta(long n, double theta) {
//     double sum = 0.0;
//     for (long i = 0; i < n; i++) {
//       sum += 1 / std::pow(i + 1, theta);
//     }
//     return sum;
//   }

//   // FNV hash from https://create.stephan-brumme.com/fnv-hash/
//   static const uint32_t PRIME = 0x01000193;  //   16777619
//   static const uint32_t SEED = 0x811C9DC5;   // 2166136261
//   /// hash a single byte
//   inline uint32_t fnv1a(unsigned char oneByte, uint32_t hash = SEED) {
//     return (oneByte ^ hash) * PRIME;
//   }
//   /// hash a 32 bit integer (four bytes)
//   inline uint32_t fnv1a(int fourBytes, uint32_t hash = SEED) {
//     const unsigned char* ptr = (const unsigned char*)&fourBytes;
//     hash = fnv1a(*ptr++, hash);
//     hash = fnv1a(*ptr++, hash);
//     hash = fnv1a(*ptr++, hash);
//     return fnv1a(*ptr, hash);
//   }
// };

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

class ScrambledZipfianGenerator {
 public:
  int num_keys_;
  double zipfian_constant_;  // now true Zipf alpha
  double zeta_n_;            // H_{N,alpha}

  // kept for interface compatibility (unused now)
  double alpha_;
  double eta_;

  std::mt19937_64 gen_;
  std::uniform_real_distribution<double> dis_;

  std::vector<double> cdf_;  // cumulative distribution

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

    // binary search
    auto it = std::lower_bound(cdf_.begin(), cdf_.end(), u);
    int rank = std::distance(cdf_.begin(), it);

    if (rank >= num_keys_) {
      rank = num_keys_ - 1;
    }

    // scramble
    int ret = fnv1a(rank) % num_keys_;
    return ret;
  }

 private:

  void buildCDF() {
    cdf_.resize(num_keys_);

    // Special case: alpha == 0 → uniform
    if (zipfian_constant_ == 0.0) {
      for (int i = 0; i < num_keys_; ++i) {
        cdf_[i] = double(i + 1) / num_keys_;
      }
      zeta_n_ = 1.0;
      return;
    }

    // compute normalization constant H_{N,alpha}
    zeta_n_ = 0.0;
    for (int i = 1; i <= num_keys_; ++i) {
      zeta_n_ += 1.0 / std::pow(i, zipfian_constant_);
    }

    // build cumulative
    double cumulative = 0.0;
    for (int i = 1; i <= num_keys_; ++i) {
      cumulative += (1.0 / std::pow(i, zipfian_constant_)) / zeta_n_;
      cdf_[i - 1] = cumulative;
    }

    // force last element to exactly 1.0
    cdf_[num_keys_ - 1] = 1.0;
  }

 public:

  // ---- FNV hash ----

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

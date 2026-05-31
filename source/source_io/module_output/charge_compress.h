#ifndef CHARGE_COMPRESS_H
#define CHARGE_COMPRESS_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ModuleIO
{

constexpr uint32_t CHARGE_COMPRESS_MAGIC = 0x504D435A; // "ZCMP" little-endian
constexpr size_t CHARGE_COMPRESS_HEADER_SIZE = 12;     // 4B magic + 8B count

// Serial compression via zlib
bool compress_charge_data(const double* src, size_t n, std::vector<uint8_t>& dst);

bool decompress_charge_data(const uint8_t* src, size_t src_len, double* dst, size_t n);

// OpenMP-parallel compression. nthreads=0 means use omp_get_max_threads().
// Wire format for parallel: [magic:4B] [nthreads:4B] [total_n:8B]
//   then per chunk: [chunk_size:4B] [chunk_data...]
bool compress_charge_data_omp(const double* src, size_t n,
                              std::vector<uint8_t>& dst, int nthreads = 0);

bool decompress_charge_data_omp(const uint8_t* src, size_t src_len,
                                double* dst, size_t n);

} // namespace ModuleIO

#endif

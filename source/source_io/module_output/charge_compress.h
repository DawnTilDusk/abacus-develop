#ifndef CHARGE_COMPRESS_H
#define CHARGE_COMPRESS_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ModuleIO
{

// "ZCMP" little-endian — legacy serial / OMP parallel wire format (v0)
constexpr uint32_t CHARGE_COMPRESS_MAGIC = 0x504D435A;
// "ZCM2" little-endian — improved wire format (v1) with flags byte & fallback support
constexpr uint32_t CHARGE_COMPRESS_MAGIC_V1 = 0x324D435A;

// v0 header sizes
constexpr size_t CHARGE_COMPRESS_HEADER_SIZE = 12; // magic:4B + count:8B  (serial v0)
constexpr size_t OMP_HEADER_SIZE_V0        = 16; // magic:4B + nthreads:4B + total_n:8B (OMP v0)

// v1 header: magic(4B) + fmt_type(1B) + flags(1B) + ...
//   serial: ... + total_n(8B) = 14 bytes
//   OMP:    ... + nthreads(2B) + total_n(8B) = 16 bytes
constexpr size_t V1_SERIAL_HEADER_SIZE = 14; // magic:4B + fmt:1B + flags:1B + total_n:8B
constexpr size_t V1_OMP_HEADER_SIZE    = 16; // magic:4B + fmt:1B + flags:1B + nthreads:2B + total_n:8B

// Format type byte values (v1 format, offset 4)
constexpr uint8_t V1_FMT_SERIAL = 1;
constexpr uint8_t V1_FMT_OMP    = 2;

// Flags byte bits (v1 format, offset 5)
constexpr uint8_t COMPRESS_FLAG_COMPRESSED = 0x01; // 1 = zlib compressed, 0 = raw store

// ---------------------------------------------------------------------------
// Serial compression / decompression (zlib, v0 wire format — kept for compat)
// ---------------------------------------------------------------------------

/// Compress `n` doubles via zlib. Output: legacy v0 serial wire format.
bool compress_charge_data(const double* src, size_t n, std::vector<uint8_t>& dst);

/// Decompress legacy v0 serial format.
bool decompress_charge_data(const uint8_t* src, size_t src_len, double* dst, size_t n);

// ---------------------------------------------------------------------------
// OpenMP-parallel compression / decompression (v0 wire format, kept for compat)
// ---------------------------------------------------------------------------

/// Compress in parallel chunks. nthreads=0 → omp_get_max_threads().
/// Output: legacy v0 OMP wire format.
bool compress_charge_data_omp(const double* src, size_t n,
                              std::vector<uint8_t>& dst, int nthreads = 0);

/// Decompress legacy v0 OMP format (also falls back to legacy serial).
bool decompress_charge_data_omp(const uint8_t* src, size_t src_len,
                                double* dst, size_t n);

// ---------------------------------------------------------------------------
// v1 wire format — adds flags byte for raw-store fallback
// ---------------------------------------------------------------------------

/// Compress with auto-fallback: if compressed size >= raw size, store raw.
/// Output: v1 wire format (magic "ZCM2").
bool compress_charge_data_v1(const double* src, size_t n, std::vector<uint8_t>& dst);

/// OMP compress with auto-fallback. Output: v1 wire format (OMP header).
bool compress_charge_data_omp_v1(const double* src, size_t n,
                                 std::vector<uint8_t>& dst, int nthreads = 0);

/// Decompress v0 or v1 format (auto-detect via magic). Thread-safe.
bool decompress_charge_data_any(const uint8_t* src, size_t src_len,
                                double* dst, size_t n);

} // namespace ModuleIO

#endif
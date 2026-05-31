#include "charge_compress.h"

#include <algorithm>
#include <cstring>

#include <zlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ModuleIO
{

// ===========================================================================
// Serial compression / decompression
// ===========================================================================

bool compress_charge_data(const double* src, size_t n, std::vector<uint8_t>& dst)
{
    dst.clear();
    dst.resize(CHARGE_COMPRESS_HEADER_SIZE);

    std::memcpy(dst.data(), &CHARGE_COMPRESS_MAGIC, 4);

    uint64_t count = static_cast<uint64_t>(n);
    std::memcpy(dst.data() + 4, &count, 8);

    uLongf src_len = static_cast<uLongf>(n * sizeof(double));
    uLongf bound = compressBound(src_len);
    std::vector<uint8_t> cbuf(bound);

    int ret = compress2(cbuf.data(), &bound,
                        reinterpret_cast<const Bytef*>(src), src_len,
                        Z_BEST_COMPRESSION);
    if (ret != Z_OK)
        return false;

    dst.insert(dst.end(), cbuf.begin(), cbuf.begin() + bound);
    return true;
}

bool decompress_charge_data(const uint8_t* src, size_t src_len,
                            double* dst, size_t n)
{
    if (src_len < CHARGE_COMPRESS_HEADER_SIZE)
        return false;

    uint32_t magic = 0;
    std::memcpy(&magic, src, 4);
    if (magic != CHARGE_COMPRESS_MAGIC)
        return false;

    uint64_t count = 0;
    std::memcpy(&count, src + 4, 8);
    if (count != static_cast<uint64_t>(n))
        return false;

    uLongf dst_len = static_cast<uLongf>(n * sizeof(double));
    int ret = uncompress(reinterpret_cast<Bytef*>(dst), &dst_len,
                         src + CHARGE_COMPRESS_HEADER_SIZE,
                         src_len - CHARGE_COMPRESS_HEADER_SIZE);
    return ret == Z_OK;
}

// ===========================================================================
// OpenMP-parallel compression / decompression
// ===========================================================================

// Parallel wire format:
//   [magic:4B "ZCMP"] [nthreads:4B] [total_n:8B]
//   for each thread t:
//     [chunk_compressed_size:4B] [chunk_data...]

static constexpr size_t OMP_HEADER_SIZE = 16; // 4B magic + 4B nthreads + 8B total_n

bool compress_charge_data_omp(const double* src, size_t n,
                              std::vector<uint8_t>& dst, int nthreads)
{
#ifdef _OPENMP
    if (nthreads <= 0)
        nthreads = omp_get_max_threads();
#else
    nthreads = 1;
#endif

    if (nthreads <= 1)
        return compress_charge_data(src, n, dst);

    // Split data into chunks
    size_t chunk_n = n / nthreads;
    size_t remainder = n % nthreads;

    // Per-thread compressed buffers
    std::vector<std::vector<uint8_t>> chunk_bufs(nthreads);
    std::vector<uLongf> chunk_sizes(nthreads);

    bool all_ok = true;

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int t = 0; t < nthreads; ++t)
    {
        size_t start = t * chunk_n + std::min<size_t>(t, remainder);
        size_t count = chunk_n + (t < static_cast<int>(remainder) ? 1 : 0);

        uLongf src_len = static_cast<uLongf>(count * sizeof(double));
        uLongf bound = compressBound(src_len);
        chunk_bufs[t].resize(bound);

        int ret = compress2(chunk_bufs[t].data(), &bound,
                            reinterpret_cast<const Bytef*>(src + start), src_len,
                            Z_BEST_COMPRESSION);
        if (ret != Z_OK)
            all_ok = false;

        chunk_sizes[t] = bound;
    }

    if (!all_ok)
        return false;

    // Build the parallel wire format
    dst.clear();
    dst.resize(OMP_HEADER_SIZE);

    std::memcpy(dst.data(), &CHARGE_COMPRESS_MAGIC, 4);
    uint32_t nt = static_cast<uint32_t>(nthreads);
    std::memcpy(dst.data() + 4, &nt, 4);
    uint64_t total = static_cast<uint64_t>(n);
    std::memcpy(dst.data() + 8, &total, 8);

    for (int t = 0; t < nthreads; ++t)
    {
        uint32_t csize = static_cast<uint32_t>(chunk_sizes[t]);
        size_t old_size = dst.size();
        dst.resize(old_size + 4 + csize);
        std::memcpy(dst.data() + old_size, &csize, 4);
        std::memcpy(dst.data() + old_size + 4, chunk_bufs[t].data(), csize);
    }

    return true;
}

bool decompress_charge_data_omp(const uint8_t* src, size_t src_len,
                                double* dst, size_t n)
{
    if (src_len < OMP_HEADER_SIZE)
        return false;

    uint32_t magic = 0;
    std::memcpy(&magic, src, 4);
    if (magic != CHARGE_COMPRESS_MAGIC)
        return false;

    uint32_t nthreads = 0;
    std::memcpy(&nthreads, src + 4, 4);

    // If nthreads == 0 (old format: magic + count = 12 bytes), fall back to serial
    if (nthreads == 0)
        return decompress_charge_data(src, src_len, dst, n);

    uint64_t total_n = 0;
    std::memcpy(&total_n, src + 8, 8);
    if (total_n != static_cast<uint64_t>(n))
        return false;

    size_t pos = OMP_HEADER_SIZE;
    size_t chunk_n = n / nthreads;
    size_t remainder = n % nthreads;

    for (uint32_t t = 0; t < nthreads; ++t)
    {
        if (pos + 4 > src_len)
            return false;

        uint32_t csize = 0;
        std::memcpy(&csize, src + pos, 4);
        pos += 4;

        if (pos + csize > src_len)
            return false;

        size_t chunk_start = t * chunk_n + std::min<size_t>(t, remainder);
        size_t chunk_count = chunk_n + (t < static_cast<size_t>(remainder) ? 1 : 0);

        uLongf dst_len = static_cast<uLongf>(chunk_count * sizeof(double));
        int ret = uncompress(reinterpret_cast<Bytef*>(dst + chunk_start), &dst_len,
                             src + pos, csize);
        if (ret != Z_OK)
            return false;

        pos += csize;
    }

    return true;
}

} // namespace ModuleIO

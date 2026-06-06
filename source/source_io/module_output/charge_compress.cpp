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
// Serial compression / decompression  (v0 legacy wire format)
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
// OpenMP-parallel compression / decompression  (v0 legacy wire format)
// ===========================================================================

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
    std::vector<bool> chunk_ok(nthreads, true);

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
            chunk_ok[t] = false;

        chunk_sizes[t] = bound;
    }

    for (int t = 0; t < nthreads; ++t)
        if (!chunk_ok[t])
            return false;

    // Build the OMP v0 wire format
    dst.clear();
    dst.resize(OMP_HEADER_SIZE_V0);

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
    if (src_len < OMP_HEADER_SIZE_V0)
        return false;

    uint32_t magic = 0;
    std::memcpy(&magic, src, 4);

    // If magic doesn't match legacy ZCMP, try the generic any-format decompressor
    if (magic != CHARGE_COMPRESS_MAGIC)
        return decompress_charge_data_any(src, src_len, dst, n);

    uint32_t nthreads = 0;
    std::memcpy(&nthreads, src + 4, 4);

    // nthreads == 0 → legacy serial format (magic + count)
    if (nthreads == 0)
        return decompress_charge_data(src, src_len, dst, n);

    uint64_t total_n = 0;
    std::memcpy(&total_n, src + 8, 8);
    if (total_n != static_cast<uint64_t>(n))
        return false;

    size_t pos = OMP_HEADER_SIZE_V0;
    size_t chunk_n = n / nthreads;
    size_t remainder = n % nthreads;

    // Each thread computes its own chunk boundaries.
    // Since `pos` depends on variable chunk sizes, we do a serial pre-scan
    // to record each chunk's offset and size, then decompress in parallel.
    std::vector<size_t> chunk_offsets(nthreads);
    std::vector<uint32_t> chunk_csizes(nthreads);
    for (uint32_t t = 0; t < nthreads; ++t)
    {
        if (pos + 4 > src_len)
            return false;
        uint32_t csize = 0;
        std::memcpy(&csize, src + pos, 4);
        pos += 4;
        if (pos + csize > src_len)
            return false;
        chunk_offsets[t] = pos;
        chunk_csizes[t] = csize;
        pos += csize;
    }

    bool all_ok = true;

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (uint32_t t = 0; t < nthreads; ++t)
    {
        size_t chunk_start = t * chunk_n + std::min<size_t>(t, remainder);
        size_t chunk_count = chunk_n + (t < static_cast<size_t>(remainder) ? 1 : 0);

        uLongf dst_len = static_cast<uLongf>(chunk_count * sizeof(double));
        int ret = uncompress(reinterpret_cast<Bytef*>(dst + chunk_start), &dst_len,
                             src + chunk_offsets[t], chunk_csizes[t]);
        if (ret != Z_OK)
            all_ok = false;
    }

    return all_ok;
}

// ===========================================================================
// v1 wire format — flags byte + raw-store fallback
// ===========================================================================

// Forward declarations for internal use
static bool decompress_v1(const uint8_t* src, size_t src_len, double* dst, size_t n);
static bool decompress_v1_omp(const uint8_t* src, size_t src_len, double* dst, size_t n);

bool compress_charge_data_v1(const double* src, size_t n, std::vector<uint8_t>& dst)
{
    // First, try compressing using v0 (gives us compressed payload)
    std::vector<uint8_t> cbuf;
    if (!compress_charge_data(src, n, cbuf))
        return false;

    // cbuf has legacy v0 format: magic(4B) + count(8B) + payload
    size_t raw_payload_size = n * sizeof(double);
    size_t compressed_payload_size = cbuf.size() - CHARGE_COMPRESS_HEADER_SIZE;
    uint8_t flags = 0;

    dst.clear();
    dst.resize(V1_SERIAL_HEADER_SIZE);

    std::memcpy(dst.data(), &CHARGE_COMPRESS_MAGIC_V1, 4);
    dst[4] = V1_FMT_SERIAL;  // format type: serial

    if (compressed_payload_size < raw_payload_size)
    {
        // Compression helped — store compressed
        flags = COMPRESS_FLAG_COMPRESSED;
        dst[5] = flags;
        uint64_t total = static_cast<uint64_t>(n);
        std::memcpy(dst.data() + 6, &total, 8);

        // Append compressed payload (skip the old v0 header)
        dst.insert(dst.end(),
                   cbuf.begin() + CHARGE_COMPRESS_HEADER_SIZE,
                   cbuf.end());
    }
    else
    {
        // Compression didn't help — store raw
        flags = 0; // uncompressed
        dst[5] = flags;
        uint64_t total = static_cast<uint64_t>(n);
        std::memcpy(dst.data() + 6, &total, 8);

        // Append raw data directly
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(src);
        dst.insert(dst.end(), raw, raw + raw_payload_size);
    }

    return true;
}

bool compress_charge_data_omp_v1(const double* src, size_t n,
                                 std::vector<uint8_t>& dst, int nthreads)
{
#ifdef _OPENMP
    if (nthreads <= 0)
        nthreads = omp_get_max_threads();
#else
    nthreads = 1;
#endif

    if (nthreads <= 1)
        return compress_charge_data_v1(src, n, dst);

    // Compress using OMP v0 first, then repack with v1 header
    std::vector<uint8_t> cbuf;
    if (!compress_charge_data_omp(src, n, cbuf, nthreads))
        return false;

    size_t raw_payload_size = n * sizeof(double);
    size_t compressed_payload_size = cbuf.size() - OMP_HEADER_SIZE_V0;
    uint8_t flags = 0;

    dst.clear();
    dst.resize(V1_OMP_HEADER_SIZE);

    std::memcpy(dst.data(), &CHARGE_COMPRESS_MAGIC_V1, 4);
    dst[4] = V1_FMT_OMP;  // format type: OMP

    if (compressed_payload_size < raw_payload_size)
    {
        // Compression helped
        flags = COMPRESS_FLAG_COMPRESSED;
        dst[5] = flags;
        uint16_t nt = static_cast<uint16_t>(nthreads);
        std::memcpy(dst.data() + 6, &nt, 2);
        uint64_t total = static_cast<uint64_t>(n);
        std::memcpy(dst.data() + 8, &total, 8);

        // Append payload (chunk sizes + compressed data) from v0, excluding v0 header
        dst.insert(dst.end(),
                   cbuf.begin() + OMP_HEADER_SIZE_V0,
                   cbuf.end());
    }
    else
    {
        // Compression didn't help — store raw chunks
        flags = 0; // uncompressed
        dst[5] = flags;
        uint16_t nt = static_cast<uint16_t>(nthreads);
        std::memcpy(dst.data() + 6, &nt, 2);
        uint64_t total = static_cast<uint64_t>(n);
        std::memcpy(dst.data() + 8, &total, 8);

        // Store raw data in chunks matching the OMP partition scheme
        size_t chunk_n = n / nthreads;
        size_t remainder = n % nthreads;
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(src);

        for (int t = 0; t < nthreads; ++t)
        {
            size_t start = t * chunk_n + std::min<size_t>(t, remainder);
            size_t count = chunk_n + (t < static_cast<int>(remainder) ? 1 : 0);
            size_t byte_count = count * sizeof(double);
            uint32_t csize = static_cast<uint32_t>(byte_count);

            size_t old_sz = dst.size();
            dst.resize(old_sz + 4 + byte_count);
            std::memcpy(dst.data() + old_sz, &csize, 4);
            std::memcpy(dst.data() + old_sz + 4, raw + start * sizeof(double), byte_count);
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Internal: decompress v1 serial format
// ---------------------------------------------------------------------------

static bool decompress_v1(const uint8_t* src, size_t src_len, double* dst, size_t n)
{
    if (src_len < V1_SERIAL_HEADER_SIZE)
        return false;

    // Verify format type byte
    if (src[4] != V1_FMT_SERIAL)
        return false;

    uint8_t flags = src[5];
    uint64_t total_n = 0;
    std::memcpy(&total_n, src + 6, 8);
    if (total_n != static_cast<uint64_t>(n))
        return false;

    const uint8_t* payload = src + V1_SERIAL_HEADER_SIZE;
    size_t payload_len = src_len - V1_SERIAL_HEADER_SIZE;

    if (flags & COMPRESS_FLAG_COMPRESSED)
    {
        // Compressed payload — uncompress
        uLongf dst_len = static_cast<uLongf>(n * sizeof(double));
        int ret = uncompress(reinterpret_cast<Bytef*>(dst), &dst_len,
                             payload, payload_len);
        return ret == Z_OK;
    }
    else
    {
        // Raw payload — direct copy
        if (payload_len < n * sizeof(double))
            return false;
        std::memcpy(dst, payload, n * sizeof(double));
        return true;
    }
}

// ---------------------------------------------------------------------------
// Internal: decompress v1 OMP format (with OpenMP parallelism)
// ---------------------------------------------------------------------------

static bool decompress_v1_omp(const uint8_t* src, size_t src_len, double* dst, size_t n)
{
    if (src_len < V1_OMP_HEADER_SIZE)
        return false;

    // Verify format type byte
    if (src[4] != V1_FMT_OMP)
        return false;

    uint8_t flags = src[5];
    uint16_t nthreads = 0;
    std::memcpy(&nthreads, src + 6, 2);
    uint64_t total_n = 0;
    std::memcpy(&total_n, src + 8, 8);
    if (total_n != static_cast<uint64_t>(n))
        return false;

    // Pre-scan chunk offsets so decompression can be parallel
    size_t pos = V1_OMP_HEADER_SIZE;
    std::vector<size_t> chunk_offsets(nthreads);
    std::vector<uint32_t> chunk_csizes(nthreads);
    for (uint16_t s = 0; s < nthreads; ++s)
    {
        if (pos + 4 > src_len)
            return false;
        uint32_t csize = 0;
        std::memcpy(&csize, src + pos, 4);
        pos += 4;
        if (pos + csize > src_len)
            return false;
        chunk_offsets[s] = pos;
        chunk_csizes[s] = csize;
        pos += csize;
    }

    size_t chunk_n = n / nthreads;
    size_t remainder = n % nthreads;

    if (flags & COMPRESS_FLAG_COMPRESSED)
    {
        // Compressed chunks — uncompress each in parallel
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (uint16_t t = 0; t < nthreads; ++t)
        {
            size_t chunk_start = t * chunk_n + std::min<size_t>(t, remainder);
            size_t chunk_count = chunk_n + (t < static_cast<size_t>(remainder) ? 1 : 0);

            uLongf dst_len = static_cast<uLongf>(chunk_count * sizeof(double));
            uncompress(reinterpret_cast<Bytef*>(dst + chunk_start), &dst_len,
                       src + chunk_offsets[t], chunk_csizes[t]);
        }
        return true;
    }
    else
    {
        // Raw chunks — direct copy each in parallel
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (uint16_t t = 0; t < nthreads; ++t)
        {
            size_t chunk_start = t * chunk_n + std::min<size_t>(t, remainder);
            size_t chunk_count = chunk_n + (t < static_cast<size_t>(remainder) ? 1 : 0);

            std::memcpy(dst + chunk_start, src + chunk_offsets[t],
                        chunk_count * sizeof(double));
        }
        return true;
    }
}

// ===========================================================================
// Universal decompress: auto-detect v0 serial, v0 OMP, v1 serial, v1 OMP
// ===========================================================================

bool decompress_charge_data_any(const uint8_t* src, size_t src_len,
                                double* dst, size_t n)
{
    if (src_len < 4)
        return false;

    uint32_t magic = 0;
    std::memcpy(&magic, src, 4);

    if (magic == CHARGE_COMPRESS_MAGIC_V1)
    {
        // v1 format — dispatch on format type byte (offset 4)
        if (src_len < 5)
            return false;
        uint8_t fmt = src[4];
        if (fmt == V1_FMT_OMP)
            return decompress_v1_omp(src, src_len, dst, n);
        else
            return decompress_v1(src, src_len, dst, n);
    }

    if (magic == CHARGE_COMPRESS_MAGIC)
    {
        // v0 legacy format — try OMP first (nthreads != 0), then serial
        if (src_len >= OMP_HEADER_SIZE_V0)
        {
            uint32_t nthreads = 0;
            std::memcpy(&nthreads, src + 4, 4);
            if (nthreads > 1)
                return decompress_charge_data_omp(src, src_len, dst, n);
        }
        return decompress_charge_data(src, src_len, dst, n);
    }

    return false;
}

} // namespace ModuleIO
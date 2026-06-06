#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
struct AxisInterpolationMap
{
    int low = 0;
    int high = 0;
    double w_low = 1.0;
    double w_high = 0.0;
};

std::vector<AxisInterpolationMap> build_axis_interpolation_map(const int src_size, const int dst_size)
{
    std::vector<AxisInterpolationMap> axis_map(dst_size);
    if (src_size <= 1)
    {
        for (auto& item : axis_map)
        {
            item.low = 0;
            item.high = 0;
            item.w_low = 1.0;
            item.w_high = 0.0;
        }
        return axis_map;
    }

    const double scale = static_cast<double>(src_size) / static_cast<double>(dst_size);
    const double period = static_cast<double>(src_size);
    for (int i = 0; i < dst_size; ++i)
    {
        double frac = 0.5 * (scale * (1.0 + 2.0 * i) - 1.0);
        frac -= std::floor(frac / period) * period;

        const int low = static_cast<int>(frac);
        const double delta = frac - static_cast<double>(low);

        axis_map[i].low = low;
        axis_map[i].high = (low + 1 == src_size) ? 0 : (low + 1);
        axis_map[i].w_low = 1.0 - delta;
        axis_map[i].w_high = delta;
    }
    return axis_map;
}

struct RefAxisMap
{
    int low = 0;
    int high = 0;
    double w_low = 1.0;
    double w_high = 0.0;
};

RefAxisMap build_ref_axis_map(const int src_size, const int dst_size, const int dst_index)
{
    if (src_size <= 1)
    {
        return {0, 0, 1.0, 0.0};
    }

    double frac = 0.5 * (static_cast<double>(src_size) / static_cast<double>(dst_size) * (1.0 + 2.0 * dst_index) - 1.0);
    frac -= std::floor(frac / static_cast<double>(src_size)) * static_cast<double>(src_size);

    const int low = static_cast<int>(std::floor(frac));
    const double delta = frac - static_cast<double>(low);
    return {low, (low + 1 == src_size) ? 0 : (low + 1), 1.0 - delta, delta};
}

double trilinear_reference_value(const double* const data_in,
                                 const int nx_read,
                                 const int ny_read,
                                 const int nz_read,
                                 const int nx,
                                 const int ny,
                                 const int nz,
                                 const int ix,
                                 const int iy,
                                 const int iz)
{
    const RefAxisMap x_map = build_ref_axis_map(nx_read, nx, ix);
    const RefAxisMap y_map = build_ref_axis_map(ny_read, ny, iy);
    const RefAxisMap z_map = build_ref_axis_map(nz_read, nz, iz);

    const auto index = [=](const int x, const int y, const int z) {
        return (x * ny_read + y) * nz_read + z;
    };

    return data_in[index(x_map.low, y_map.low, z_map.low)] * x_map.w_low * y_map.w_low * z_map.w_low
         + data_in[index(x_map.high, y_map.low, z_map.low)] * x_map.w_high * y_map.w_low * z_map.w_low
         + data_in[index(x_map.low, y_map.high, z_map.low)] * x_map.w_low * y_map.w_high * z_map.w_low
         + data_in[index(x_map.low, y_map.low, z_map.high)] * x_map.w_low * y_map.w_low * z_map.w_high
         + data_in[index(x_map.high, y_map.high, z_map.low)] * x_map.w_high * y_map.w_high * z_map.w_low
         + data_in[index(x_map.high, y_map.low, z_map.high)] * x_map.w_high * y_map.w_low * z_map.w_high
         + data_in[index(x_map.low, y_map.high, z_map.high)] * x_map.w_low * y_map.w_high * z_map.w_high
         + data_in[index(x_map.high, y_map.high, z_map.high)] * x_map.w_high * y_map.w_high * z_map.w_high;
}

double max_abs_error_against_reference(const std::vector<double>& data_in,
                                       const int nx_read,
                                       const int ny_read,
                                       const int nz_read,
                                       const int nx,
                                       const int ny,
                                       const int nz,
                                       const std::vector<double>& actual)
{
    double max_err = 0.0;
    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            for (int iz = 0; iz < nz; ++iz)
            {
                const double ref =
                    trilinear_reference_value(data_in.data(), nx_read, ny_read, nz_read, nx, ny, nz, ix, iy, iz);
                const double value = actual[(ix * ny + iy) * nz + iz];
                max_err = std::max(max_err, std::abs(value - ref));
            }
        }
    }
    return max_err;
}

std::vector<double> make_interp_input(const int nx, const int ny, const int nz)
{
    std::vector<double> data(static_cast<std::size_t>(nx) * ny * nz, 0.0);
    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            for (int iz = 0; iz < nz; ++iz)
            {
                const double x = static_cast<double>(ix + 1);
                const double y = static_cast<double>(iy + 2);
                const double z = static_cast<double>(iz + 3);
                data[(static_cast<std::size_t>(ix) * ny + iy) * nz + iz] =
                    std::sin(0.17 * x) + std::cos(0.23 * y) + 0.5 * std::sin(0.31 * z) + 0.01 * x * y;
            }
        }
    }
    return data;
}

void trilinear_interpolate_current(const double* const data_in,
                                   const int nx_read,
                                   const int ny_read,
                                   const int nz_read,
                                   const int nx,
                                   const int ny,
                                   const int nz,
                                   double* data_out)
{
    const std::vector<AxisInterpolationMap> x_map = build_axis_interpolation_map(nx_read, nx);
    const std::vector<AxisInterpolationMap> y_map = build_axis_interpolation_map(ny_read, ny);
    const std::vector<AxisInterpolationMap> z_map = build_axis_interpolation_map(nz_read, nz);

#if defined(_OPENMP) && !defined(TRILINEAR_DISABLE_OMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            const AxisInterpolationMap& x_item = x_map[ix];
            const AxisInterpolationMap& y_item = y_map[iy];

            const int idx_x0y0 = (x_item.low * ny_read + y_item.low) * nz_read;
            const int idx_x1y0 = (x_item.high * ny_read + y_item.low) * nz_read;
            const int idx_x0y1 = (x_item.low * ny_read + y_item.high) * nz_read;
            const int idx_x1y1 = (x_item.high * ny_read + y_item.high) * nz_read;

            const double w00 = x_item.w_low * y_item.w_low;
            const double w10 = x_item.w_high * y_item.w_low;
            const double w01 = x_item.w_low * y_item.w_high;
            const double w11 = x_item.w_high * y_item.w_high;

            double* const out_row = data_out + (static_cast<std::size_t>(ix) * ny + iy) * nz;

#if defined(_OPENMP) && !defined(TRILINEAR_DISABLE_SIMD)
#pragma omp simd
#endif
            for (int iz = 0; iz < nz; ++iz)
            {
                const AxisInterpolationMap& z_item = z_map[iz];
                const int lowz = z_item.low;
                const int highz = z_item.high;

                const double low_plane = data_in[idx_x0y0 + lowz] * w00
                                         + data_in[idx_x1y0 + lowz] * w10
                                         + data_in[idx_x0y1 + lowz] * w01
                                         + data_in[idx_x1y1 + lowz] * w11;
                const double high_plane = data_in[idx_x0y0 + highz] * w00
                                          + data_in[idx_x1y0 + highz] * w10
                                          + data_in[idx_x0y1 + highz] * w01
                                          + data_in[idx_x1y1 + highz] * w11;

                out_row[iz] = low_plane * z_item.w_low + high_plane * z_item.w_high;
            }
        }
    }
}

struct BenchCase
{
    std::string name;
    int nx_read;
    int ny_read;
    int nz_read;
    int nx;
    int ny;
    int nz;
    int warmup;
    int iterations;
};

struct BenchResult
{
    double median_ms = 0.0;
    double checksum = 0.0;
};

BenchResult run_case(const BenchCase& bench_case, const int threads)
{
    const std::vector<double> data_in =
        make_interp_input(bench_case.nx_read, bench_case.ny_read, bench_case.nz_read);
    std::vector<double> data_out(
        static_cast<std::size_t>(bench_case.nx) * bench_case.ny * bench_case.nz,
        0.0);
    std::vector<double> samples;
    samples.reserve(bench_case.iterations);

#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif

    for (int i = 0; i < bench_case.warmup + bench_case.iterations; ++i)
    {
        std::fill(data_out.begin(), data_out.end(), 0.0);
        const auto begin = std::chrono::steady_clock::now();
        trilinear_interpolate_current(data_in.data(),
                                      bench_case.nx_read,
                                      bench_case.ny_read,
                                      bench_case.nz_read,
                                      bench_case.nx,
                                      bench_case.ny,
                                      bench_case.nz,
                                      data_out.data());
        const auto end = std::chrono::steady_clock::now();

        if (i >= bench_case.warmup)
        {
            const std::chrono::duration<double, std::milli> elapsed = end - begin;
            samples.push_back(elapsed.count());
        }
    }

    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    const double median_ms = (samples.size() % 2 == 0)
                                 ? 0.5 * (samples[middle - 1] + samples[middle])
                                 : samples[middle];
    const double checksum = std::accumulate(data_out.begin(), data_out.end(), 0.0);
    return {median_ms, checksum};
}

void print_usage(const char* argv0)
{
    std::cout << "Usage: " << argv0 << " [--threads 1,2,4,8]\n";
}

std::vector<int> parse_threads(int argc, char** argv)
{
    std::vector<int> threads = {1, 2, 4, 8};
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc)
        {
            threads.clear();
            std::string values = argv[++i];
            std::size_t start = 0;
            while (start < values.size())
            {
                const std::size_t comma = values.find(',', start);
                const std::string token =
                    values.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if (!token.empty())
                {
                    threads.push_back(std::max(1, std::stoi(token)));
                }
                if (comma == std::string::npos)
                {
                    break;
                }
                start = comma + 1;
            }
        }
        else if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
    }
    return threads;
}

double working_set_mb(const BenchCase& bench_case)
{
    const double input_bytes =
        static_cast<double>(bench_case.nx_read) * bench_case.ny_read * bench_case.nz_read * sizeof(double);
    const double output_bytes =
        static_cast<double>(bench_case.nx) * bench_case.ny * bench_case.nz * sizeof(double);
    return (input_bytes + output_bytes) / (1024.0 * 1024.0);
}

double traffic_gb_per_s(const BenchCase& bench_case, const double median_ms)
{
    const double output_points = static_cast<double>(bench_case.nx) * bench_case.ny * bench_case.nz;
    const double traffic_bytes = output_points * (8.0 * sizeof(double) + sizeof(double));
    return traffic_bytes / (median_ms / 1000.0) / 1.0e9;
}
} // namespace

int main(int argc, char** argv)
{
    const std::vector<int> threads = parse_threads(argc, argv);
    const std::vector<BenchCase> cases = {
        {"l3_fit", 64, 64, 64, 128, 128, 128, 1, 7},
        {"l3_exceed", 96, 96, 96, 192, 192, 192, 1, 5},
    };

    std::cout << "TRILINEAR_BENCH"
              << " openmp="
#ifdef _OPENMP
              << "on"
#else
              << "off"
#endif
              << " simd="
#ifdef TRILINEAR_DISABLE_SIMD
              << "off"
#else
              << "on"
#endif
              << '\n';

#ifdef _OPENMP
    std::cout << "omp_max_threads=" << omp_get_max_threads() << '\n';
#endif

    for (const BenchCase& bench_case : cases)
    {
        const std::vector<double> input = make_interp_input(bench_case.nx_read, bench_case.ny_read, bench_case.nz_read);
        std::vector<double> output(static_cast<std::size_t>(bench_case.nx) * bench_case.ny * bench_case.nz, 0.0);
        trilinear_interpolate_current(input.data(),
                                      bench_case.nx_read,
                                      bench_case.ny_read,
                                      bench_case.nz_read,
                                      bench_case.nx,
                                      bench_case.ny,
                                      bench_case.nz,
                                      output.data());
        const double max_err = max_abs_error_against_reference(input,
                                                               bench_case.nx_read,
                                                               bench_case.ny_read,
                                                               bench_case.nz_read,
                                                               bench_case.nx,
                                                               bench_case.ny,
                                                               bench_case.nz,
                                                               output);

        std::cout << "CASE " << bench_case.name
                  << " src=" << bench_case.nx_read << 'x' << bench_case.ny_read << 'x' << bench_case.nz_read
                  << " dst=" << bench_case.nx << 'x' << bench_case.ny << 'x' << bench_case.nz
                  << " working_set_mb=" << std::fixed << std::setprecision(2) << working_set_mb(bench_case)
                  << " max_err=" << std::scientific << max_err << '\n';

        double baseline_ms = 0.0;
        for (const int thread_count : threads)
        {
            const BenchResult result = run_case(bench_case, thread_count);
            if (thread_count == threads.front())
            {
                baseline_ms = result.median_ms;
            }

            std::cout << std::fixed << std::setprecision(3)
                      << "RESULT case=" << bench_case.name
                      << " threads=" << thread_count
                      << " median_ms=" << result.median_ms
                      << " speedup=" << (baseline_ms / result.median_ms)
                      << " est_gbps=" << traffic_gb_per_s(bench_case, result.median_ms)
                      << " checksum=" << std::setprecision(8) << result.checksum
                      << '\n';
        }
    }

    return 0;
}

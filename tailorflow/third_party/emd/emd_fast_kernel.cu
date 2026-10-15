// Multi-block approxmatch / matchcost (derived from PyTorchEMD, see NOTICE): same arithmetic and summation order per point as
// the original one-block-per-item kernel, but one thread per point over a (tiles, batch) grid.

#include <cmath>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAContext.h>

#define TPB 256
#define TILE 1024

template<typename scalar_t>
__global__ void am_init(int n, int m, scalar_t multiL, scalar_t multiR, scalar_t * __restrict__ remainL, scalar_t * __restrict__ remainR){
	int i = blockIdx.y, t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t < n) remainL[i * n + t] = multiL;
	if (t < m) remainR[i * m + t] = multiR;
}

template<typename scalar_t>
__global__ void am_rows(int n, int m, scalar_t level, const scalar_t * __restrict__ xyz1, const scalar_t * __restrict__ xyz2,
                        const scalar_t * __restrict__ remainL, const scalar_t * __restrict__ remainR, scalar_t * __restrict__ ratioL){
	__shared__ scalar_t buf[TILE * 4];
	int i = blockIdx.y, k = blockIdx.x * blockDim.x + threadIdx.x;
	scalar_t x1 = 0, y1 = 0, z1 = 0;
	if (k < n){ x1 = xyz1[(i * n + k) * 3 + 0]; y1 = xyz1[(i * n + k) * 3 + 1]; z1 = xyz1[(i * n + k) * 3 + 2]; }
	scalar_t suml = 1e-9f;
	for (int l0 = 0; l0 < m; l0 += TILE){
		int lend = min(m, l0 + TILE) - l0;
		for (int l = threadIdx.x; l < lend; l += blockDim.x){
			buf[l * 4 + 0] = xyz2[(i * m + l0 + l) * 3 + 0];
			buf[l * 4 + 1] = xyz2[(i * m + l0 + l) * 3 + 1];
			buf[l * 4 + 2] = xyz2[(i * m + l0 + l) * 3 + 2];
			buf[l * 4 + 3] = remainR[i * m + l0 + l];
		}
		__syncthreads();
		for (int l = 0; l < lend; l++){
			scalar_t x2 = buf[l * 4 + 0], y2 = buf[l * 4 + 1], z2 = buf[l * 4 + 2];
			scalar_t d = level * ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1));
			scalar_t w = __expf(d) * buf[l * 4 + 3];
			suml += w;
		}
		__syncthreads();
	}
	if (k < n) ratioL[i * n + k] = remainL[i * n + k] / suml;
}

template<typename scalar_t>
__global__ void am_cols(int n, int m, scalar_t level, const scalar_t * __restrict__ xyz1, const scalar_t * __restrict__ xyz2,
                        const scalar_t * __restrict__ ratioL, scalar_t * __restrict__ remainR, scalar_t * __restrict__ ratioR){
	__shared__ scalar_t buf[TILE * 4];
	int i = blockIdx.y, l = blockIdx.x * blockDim.x + threadIdx.x;
	scalar_t x2 = 0, y2 = 0, z2 = 0;
	if (l < m){ x2 = xyz2[(i * m + l) * 3 + 0]; y2 = xyz2[(i * m + l) * 3 + 1]; z2 = xyz2[(i * m + l) * 3 + 2]; }
	scalar_t sumr = 0;
	for (int k0 = 0; k0 < n; k0 += TILE){
		int kend = min(n, k0 + TILE) - k0;
		for (int k = threadIdx.x; k < kend; k += blockDim.x){
			buf[k * 4 + 0] = xyz1[(i * n + k0 + k) * 3 + 0];
			buf[k * 4 + 1] = xyz1[(i * n + k0 + k) * 3 + 1];
			buf[k * 4 + 2] = xyz1[(i * n + k0 + k) * 3 + 2];
			buf[k * 4 + 3] = ratioL[i * n + k0 + k];
		}
		__syncthreads();
		for (int k = 0; k < kend; k++){
			scalar_t x1 = buf[k * 4 + 0], y1 = buf[k * 4 + 1], z1 = buf[k * 4 + 2];
			scalar_t w = __expf(level * ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1))) * buf[k * 4 + 3];
			sumr += w;
		}
		__syncthreads();
	}
	if (l < m){
		scalar_t r = remainR[i * m + l];
		sumr *= r;
		scalar_t consumption = fminf(r / (sumr + 1e-9f), 1.0f);
		ratioR[i * m + l] = consumption * r;
		remainR[i * m + l] = fmaxf(0.0f, r - sumr);
	}
}

template<typename scalar_t>
__global__ void am_suml(int n, int m, scalar_t level, const scalar_t * __restrict__ xyz1, const scalar_t * __restrict__ xyz2,
                        const scalar_t * __restrict__ ratioL, const scalar_t * __restrict__ ratioR, scalar_t * __restrict__ remainL){
	__shared__ scalar_t buf[TILE * 4];
	int i = blockIdx.y, k = blockIdx.x * blockDim.x + threadIdx.x;
	scalar_t x1 = 0, y1 = 0, z1 = 0;
	if (k < n){ x1 = xyz1[(i * n + k) * 3 + 0]; y1 = xyz1[(i * n + k) * 3 + 1]; z1 = xyz1[(i * n + k) * 3 + 2]; }
	scalar_t rl = k < n ? ratioL[i * n + k] : 0;
	scalar_t suml = 0;
	for (int l0 = 0; l0 < m; l0 += TILE){
		int lend = min(m, l0 + TILE) - l0;
		for (int l = threadIdx.x; l < lend; l += blockDim.x){
			buf[l * 4 + 0] = xyz2[(i * m + l0 + l) * 3 + 0];
			buf[l * 4 + 1] = xyz2[(i * m + l0 + l) * 3 + 1];
			buf[l * 4 + 2] = xyz2[(i * m + l0 + l) * 3 + 2];
			buf[l * 4 + 3] = ratioR[i * m + l0 + l];
		}
		__syncthreads();
		if (k < n){
			for (int l = 0; l < lend; l++){
				scalar_t x2 = buf[l * 4 + 0], y2 = buf[l * 4 + 1], z2 = buf[l * 4 + 2];
				scalar_t w = __expf(level * ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1))) * rl * buf[l * 4 + 3];
				suml += w;
			}
		}
		__syncthreads();
	}
	if (k < n) remainL[i * n + k] = fmaxf(0.0f, remainL[i * n + k] - suml);
}

#define NLEV 10
#define FTILE 256
struct Levels { float v[NLEV]; };

template<typename scalar_t>
__global__ void am_final(int b, int n, int m, Levels lev, const scalar_t * __restrict__ xyz1, const scalar_t * __restrict__ xyz2,
                         const scalar_t * __restrict__ ratioL, const scalar_t * __restrict__ ratioR, scalar_t * __restrict__ match){
	__shared__ scalar_t buf[FTILE * 3];
	__shared__ scalar_t rr[NLEV * FTILE];
	int i = blockIdx.y, k = blockIdx.x * blockDim.x + threadIdx.x;
	scalar_t x1 = 0, y1 = 0, z1 = 0, rl[NLEV];
	if (k < n){ x1 = xyz1[(i * n + k) * 3 + 0]; y1 = xyz1[(i * n + k) * 3 + 1]; z1 = xyz1[(i * n + k) * 3 + 2]; }
#pragma unroll
	for (int j = 0; j < NLEV; j++) rl[j] = k < n ? ratioL[((size_t)j * b + i) * n + k] : 0;
	scalar_t * mi = match + (size_t)i * n * m;
	for (int l0 = 0; l0 < m; l0 += FTILE){
		int lend = min(m, l0 + FTILE) - l0;
		for (int l = threadIdx.x; l < lend; l += blockDim.x){
			buf[l * 3 + 0] = xyz2[(i * m + l0 + l) * 3 + 0];
			buf[l * 3 + 1] = xyz2[(i * m + l0 + l) * 3 + 1];
			buf[l * 3 + 2] = xyz2[(i * m + l0 + l) * 3 + 2];
#pragma unroll
			for (int j = 0; j < NLEV; j++) rr[j * FTILE + l] = ratioR[((size_t)j * b + i) * m + l0 + l];
		}
		__syncthreads();
		if (k < n){
			for (int l = 0; l < lend; l++){
				scalar_t x2 = buf[l * 3 + 0], y2 = buf[l * 3 + 1], z2 = buf[l * 3 + 2];
				scalar_t acc = 0;
#pragma unroll
				for (int j = 0; j < NLEV; j++){
					scalar_t level = lev.v[j];
					scalar_t w = __expf(level * ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1))) * rl[j] * rr[j * FTILE + l];
					acc += w;
				}
				mi[(size_t)(l0 + l) * n + k] = acc;
			}
		}
		__syncthreads();
	}
}

at::Tensor ApproxMatchForwardFast(const at::Tensor xyz1, const at::Tensor xyz2){
	const int b = xyz1.size(0), n = xyz1.size(1), m = xyz2.size(1);
	TORCH_CHECK(xyz2.size(0) == b && xyz1.size(2) == 3 && xyz2.size(2) == 3, "approxmatch: bad shapes");
	auto match = at::empty({b, m, n}, xyz1.options());
	auto remainL = at::empty({b, n}, xyz1.options()), remainR = at::empty({b, m}, xyz1.options());
	auto ratioL = at::empty({NLEV, b, n}, xyz1.options()), ratioR = at::empty({NLEV, b, m}, xyz1.options());
	auto stream = at::cuda::getCurrentCUDAStream();
	dim3 gn((n + TPB - 1) / TPB, b), gm((m + TPB - 1) / TPB, b), gi((std::max(n, m) + TPB - 1) / TPB, b);
	AT_DISPATCH_FLOATING_TYPES(xyz1.scalar_type(), "ApproxMatchForwardFast", ([&] {
		scalar_t multiL, multiR;
		if (n >= m){ multiL = 1; multiR = n / m; } else { multiL = m / n; multiR = 1; }
		const scalar_t *p1 = xyz1.data_ptr<scalar_t>(), *p2 = xyz2.data_ptr<scalar_t>();
		scalar_t *rL = remainL.data_ptr<scalar_t>(), *rR = remainR.data_ptr<scalar_t>();
		Levels lev;
		am_init<scalar_t><<<gi, TPB, 0, stream>>>(n, m, multiL, multiR, rL, rR);
		for (int j = 7, s = 0; j >= -2; j--, s++){
			scalar_t level = j == -2 ? 0 : -powf(4.0f, j);
			lev.v[s] = level;
			scalar_t *qL = ratioL.data_ptr<scalar_t>() + (size_t)s * b * n, *qR = ratioR.data_ptr<scalar_t>() + (size_t)s * b * m;
			am_rows<scalar_t><<<gn, TPB, 0, stream>>>(n, m, level, p1, p2, rL, rR, qL);
			am_cols<scalar_t><<<gm, TPB, 0, stream>>>(n, m, level, p1, p2, qL, rR, qR);
			am_suml<scalar_t><<<gn, TPB, 0, stream>>>(n, m, level, p1, p2, qL, qR, rL);
		}
		am_final<scalar_t><<<dim3((n + FTILE - 1) / FTILE, b), FTILE, 0, stream>>>(b, n, m, lev, p1, p2, ratioL.data_ptr<scalar_t>(),
			ratioR.data_ptr<scalar_t>(), match.data_ptr<scalar_t>());
	}));
	AT_CUDA_CHECK(cudaGetLastError());
	return match;
}

template<typename scalar_t>
__global__ void mc_partial(int n, int m, const scalar_t * __restrict__ xyz1, const scalar_t * __restrict__ xyz2,
                           const scalar_t * __restrict__ match, scalar_t * __restrict__ partial){
	__shared__ scalar_t buf[TILE * 3];
	__shared__ scalar_t red[TPB];
	int i = blockIdx.y, k = blockIdx.x * blockDim.x + threadIdx.x;
	scalar_t x1 = 0, y1 = 0, z1 = 0;
	if (k < n){ x1 = xyz1[(i * n + k) * 3 + 0]; y1 = xyz1[(i * n + k) * 3 + 1]; z1 = xyz1[(i * n + k) * 3 + 2]; }
	const scalar_t * mi = match + (size_t)i * n * m;
	scalar_t s = 0;
	for (int l0 = 0; l0 < m; l0 += TILE){
		int lend = min(m, l0 + TILE) - l0;
		for (int l = threadIdx.x; l < lend * 3; l += blockDim.x) buf[l] = xyz2[(i * m + l0) * 3 + l];
		__syncthreads();
		if (k < n){
			for (int l = 0; l < lend; l++){
				scalar_t x2 = buf[l * 3 + 0], y2 = buf[l * 3 + 1], z2 = buf[l * 3 + 2];
				scalar_t d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
				s += d * mi[(size_t)(l0 + l) * n + k];
			}
		}
		__syncthreads();
	}
	red[threadIdx.x] = s;
	for (int j = blockDim.x / 2; j > 0; j >>= 1){
		__syncthreads();
		if (threadIdx.x < j) red[threadIdx.x] += red[threadIdx.x + j];
	}
	if (threadIdx.x == 0) partial[i * gridDim.x + blockIdx.x] = red[0];
}

at::Tensor MatchCostForwardFast(const at::Tensor xyz1, const at::Tensor xyz2, const at::Tensor match){
	const int b = xyz1.size(0), n = xyz1.size(1), m = xyz2.size(1);
	dim3 g((n + TPB - 1) / TPB, b);
	auto partial = at::zeros({b, (int)g.x}, xyz1.options());
	AT_DISPATCH_FLOATING_TYPES(xyz1.scalar_type(), "MatchCostForwardFast", ([&] {
		mc_partial<scalar_t><<<g, TPB, 0, at::cuda::getCurrentCUDAStream()>>>(n, m, xyz1.data_ptr<scalar_t>(), xyz2.data_ptr<scalar_t>(),
			match.data_ptr<scalar_t>(), partial.data_ptr<scalar_t>());
	}));
	AT_CUDA_CHECK(cudaGetLastError());
	return partial.sum(1);
}

template<typename scalar_t>
__global__ void mc_grad1(int n, int m, const scalar_t * __restrict__ grad_cost, const scalar_t * __restrict__ xyz1,
                         const scalar_t * __restrict__ xyz2, const scalar_t * __restrict__ match, scalar_t * __restrict__ grad1){
	int i = blockIdx.y, l = blockIdx.x * blockDim.x + threadIdx.x;
	if (l >= n) return;
	scalar_t x1 = xyz1[(i * n + l) * 3 + 0], y1 = xyz1[(i * n + l) * 3 + 1], z1 = xyz1[(i * n + l) * 3 + 2];
	const scalar_t * mi = match + (size_t)i * n * m;
	scalar_t dx = 0, dy = 0, dz = 0;
	for (int k = 0; k < m; k++){
		scalar_t x2 = xyz2[(i * m + k) * 3 + 0], y2 = xyz2[(i * m + k) * 3 + 1], z2 = xyz2[(i * m + k) * 3 + 2];
		scalar_t d = mi[(size_t)k * n + l] * 2;
		dx += (x1 - x2) * d;
		dy += (y1 - y2) * d;
		dz += (z1 - z2) * d;
	}
	grad1[(i * n + l) * 3 + 0] = dx * grad_cost[i];
	grad1[(i * n + l) * 3 + 1] = dy * grad_cost[i];
	grad1[(i * n + l) * 3 + 2] = dz * grad_cost[i];
}

template<typename scalar_t>
__global__ void mc_grad2(int n, int m, const scalar_t * __restrict__ grad_cost, const scalar_t * __restrict__ xyz1,
                         const scalar_t * __restrict__ xyz2, const scalar_t * __restrict__ match, scalar_t * __restrict__ grad2){
	__shared__ scalar_t sum_grad[256 * 3];
	int i = blockIdx.x;
	int kbeg = m * blockIdx.y / gridDim.y, kend = m * (blockIdx.y + 1) / gridDim.y;
	const scalar_t * mi = match + (size_t)i * n * m;
	for (int k = kbeg; k < kend; k++){
		scalar_t x2 = xyz2[(i * m + k) * 3 + 0], y2 = xyz2[(i * m + k) * 3 + 1], z2 = xyz2[(i * m + k) * 3 + 2];
		scalar_t sx = 0, sy = 0, sz = 0;
		for (int j = threadIdx.x; j < n; j += blockDim.x){
			scalar_t x1 = x2 - xyz1[(i * n + j) * 3 + 0];
			scalar_t y1 = y2 - xyz1[(i * n + j) * 3 + 1];
			scalar_t z1 = z2 - xyz1[(i * n + j) * 3 + 2];
			scalar_t d = mi[(size_t)k * n + j] * 2;
			sx += x1 * d;
			sy += y1 * d;
			sz += z1 * d;
		}
		sum_grad[threadIdx.x * 3 + 0] = sx;
		sum_grad[threadIdx.x * 3 + 1] = sy;
		sum_grad[threadIdx.x * 3 + 2] = sz;
		for (int j = 1; j < blockDim.x; j <<= 1){
			__syncthreads();
			int j1 = threadIdx.x, j2 = threadIdx.x + j;
			if ((j1 & j) == 0 && j2 < blockDim.x){
				sum_grad[j1 * 3 + 0] += sum_grad[j2 * 3 + 0];
				sum_grad[j1 * 3 + 1] += sum_grad[j2 * 3 + 1];
				sum_grad[j1 * 3 + 2] += sum_grad[j2 * 3 + 2];
			}
		}
		if (threadIdx.x == 0){
			grad2[(i * m + k) * 3 + 0] = sum_grad[0] * grad_cost[i];
			grad2[(i * m + k) * 3 + 1] = sum_grad[1] * grad_cost[i];
			grad2[(i * m + k) * 3 + 2] = sum_grad[2] * grad_cost[i];
		}
		__syncthreads();
	}
}

std::vector<at::Tensor> MatchCostBackwardFast(const at::Tensor grad_cost, const at::Tensor xyz1, const at::Tensor xyz2, const at::Tensor match){
	const int b = xyz1.size(0), n = xyz1.size(1), m = xyz2.size(1);
	auto grad1 = at::zeros({b, n, 3}, xyz1.options());
	auto grad2 = at::zeros({b, m, 3}, xyz1.options());
	auto stream = at::cuda::getCurrentCUDAStream();
	int ys = std::max(1, std::min(m, 2048 / std::max(b, 1)));
	AT_DISPATCH_FLOATING_TYPES(xyz1.scalar_type(), "MatchCostBackwardFast", ([&] {
		mc_grad1<scalar_t><<<dim3((n + TPB - 1) / TPB, b), TPB, 0, stream>>>(n, m, grad_cost.data_ptr<scalar_t>(), xyz1.data_ptr<scalar_t>(),
			xyz2.data_ptr<scalar_t>(), match.data_ptr<scalar_t>(), grad1.data_ptr<scalar_t>());
		mc_grad2<scalar_t><<<dim3(b, ys), 256, 0, stream>>>(n, m, grad_cost.data_ptr<scalar_t>(), xyz1.data_ptr<scalar_t>(),
			xyz2.data_ptr<scalar_t>(), match.data_ptr<scalar_t>(), grad2.data_ptr<scalar_t>());
	}));
	AT_CUDA_CHECK(cudaGetLastError());
	return {grad1, grad2};
}

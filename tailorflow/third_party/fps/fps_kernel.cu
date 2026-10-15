// Farthest point sampling with the selection rule of pytorch3d's FarthestPointSamplingKernel (threads = min(2^floor(log2 P),
// 1024), point p owned by thread p % threads, ties -> lowest thread, then lowest p), with each thread's points and running
// min distances kept in registers and warp-shuffle reductions.

#include <vector>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAContext.h>

#define MAXPPT 16

template<int T, int M>
__global__ void __launch_bounds__(T) fps_kernel(int P, int K, const float * __restrict__ pts, const int64_t * __restrict__ start, int64_t * __restrict__ out){
	const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
	const float * pb = pts + (size_t)b * P * 3;
	float px[M], py[M], pz[M], md[M];
#pragma unroll
	for (int j = 0; j < M; j++){
		int p = tid + j * T;
		if (p < P){ px[j] = pb[p * 3 + 0]; py[j] = pb[p * 3 + 1]; pz[j] = pb[p * 3 + 2]; }
		md[j] = 1e10f;
	}
	__shared__ float wv[32];
	__shared__ int wi[32];
	__shared__ int sel_s;
	int sel = (int)start[b];
	if (tid == 0) out[(size_t)b * K] = sel;
	for (int k = 1; k < K; k++){
		const float sx = pb[sel * 3 + 0], sy = pb[sel * 3 + 1], sz = pb[sel * 3 + 2];
		float best = -1.0f;
		int bi = 0;
#pragma unroll
		for (int j = 0; j < M; j++){
			int p = tid + j * T;
			if (p < P){
				float dist2 = 0.0f, diff;
				diff = sx - px[j]; dist2 += diff * diff;
				diff = sy - py[j]; dist2 += diff * diff;
				diff = sz - pz[j]; dist2 += diff * diff;
				float m = fminf(dist2, md[j]);
				md[j] = m;
				if (m > best){ best = m; bi = p; }
			}
		}
		int bt = tid;
#pragma unroll
		for (int off = 16; off > 0; off >>= 1){
			float ov = __shfl_down_sync(0xffffffff, best, off);
			int oi = __shfl_down_sync(0xffffffff, bi, off);
			int ot = __shfl_down_sync(0xffffffff, bt, off);
			if (ov > best || (ov == best && ot < bt)){ best = ov; bi = oi; bt = ot; }
		}
		if (T > 32){
			if (lane == 0){ wv[warp] = best; wi[warp] = bi; }
			__syncthreads();
			if (warp == 0){
				const int nw = T / 32;
				best = lane < nw ? wv[lane] : -2.0f;
				bi = lane < nw ? wi[lane] : 0;
				bt = lane;
#pragma unroll
				for (int off = 16; off > 0; off >>= 1){
					float ov = __shfl_down_sync(0xffffffff, best, off);
					int oi = __shfl_down_sync(0xffffffff, bi, off);
					int ot = __shfl_down_sync(0xffffffff, bt, off);
					if (ov > best || (ov == best && ot < bt)){ best = ov; bi = oi; bt = ot; }
				}
				if (lane == 0){ sel_s = bi; out[(size_t)b * K + k] = bi; }
			}
			__syncthreads();
			sel = sel_s;
		} else {
			sel = __shfl_sync(0xffffffff, bi, 0);
			if (tid == 0) out[(size_t)b * K + k] = sel;
		}
	}
}

at::Tensor FPSForward(const at::Tensor points, int64_t K, const at::Tensor start){
	TORCH_CHECK(points.is_cuda() && points.scalar_type() == at::kFloat && points.dim() == 3 && points.size(2) == 3, "fps: points [B, P, 3] float cuda");
	const int B = points.size(0), P = points.size(1);
	TORCH_CHECK(P <= 1024 * MAXPPT, "fps: P too large");
	auto pts = points.contiguous();
	auto out = at::empty({B, K}, points.options().dtype(at::kLong));
	auto st = start.to(at::kLong).contiguous();
	int lg = (int)(std::log((double)P) / std::log(2.0));
	int T = std::max(std::min(1 << lg, 1024), 2);
	auto stream = at::cuda::getCurrentCUDAStream();
	const float * pp = pts.data_ptr<float>();
	const int64_t * sp = st.data_ptr<int64_t>();
	int64_t * op = out.data_ptr<int64_t>();
	switch (T){
		case 1024:
			if (P > 1024 * 8) fps_kernel<1024, 16><<<B, 1024, 0, stream>>>(P, K, pp, sp, op);
			else fps_kernel<1024, 8><<<B, 1024, 0, stream>>>(P, K, pp, sp, op);
			break;
		case 512: fps_kernel<512, 8><<<B, 512, 0, stream>>>(P, K, pp, sp, op); break;
		case 256: fps_kernel<256, 8><<<B, 256, 0, stream>>>(P, K, pp, sp, op); break;
		case 128: fps_kernel<128, 8><<<B, 128, 0, stream>>>(P, K, pp, sp, op); break;
		case 64: fps_kernel<64, 8><<<B, 64, 0, stream>>>(P, K, pp, sp, op); break;
		default: TORCH_CHECK(false, "fps: unsupported P");
	}
	AT_CUDA_CHECK(cudaGetLastError());
	return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("fps", &FPSForward, "farthest point sampling (CUDA)");
}

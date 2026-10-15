#include <vector>
#include <torch/extension.h>

at::Tensor ApproxMatchForwardFast(const at::Tensor xyz1, const at::Tensor xyz2);
at::Tensor MatchCostForwardFast(const at::Tensor xyz1, const at::Tensor xyz2, const at::Tensor match);
std::vector<at::Tensor> MatchCostBackwardFast(const at::Tensor grad_cost, const at::Tensor xyz1, const at::Tensor xyz2, const at::Tensor match);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("approxmatch_forward", &ApproxMatchForwardFast, "ApproxMatch forward, multi-block (CUDA)");
  m.def("matchcost_forward", &MatchCostForwardFast, "MatchCost forward, multi-block (CUDA)");
  m.def("matchcost_backward", &MatchCostBackwardFast, "MatchCost backward, multi-block (CUDA)");
}

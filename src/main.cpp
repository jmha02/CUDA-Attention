#include <torch/extension.h>

torch::Tensor flash_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor naive_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor flash_attention_unified(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor naive_attention_unified(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention", torch::wrap_pybind_function(flash_attention), "flash_attention");
    m.def("naive_attention", torch::wrap_pybind_function(naive_attention), "naive_attention");
    m.def("flash_attention_unified", torch::wrap_pybind_function(flash_attention_unified), "flash_attention_unified");
    m.def("naive_attention_unified", torch::wrap_pybind_function(naive_attention_unified), "naive_attention_unified");
}

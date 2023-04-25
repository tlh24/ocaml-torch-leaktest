#include <torch/torch.h>
#include <iostream>
#include <stdio.h>
#include <nvml.h>

void check_gpu_memory_usage() {
    nvmlReturn_t result;
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        throw std::runtime_error("Failed to initialize NVML");
    }

    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        throw std::runtime_error("Failed to get GPU device handle");
    }

    nvmlMemory_t memory_info;
    result = nvmlDeviceGetMemoryInfo(device, &memory_info);
    if (result != NVML_SUCCESS) {
        throw std::runtime_error("Failed to get GPU memory info");
    }

    std::cout << "Total GPU memory: " << memory_info.total / (1024 * 1024) << " MiB" << std::endl;
    std::cout << "Free GPU memory: " << memory_info.free / (1024 * 1024) << " MiB" << std::endl;
    std::cout << "Used GPU memory: " << memory_info.used / (1024 * 1024) << " MiB" << std::endl;

    result = nvmlShutdown();
    if (result != NVML_SUCCESS) {
        throw std::runtime_error("Failed to shut down NVML");
    }
}


std::pair<int64_t, float> find_min_mse(const torch::Tensor& A, const torch::Tensor& B) {
    int64_t n = A.size(0);

    // Reshape A and B to add a singleton dimension for broadcasting.
    auto A_reshaped = A.view({n, 1, -1}); // Shape: [n, 1, x * x]
    auto B_reshaped = B.view({1, -1}); // Shape: [1, x * x]

    // Compute (A-B)^2.  I think this requires a temp tensor, alas.
    auto squared_diff = (A_reshaped - B_reshaped).pow(2); // Shape: [n, x * x]

    // Sum over the last dimension [x * x].
    auto mse = squared_diff.sum({-1}); // Shape: [n]

    // Find the index of the smallest mean squared error value and its corresponding distance.
    auto min_mse = mse.min(0);
	 int64_t index = std::get<1>(min_mse).item<int64_t>();
    float distance = std::get<0>(min_mse).item<float>();

    return {index, distance};
}

int main() {
	torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	std::cout << "Running on " << (device.type() == torch::kCUDA ? "GPU" : "CPU") << std::endl;

	// Create sample tensors A and B
	torch::Tensor A = torch::randn({4*2048, 30, 30}).to(device);

	// Find the minimum mean squared error and its index
	for(int i=0; i<30000; i++){
		torch::Tensor B = torch::randn({30, 30}).to(device);
		auto result = find_min_mse(A, B);
		int64_t index = result.first;
		float distance = result.second;

		// Print the results
		std::cout << "Minimum mean squared error index: " << index << std::endl;
		std::cout << "Minimum mean squared error distance: " << distance << std::endl;
      if(i == 15){
          check_gpu_memory_usage();
      }
	 }
	 check_gpu_memory_usage();
    return 0;
}


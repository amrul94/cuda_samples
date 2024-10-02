#include <cstdio>

namespace utils {

void handleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    std::exit(EXIT_FAILURE);
  }
}

void handleError(void *pointer, const char *file, int line) {
  if (pointer == nullptr) {
    std::printf("Host memory failed in %s at line %d\n", file, line);
    std::exit(EXIT_FAILURE);
  }
}

} // namespace utils

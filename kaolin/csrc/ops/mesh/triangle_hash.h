#include <torch/custom_class.h>

namespace kaolin {

struct TriangleHash : torch::CustomClassHolder {
  std::vector<std::vector<int>> _spatial_hash { };
  int _resolution;

public:
  TriangleHash(at::Tensor triangles, int resolution);
  std::vector<at::Tensor> query(at::Tensor points);

private:
  void _build_hash(at::Tensor triangles);
};

} // namespace

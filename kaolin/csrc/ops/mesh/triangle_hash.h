#include <torch/custom_class.h>

namespace kaolin {

struct TriangleHash : torch::CustomClassHolder {
  // This class is a 2D grid listing the triangle IDs
  // where each pixel is overlapping with the triangle AABB
  // A bit like tile-rendering
  std::vector<std::vector<int>> _spatial_hash { };
  int _resolution;

public:
  TriangleHash(at::Tensor triangles, int resolution);

  // check on the 2d grid the matching triangle ID for each 2d points
  std::vector<at::Tensor> query(at::Tensor points);

private:
  void _build_hash(at::Tensor triangles);
};

} // namespace

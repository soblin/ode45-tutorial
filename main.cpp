#include <matplotlibcpp17/pyplot.h>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>

#include <vector>

using namespace std;

int main() {
  pybind11::scoped_interpreter guard{};
  auto plt = matplotlibcpp17::pyplot::import();

  auto ts_ = xt::arange(0.0, 0.1, 0.001);
  auto analytical_ = 10.0 * xt::exp(-150 * ts_);
  vector<double> ts(ts_.begin(), ts_.end());
  vector<double> analytical(analytical_.begin(), analytical_.end());
  
  /// user code
  plt.plot(Args(ts, analytical),
           Kwargs("color"_a = "blue", "linewidth"_a = 1.0));
  plt.show();
}

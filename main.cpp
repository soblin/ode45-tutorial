#include <matplotlibcpp17/pyplot.h>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>

#include <vector>
#include <functional>

using namespace std;

double euler(const double t, const double y, const double h,
             std::function<double(double, double)> dy) {
  return y + dy(t, y) * h;
}

double rk4(const double t, const double y, const double h,
           std::function<double(double, double)> dy) {
  double k1 = dy(t, y);
  double k2 = dy(t + h / 2, y + h / 2 * k1);
  double k3 = dy(t + h / 2, y + h / 2 * k2);
  double k4 = dy(t + h / 2, y + h * k3);
  return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

int main() {
  pybind11::scoped_interpreter guard{};
  auto plt = matplotlibcpp17::pyplot::import();

  // data
  static constexpr double y0 = 10.0;
  static constexpr double a = 150;
  static constexpr double dt = 0.001;
  static constexpr double h = 0.01;
  auto dy = [&](const double t, const double y) { return -a * y; };

  // analytical
  auto dts_ = xt::arange(0.0, 0.1, dt);
  auto analytical_ = y0 * xt::exp(-a * dts_);
  vector<double> dts(dts_.begin(), dts_.end());
  // numerical
  auto ts_ = xt::arange(0.0, 0.1, h);
  vector<double> ts(ts_.begin(), ts_.end());
  vector<double> analytical(analytical_.begin(), analytical_.end());
  vector<double> euler_solution({y0});
  vector<double> rk4_solution({y0});

  for (unsigned i = 1; i < ts.size(); ++i) {
    euler_solution.push_back(euler(ts[i - 1], euler_solution[i - 1], h, dy));
    rk4_solution.push_back(rk4(ts[i - 1], rk4_solution[i - 1], h, dy));
  }

  // plot
  auto [fig, ax] = plt.subplots();
  ax.plot(Args(dts, analytical), Kwargs("color"_a = "blue", "linewidth"_a = 1.5,
                                        "label"_a = "Analytical solution"));
  ax.plot(Args(ts, euler_solution),
          Kwargs("color"_a = "orange", "linewidth"_a = 1.0,
                 "label"_a = "1st Euler", "marker"_a = "*"));
  ax.plot(Args(ts, rk4_solution),
          Kwargs("color"_a = "purple", "linewidth"_a = 1.0,
                 "label"_a = "4th Runge-Kutta", "marker"_a = "."));
  ax.legend();
  ax.grid(Args(true));
  plt.show();
}

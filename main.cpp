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
  double k4 = dy(t + h, y + h * k3);
  return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

// this butcher table is from Runge-Kutta-Fehlberg
double rk5(const double t, const double y, const double h,
           std::function<double(double, double)> dy) {
  double k1 = dy(t, y);
  double k2 = dy(t + h / 4, y + h / 4 * k1);
  double k3 = dy(t + 3 * h / 8, y + 3 * h / 32 * k1 + 9 * h / 32 * k2);
  double k4 =
      dy(t + 12 * h / 13, y + 1932 * h / 2197 * k1 - 7200 * h / 2197 * k2 +
                              7296 * h / 2197 * k3);
  double k5 = dy(t + h, y + 439 * h / 216 * k1 - 8 * h * k2 +
                            3680 * h / 513 * k3 - 845 * h / 4104 * k4);
  double k6 =
      dy(t + h / 2, y - 8 * h / 27 * k1 + 2 * h * k2 - 3544 * h / 2565 * k3 +
                        1859 * h / 4104 * k4 - 11 * h / 40 * k5);
  return y + h * (16.0 / 135 * k1 + 6656.0 / 12825 * k3 + 28561.0 / 56430 * k4 -
                  9.0 / 50 * k5 + 2.0 / 55 * k6);
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
  vector<double> rk5_solution({y0});

  for (unsigned i = 1; i < ts.size(); ++i) {
    euler_solution.push_back(euler(ts[i - 1], euler_solution[i - 1], h, dy));
    rk4_solution.push_back(rk4(ts[i - 1], rk4_solution[i - 1], h, dy));
    rk5_solution.push_back(rk5(ts[i - 1], rk5_solution[i - 1], h, dy));
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
                 "label"_a = "4th Runge-Kutta", "marker"_a = "o"));
  ax.plot(Args(ts, rk5_solution),
          Kwargs("color"_a = "green", "linewidth"_a = 1.0,
                 "label"_a = "5th Runge-Kutta", "marker"_a = "x"));
  ax.legend();
  ax.grid(Args(true));
  plt.show();
}

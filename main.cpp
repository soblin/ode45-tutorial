#include <matplotlibcpp17/pyplot.h>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>

#include <vector>
#include <functional>

using namespace std;

double euler1(const double t, const double y, const double h,
              std::function<double(double, double)> dy) {
  return y + dy(t, y) * h;
}

double rk4(const double t, const double y, const double h,
           std::function<double(double, double)> dy) {
  static constexpr double a21 = 1.0 / 2;
  static constexpr double a32 = 1.0 / 2;
  static constexpr double b1 = 1.0 / 6;
  static constexpr double b2 = 1.0 / 3;
  static constexpr double b3 = 1.0 / 3;
  static constexpr double b4 = 1.0 / 6;
  static constexpr double c2 = 1.0 / 2;
  static constexpr double c3 = 1.0 / 2;
  // static constexpr double c4 = 1;
  double k1 = dy(t, y);
  double k2 = dy(t + c2 * h, y + h * a21 * k1);
  double k3 = dy(t + c3 * h, y + h * a32 * k2);
  double k4 = dy(t + h, y + h * k3);
  return y + h * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4);
}

double rk5(const double t, const double y, const double h,
           std::function<double(double, double)> dy) {
  // butcher table from Runge-Kutta-Fehlberg
  static constexpr double a21 = 1.0 / 4;
  static constexpr double a31 = 3.0 / 32;
  static constexpr double a32 = 9.0 / 32;
  static constexpr double a41 = 1932.0 / 2197;
  static constexpr double a42 = -7200.0 / 2197;
  static constexpr double a43 = 7296.0 / 2197;
  static constexpr double a51 = 439.0 / 216;
  static constexpr double a52 = -8.0;
  static constexpr double a53 = 3680.0 / 513;
  static constexpr double a54 = -845 / 4104;
  static constexpr double a61 = -8.0 / 27;
  static constexpr double a62 = 2.0;
  static constexpr double a63 = -3544.0 / 2565;
  static constexpr double a64 = 1859.0 / 4104;
  static constexpr double a65 = -11.0 / 40;
  static constexpr double b1 = 16.0 / 135;
  // static constexpr double b2 = 0;
  static constexpr double b3 = 6656.0 / 12825;
  static constexpr double b4 = 28561.0 / 56430;
  static constexpr double b5 = -9.0 / 50;
  static constexpr double b6 = 2.0 / 55;
  static constexpr double c2 = 1.0 / 4;
  static constexpr double c3 = 3.0 / 8;
  static constexpr double c4 = 12.0 / 13;
  // static constexor double c5 = 1;
  static constexpr double c6 = 1.0 / 2;
  double k1 = dy(t, y);
  double k2 = dy(t + c2 * h, y + h * a21 * k1);
  double k3 = dy(t + c3 * h, y + h * (a31 * k1 + a32 * k2));
  double k4 = dy(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3));
  double k5 = dy(t + h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
  double k6 = dy(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 +
                                      a64 * k4 + a65 * k5));
  return y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6);
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
    euler_solution.push_back(euler1(ts[i - 1], euler_solution[i - 1], h, dy));
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

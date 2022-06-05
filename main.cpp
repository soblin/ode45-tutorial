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

std::pair<double, double> ode45(const double t, const double y, const double h,
                                const double tol,
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
  static constexpr double b1_ = 25.0 / 216;
  // static constexpr double b2_ = 0;
  static constexpr double b3_ = 1408.0 / 2565;
  static constexpr double b4_ = 2197.0 / 4104;
  static constexpr double b5_ = -1.0 / 5.0;
  static constexpr double b6_ = 0;
  static constexpr double c2 = 1.0 / 4;
  static constexpr double c3 = 3.0 / 8;
  static constexpr double c4 = 12.0 / 13;
  // static constexor double c5 = 1;
  static constexpr double c6 = 1.0 / 2;
  const double k1 = dy(t, y);
  const double k2 = dy(t + c2 * h, y + h * a21 * k1);
  const double k3 = dy(t + c3 * h, y + h * (a31 * k1 + a32 * k2));
  const double k4 = dy(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3));
  const double k5 =
      dy(t + h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
  const double k6 = dy(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 +
                                            a64 * k4 + a65 * k5));
  // NOTE: update discretization
  const double e =
      fabs(h * ((b1 - b1_) * k1 + (b3 - b3_) * k3 + (b4 - b4_) * k4 +
                (b5 - b5_) * k5 + (b6 - b6_) * k6));
  double h_new = h;
  if (e > h * tol)
    h_new = h / 2;
  else if (e < h * tol / 10)
    h_new = 2 * h;

  // this implementation is not complete. If h_new is updated, do re-calculation
  // So all the above solver should return [y_next, t_next(, +h for ode45)]
  return {y + h_new * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6), h_new};
}

int main() {
  pybind11::scoped_interpreter guard{};
  auto plt = matplotlibcpp17::pyplot::import();

  // data
  static constexpr double y0 = 10.0;
  static constexpr double a = 150;
  static constexpr double dt = 0.001;
  static constexpr double t0 = 0.0;
  static constexpr double tf = 0.1;
  static constexpr double h = 0.01;
  auto dy = [&](const double t, const double y) { return -a * y; };

  // analytical
  auto dts_xt = xt::arange(t0, tf, dt);
  auto analytical_xt = y0 * xt::exp(-a * dts_xt);
  vector<double> dts(dts_xt.begin(), dts_xt.end());
  vector<double> analytical(analytical_xt.begin(), analytical_xt.end());

  // euler1, rk4, rk5
  vector<double> ts({t0});
  vector<double> euler1_solution({y0});
  vector<double> rk4_solution({y0});
  vector<double> rk5_solution({y0});
  {
    double t;
    int i;
    for (t = t0, i = 0; t < tf; t += h, i++) {
      euler1_solution.push_back(euler1(ts[i], euler1_solution[i], h, dy));
      rk4_solution.push_back(rk4(ts[i], rk4_solution[i], h, dy));
      rk5_solution.push_back(rk5(ts[i], rk5_solution[i], h, dy));
      ts.push_back(t + h);
    }
  }

  // ode45
  vector<double> ts_ode45({t0});
  vector<double> ode45_solution({y0});
  const double tol = 0.1;
  {
    double t;
    int i;
    double h_ode45 = h;
    for (t = t0, i = 0; t < tf; t += h_ode45, i++) {
      auto [y_next, h_ode45_next] =
          ode45(ts_ode45[i], ode45_solution[i], h_ode45, tol, dy);
      ode45_solution.push_back(y_next);
      h_ode45 = h_ode45_next;
      ts_ode45.push_back(t + h_ode45_next);
    }
  }
  // plot
  auto [fig, ax] = plt.subplots();
  ax.plot(Args(dts, analytical), Kwargs("color"_a = "blue", "linewidth"_a = 1.5,
                                        "label"_a = "Analytical solution"));
  ax.plot(Args(ts, euler1_solution),
          Kwargs("color"_a = "orange", "linewidth"_a = 1.0,
                 "label"_a = "1st Euler", "marker"_a = "*"));
  ax.plot(Args(ts, rk4_solution),
          Kwargs("color"_a = "purple", "linewidth"_a = 1.0,
                 "label"_a = "4th Runge-Kutta", "marker"_a = "o"));
  ax.plot(Args(ts, rk5_solution),
          Kwargs("color"_a = "green", "linewidth"_a = 1.0,
                 "label"_a = "5th Runge-Kutta", "marker"_a = "x"));
  ax.plot(Args(ts_ode45, ode45_solution),
          Kwargs("color"_a = "red", "linewidth"_a = 1.0, "label"_a = "ode45",
                 "marker"_a = "x"));
  ax.legend();
  ax.grid(Args(true));
  plt.show();
}

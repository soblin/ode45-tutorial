#include <matplotlibcpp17/pyplot.h>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>

#include <vector>
#include <functional>
#include <tuple>

using namespace std;

std::tuple<double, double> euler1(const double t, const double y,
                                  const double h,
                                  std::function<double(double, double)> dy) {
  return {y + dy(t, y) * h, t + h};
}

std::tuple<double, double> rk4(const double t, const double y, const double h,
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
  return {y + h * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4), t + h};
}

std::tuple<double, double> rk5(const double t, const double y, const double h,
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
  return {y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6), t + h};
}

std::tuple<double, double, double>
ode45(const double t, const double y, const double h, const double tol,
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
  double k1 = dy(t, y);
  double k2 = dy(t + c2 * h, y + h * a21 * k1);
  double k3 = dy(t + c3 * h, y + h * (a31 * k1 + a32 * k2));
  double k4 = dy(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3));
  double k5 = dy(t + h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
  double k6 = dy(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 +
                                      a64 * k4 + a65 * k5));
  // NOTE: update discretization
  const double e =
      fabs(h * ((b1 - b1_) * k1 + (b3 - b3_) * k3 + (b4 - b4_) * k4 +
                (b5 - b5_) * k5 + (b6 - b6_) * k6));
  double h_new = h;
  bool reeval = false;
  if (e > h * tol) {
    h_new = h / 2;
    reeval = true;
  } else if (e < h * tol / 10) {
    h_new = 2 * h;
    reeval = true;
  }
  if (reeval) {
    double k1 = dy(t, y);
    double k2 = dy(t + c2 * h_new, y + h_new * a21 * k1);
    double k3 = dy(t + c3 * h_new, y + h_new * (a31 * k1 + a32 * k2));
    double k4 =
        dy(t + c4 * h_new, y + h_new * (a41 * k1 + a42 * k2 + a43 * k3));
    double k5 =
        dy(t + h_new, y + h_new * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
    double k6 = dy(t + c6 * h_new, y + h_new * (a61 * k1 + a62 * k2 + a63 * k3 +
                                                a64 * k4 + a65 * k5));
    return {y + h_new * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6),
            t + h_new, h_new};
  } else
    return {y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6), t + h,
            h};
}

int main() {
  pybind11::scoped_interpreter guard{};
  auto plt = matplotlibcpp17::pyplot::import();
  auto [fig, ax] = plt.subplots();

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
  ax.plot(Args(dts, analytical), Kwargs("color"_a = "blue", "linewidth"_a = 1.5,
                                        "label"_a = "Analytical solution"));

  // euler1
  {
    vector<double> ts({t0});
    vector<double> ys({y0});
    for (double t = t0; t < tf;) {
      auto [y_next, t_next] = euler1(ts.back(), ys.back(), h, dy);
      ys.push_back(y_next);
      ts.push_back(t_next);
      t = t_next;
    }
    ax.plot(Args(ts, ys), Kwargs("color"_a = "orange", "linewidth"_a = 1.0,
                                 "label"_a = "1st Euler", "marker"_a = "*"));
  }

  // rk4
  {
    vector<double> ts({t0});
    vector<double> ys({y0});
    for (double t = t0; t < tf;) {
      auto [y_next, t_next] = rk5(ts.back(), ys.back(), h, dy);
      ys.push_back(y_next);
      ts.push_back(t_next);
      t = t_next;
    }
    ax.plot(Args(ts, ys),
            Kwargs("color"_a = "green", "linewidth"_a = 1.0,
                   "label"_a = "5th Runge-Kutta", "marker"_a = "x"));
  }

  // rk5
  // {
  //   vector<double> ts({t0});
  //   vector<double> ys({y0});
  //   for (double t = t0; t < tf;) {
  //     auto [y_next, t_next] = rk4(ts.back(), ys.back(), h, dy);
  //     ys.push_back(y_next);
  //     ts.push_back(t_next);
  //     t = t_next;
  //   }
  //   ax.plot(Args(ts, ys),
  //           Kwargs("color"_a = "purple", "linewidth"_a = 1.0,
  //                  "label"_a = "4th Runge-Kutta", "marker"_a = "o"));
  // }

  // ode45
  {
    vector<double> ts({t0});
    vector<double> ys({y0});
    double h_ode45 = h;
    const double tol = 0.1;
    for (double t = t0; t < tf;) {
      auto [y_next, t_next, h_next] =
          ode45(ts.back(), ys.back(), h_ode45, tol, dy);
      ys.push_back(y_next);
      ts.push_back(t_next);
      h_ode45 = h_next;
      t = t_next;
    }
    ax.scatter(Args(ts, ys), Kwargs("color"_a = "blue", "label"_a = "ode45",
                                    "facecolor"_a = "None", "s"_a = 50));
  }

  ax.legend();
  ax.grid(Args(true));
  plt.show();
}

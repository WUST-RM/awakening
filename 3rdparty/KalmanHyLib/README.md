# 🚀 KalmanHyLib: A Hybrid Kalman Filter Library

**KalmanHyLib** is a lightweight, extensible, and **header-only** C++ library that implements advanced iterative Kalman filter variants, including:

- 🔁 **IEKF** – *Iterated Extended Kalman Filter*  
- 🔄 **IESEKF** – *Iterated Error-State Extended Kalman Filter*  
- ➕ **IAEKF** – *Iterated Adaptive Extended Kalman Filter*  
- 🧭 **IUKF** – *Iterated Unscented Kalman Filter*

The library is ideal for systems with **nonlinear dynamics and measurements**, where high-precision estimation and **iterative refinement** are critical.

---

## ✨ Features

- ✅ **Header-Only Design**  
  Integrate instantly – no compilation or linking needed.

- 🧮 **Ceres Jet-Compatible**  
  Leverages [`ceres::Jet`](http://ceres-solver.org/automatic_derivatives.html) for automatic differentiation of models and Jacobians.

- 🧩 **Modular Architecture**  
  Unified interfaces with easy switching between filtering strategies.

- ⚡ **Optimized Performance**  
  Template-based C++ and **Eigen3** acceleration ensure runtime efficiency.

- 🔁 **Iterative Convergence**  
  Built-in support for residual-based iterative updates.

---

## 🧠 Applications

KalmanHyLib is suitable for:

- 📷 Visual-Inertial Odometry (VIO) & SLAM  
- 🎯 Target tracking and motion prediction  
- 🧭 Navigation, control, and guidance systems  
- 🤖 Robotic estimation and autonomy  
- 🔗 Complex multi-sensor fusion tasks

---

## 🔌 Real-World Usage

KalmanHyLib is actively used in the following real-time robotic vision project:

### [WUST-RM/wust_vision](https://github.com/WUST-RM/wust_vision)

> ⚙️ *A real-time vision system for RoboMaster armor detection, 3D pose estimation, and state prediction using hybrid Kalman filtering.*

In this project, **KalmanHyLib** is used to:

- Filter yaw/pitch/distance observations with iterative EKF variants
- Integrate PnP uncertainty and angle wrapping into estimation logic
- Perform prediction and sensor fusion for dynamic target tracking

---

## 🧪 Example

Here's a simple example of using KalmanHyLib from the `wust_vision` project:

```cpp
#include "KalmanHyLib/kalman_hybird_lib.hpp"

enum class MotionModel {
    CONSTANT_VELOCITY = 0,
    CONSTANT_ROTATION = 1,
    CONSTANT_VEL_ROT = 2
};

constexpr int X_N = 10, Z_N = 4;

struct Predict {
    Predict(double dt, MotionModel model, double vrx = 0.0, double vry = 0.0, double vrz = 0.0)
        : dt(dt), model(model), vrx(vrx), vry(vry), vrz(vrz) {}

    template <typename T>
    void operator()(const T x0[X_N], T x1[X_N]) {
        for (int i = 0; i < X_N; i++) x1[i] = x0[i];
        if (model == MotionModel::CONSTANT_VEL_ROT || model == MotionModel::CONSTANT_VELOCITY) {
            x1[0] += x0[1] * dt;
            x1[2] += x0[3] * dt;
            x1[4] += x0[5] * dt;
        } else {
            x1[1] = T(0.); x1[3] = T(0.); x1[5] = T(0.);
        }
        x1[0] -= T(vrx) * T(dt);
        x1[2] -= T(vry) * T(dt);
        x1[4] -= T(vrz) * T(dt);
        if (model == MotionModel::CONSTANT_VEL_ROT || model == MotionModel::CONSTANT_ROTATION)
            x1[6] += x0[7] * dt;
        else
            x1[7] = T(0.);
    }

    double dt;
    MotionModel model;
    double vrx, vry, vrz;
};

struct Measure {
    template <typename T>
    void operator()(const T x[X_N], T z[Z_N]) {
        T armor_x = x[0] - ceres::cos(x[6]) * x[8];
        T armor_y = x[2] - ceres::sin(x[6]) * x[8];
        T armor_z = x[4] + x[9];
        T xy_dist = ceres::sqrt(armor_x * armor_x + armor_y * armor_y);
        T dist = ceres::sqrt(xy_dist * xy_dist + armor_z * armor_z);
        z[0] = ceres::atan2(armor_y, armor_x);
        z[1] = ceres::atan2(armor_z, xy_dist);
        z[2] = dist;
        z[3] = x[6];
    }
};

using RobotStateEKF = kalman_hybird_lib::ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;
using RobotStateESEKF = kalman_hybird_lib::ErrorStateEKF<X_N, Z_N, Predict, Measure>;

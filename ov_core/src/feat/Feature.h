/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_CORE_FEATURE_H
#define OV_CORE_FEATURE_H

#include <Eigen/Eigen>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace ov_core {

/**
 * @brief Sparse feature class used to collect measurements
 *
 * This feature class allows for holding of all tracking information for a given feature.
 * Each feature has a unique ID assigned to it, and should have a set of feature tracks alongside it.
 * See the FeatureDatabase class for details on how we load information into this, and how we delete features.
 */
class Feature {

public:
  /// Unique ID of this feature
  // 特征点的ID
  size_t featid;

  /// If this feature should be deleted
  // 是否从数据库中删除的标志位
  bool to_delete;

  /// UV coordinates that this feature has been seen from (mapped by camera ID)
  // 特征点在每一次被观测到时的像素坐标
  std::unordered_map<size_t, std::vector<Eigen::VectorXf>> uvs;

  /// UV normalized coordinates that this feature has been seen from (mapped by camera ID)
  // 特征点在每一次被观测到时的去畸变归一化坐标
  std::unordered_map<size_t, std::vector<Eigen::VectorXf>> uvs_norm;

  /// Timestamps of each UV measurement (mapped by camera ID)
  // 同一个feature被不同camera在不同时间看到，key对应不同的camera，value:timestamps装着相同camera的不同的时间
  // 特征点在每一次被观测到时的时间戳信息，后面MSCKF过程中计算特征点的生命周期只需要计算vector的size就可以
  std::unordered_map<size_t, std::vector<double>> timestamps;

  /// What camera ID our pose is anchored in!! By default the first measurement is the anchor.
  // 特征点坐标相对于哪一个局部坐标系，默认是第一帧
  int anchor_cam_id = -1;

  /// Timestamp of anchor clone
  // anchor 对应的时间点
  double anchor_clone_timestamp;

  /// Triangulated position of this feature, in the anchor frame
  // 在局部坐标系anchor frame 的三维坐标，三角化之后的（anchor frame：anchor对应的相机坐标系下，feature到anchor相机的位置）
  Eigen::Vector3d p_FinA;

  /// Triangulated position of this feature, in the global frame
  // 在全局坐标的三角化Pose
  Eigen::Vector3d p_FinG;

  /**
   * @brief Remove measurements that do not occur at passed timestamps.
   *
   * Given a series of valid timestamps, this will remove all measurements that have not occurred at these times.
   * This would normally be used to ensure that the measurements that we have occur at our clone times.
   *
   * @param valid_times Vector of timestamps that our measurements must occur at
   */
  void clean_old_measurements(const std::vector<double> &valid_times);

  /**
   * @brief Remove measurements that occur at the invalid timestamps
   *
   * Given a series of invalid timestamps, this will remove all measurements that have occurred at these times.
   *
   * @param invalid_times Vector of timestamps that our measurements should not
   */
  void clean_invalid_measurements(const std::vector<double> &invalid_times);

  /**
   * @brief Remove measurements that are older then the specified timestamp.
   *
   * Given a valid timestamp, this will remove all measurements that have occured earlier then this.
   *
   * @param timestamp Timestamps that our measurements must occur after
   */
  void clean_older_measurements(double timestamp);
};

} // namespace ov_core

#endif /* OV_CORE_FEATURE_H */
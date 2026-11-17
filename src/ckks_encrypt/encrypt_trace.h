//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// =========================================================
// Time Logger
// =========================================================
struct TimeEvent {
  using string = std::string;
  double start;
  double end;
  string device;
  string name;
  string var_name;
  string calc_name;
  int level;
  double aie_end = -1;  // npu events: when the AIE run completed (-1 if unknown)

  void print_for_csv() {
    std::cout << device << ", " << var_name << ", " << calc_name << ", " << level << ", " << start << ", " << end << ", " << (end - start) << ", " << aie_end << "\n";
  }
};

class TimeLogger {
public:
  using clock = std::chrono::steady_clock;
  std::vector<TimeEvent> events;
  bool use_logger = false;
  double base;

  TimeLogger() {
    base = std::chrono::duration<double, std::micro>(clock::now().time_since_epoch()).count();
  }

  int start_event(const std::string &device, const std::string &var_name, const std::string &calc_name, int level) {
    if (!use_logger) return -1;
    TimeEvent event;
    event.start = std::chrono::duration<double, std::micro>(clock::now().time_since_epoch()).count() - base;
    event.device = device;
    event.var_name = var_name;
    event.calc_name = calc_name;
    event.level = level;
    event.name = var_name + "_" + calc_name + "_lv." + std::to_string(level);
    events.push_back(event);
    return events.size() - 1;
  }

  // The host waits for a run only after its overlapped work, so stop_event()
  // marks when the host noticed completion. While logging, a watcher thread
  // waits for the run instead and records when the AIE actually finished;
  // the host then joins the watcher, so only one thread ever waits on a run.
  struct AieWatch {
    std::thread thread;
    double done = -1;
    std::atomic<bool> finished{false};  // for the host to poll
    ~AieWatch() { if (thread.joinable()) thread.join(); }
  };
  template <class Run> void watch(Run &run, AieWatch &w) {
    if (!use_logger) return;
    w.thread = std::thread([this, &run, &w] { run.wait(); w.done = now(); w.finished = true; });
  }
  template <class Run> void wait_run(Run &run, AieWatch &w, int idx) {
    if (!w.thread.joinable()) { run.wait(); return; }
    w.thread.join();
    if (idx >= 0) events[idx].aie_end = w.done;
  }
  double now() const {
    return std::chrono::duration<double, std::micro>(clock::now().time_since_epoch()).count() - base;
  }

  void stop_event(int idx) {
    if (!use_logger || idx < 0) return;
    events[idx].end = std::chrono::duration<double, std::micro>(clock::now().time_since_epoch()).count() - base;
  }

  void print_summary() {
    if (!use_logger) return;
    std::cout << "Time Logger Summary:\n";
    std::cout << "Device,var_name,calc_name,level,Start,End,Duration(us),AIE_end\n";
    // prevent e+ notation
    std::cout << std::fixed << std::setprecision(3);
    for (auto &event : events) {
      event.print_for_csv();
    }
  }
};

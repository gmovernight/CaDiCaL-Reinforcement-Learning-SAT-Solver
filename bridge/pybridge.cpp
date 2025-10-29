// First include pybind and STL
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Access CaDiCaL internals (private) in this TU only.
#define private public
#define protected public
#include "cadical.hpp"
#include "internal.hpp"
#include "resources.hpp"
#undef private
#undef protected

namespace py = pybind11;

// Status codes to mirror typical SAT solver conventions.
static inline int STATUS_UNKNOWN = 0;
static inline int STATUS_SAT     = 10;
static inline int STATUS_UNSAT   = 20;

// ------------------------- StubSolver (bridged) ---------------------------
class StubSolver {
public:
  StubSolver() { reset(); }

  void reset() {
    solver_.reset(new CaDiCaL::Solver());
    loaded_path_.clear();
    last_vars_.clear();
    last_scores_.clear();
    action_pending_ = false;
    last_action_var_ = 0;
  }

  void load_dimacs(const std::string& path) {
    if (!solver_) reset();
    int vars = 0;
    const char* err = solver_->read_dimacs(path.c_str(), vars, /*strict=*/1);
    if (err) throw std::runtime_error(std::string("read_dimacs error: ") + err);
    loaded_path_ = path;
    // Reset RL counters/flags inside the solver.
    solver_->internal->rl_next_var    = 0;
    solver_->internal->rl_window_left = 0;
    solver_->internal->rl_M           = 0;
    solver_->internal->rl_step_id     = 0;
    last_vars_.clear();
    last_scores_.clear();
    action_pending_ = false;
    last_action_var_ = 0;
  }

  // Run until (just before) next decision. If an RL action is pending
  // (applied through apply_action), perform exactly one decision first,
  // then return the next snapshot.
  py::dict solve_until_next_hook(int K, int timeout_ms) {
    ensure_loaded();
    if (K <= 0) K = 1;

    if (action_pending_) {
      // Ensure we are at a decision boundary then take exactly one decision.
      std::vector<double> _g; std::vector<int> _v; std::vector<double> _s; int _st = 0;
      solver_->internal->rl_until_hook(/*K*/1, /*timeout_ms*/0, _g, _v, _s, _st);
      int decide_res = solver_->internal->decide();
      (void) decide_res;
      action_pending_ = false;
    }

    // Produce the observation at the next decision point.
    std::vector<double> g; std::vector<int> v; std::vector<double> s; int st = 0;
    solver_->internal->rl_until_hook(K, timeout_ms, g, v, s, st);

    // Store last candidates for index->var resolution.
    last_vars_ = v;
    last_scores_ = s;

    // Build Python objects.
    py::list py_g;
    for (double x : g) py_g.append(x);

    py::list py_cands;
    for (size_t i = 0; i < v.size(); ++i) {
      py::dict d;
      d["var"] = v[i];
      d["evsids"] = (i < s.size() ? s[i] : 0.0);
      d["rank"] = static_cast<int>(i);
      py_cands.append(d);
    }

    py::dict out;
    out["global"] = py_g;
    out["cands"]  = py_cands;
    out["step_id"] = static_cast<long long>(solver_->internal->rl_step_id);
    out["status"]  = st;
    return out;
  }

  void apply_action(int action_index, int M) {
    ensure_loaded();
    if (action_index < 0 || action_index >= static_cast<int>(last_vars_.size()))
      throw std::runtime_error("apply_action: action_index out of range (call solve_until_next_hook first)");
    if (M <= 0)
      throw std::runtime_error("apply_action: M must be positive");

    int var = last_vars_[action_index];
    solver_->internal->rl_next_var    = var;
    solver_->internal->rl_window_left = M;
    solver_->internal->rl_M           = M;
    last_action_var_ = var;
    action_pending_  = true;
  }

  py::dict get_metrics() const {
    ensure_loaded();
    const auto *I = solver_->internal;
    py::dict m;
    const double t = CaDiCaL::absolute_process_time();
    const double decisions = (double) I->stats.decisions;
    const double props     = (double) I->stats.propagations.search;
    m["time_s"] = t;
    m["conflicts"] = (long long) I->stats.conflicts;
    m["decisions"] = (long long) I->stats.decisions;
    m["props_per_dec"] = decisions > 0.0 ? (props / decisions) : 0.0;
    m["restarts"] = (long long) I->stats.restarts;
    m["props"] = (long long) I->stats.propagations.search;
    m["last_action_var"] = last_action_var_;
    m["window_left"] = I->rl_window_left;
    // Expose moving-average LBD signals if available
    try {
      m["meanLBD_fast"] = I->averages.current.glue.fast.value;
      m["meanLBD_slow"] = I->averages.current.glue.slow.value;
    } catch (...) {
      // leave absent if structure changes
    }
    return m;
  }

  static const char* version() { return "bridge-api-r2"; }

private:
  void ensure_loaded() const {
    if (!solver_) throw std::runtime_error("no solver (call reset())");
    if (loaded_path_.empty()) throw std::runtime_error("no instance loaded (call load_dimacs(path))");
  }

  std::unique_ptr<CaDiCaL::Solver> solver_;
  std::string loaded_path_;
  std::vector<int> last_vars_;
  std::vector<double> last_scores_;
  bool action_pending_ = false;
  int  last_action_var_ = 0;
  
};

// Optional compatibility wrapper that maintains (K,M) defaults.
class BridgeSolver {
public:
  BridgeSolver(): K_default_(16), M_default_(50) {}
  void reset() { core_.reset(); }
  void load_dimacs(const std::string& path, int K, int M) {
    if (K <= 0 || M <= 0) throw std::runtime_error("BridgeSolver.load_dimacs: K and M must be positive");
    K_default_ = K; M_default_ = M; core_.load_dimacs(path);
  }
  py::dict solve_until_next_hook(int K = -1, int timeout_ms = 500) {
    return core_.solve_until_next_hook(K > 0 ? K : K_default_, timeout_ms);
  }
  void apply_action(int action_index, int M = -1) {
    core_.apply_action(action_index, M > 0 ? M : M_default_);
  }
  py::dict get_metrics() const { return core_.get_metrics(); }
  static const char* version() { return StubSolver::version(); }
private:
  int K_default_;
  int M_default_;
  StubSolver core_;
};

// ------------------------- Module ---------------------------------
PYBIND11_MODULE(satrl_bridge, m) {
  m.doc() = "SAT RL bridge (pybind; hooks + actions).";
  m.attr("BUILD_ID") = "bridge-api-r2";

  py::class_<StubSolver>(m, "StubSolver",
    R"doc(RL-controllable CaDiCaL bridge.

Methods
-------
reset()
    Clear runtime state and create a fresh solver.
load_dimacs(path: str)
    Read DIMACS CNF into the solver.
solve_until_next_hook(K: int, timeout_ms: int)
    Run until the next decision point and return observation.
apply_action(action_index: int, M: int)
    Force the next decision to be candidate[index], with RL window M.
get_metrics() -> dict
    Return time_s, conflicts, decisions, props_per_dec, restarts, last_action_var, window_left.
)doc")
      .def(py::init<>())
      .def("reset", &StubSolver::reset)
      .def("load_dimacs", &StubSolver::load_dimacs, py::arg("path"))
      .def("solve_until_next_hook", &StubSolver::solve_until_next_hook,
           py::arg("K"), py::arg("timeout_ms") = 500)
      .def("apply_action", &StubSolver::apply_action, py::arg("action_index"), py::arg("M"))
      .def("get_metrics", &StubSolver::get_metrics)
      .def_static("version", &StubSolver::version);

  py::class_<BridgeSolver>(m, "BridgeSolver")
      .def(py::init<>())
      .def("reset", &BridgeSolver::reset)
      .def("load_dimacs", &BridgeSolver::load_dimacs,
           py::arg("path"), py::arg("K"), py::arg("M"))
      .def("solve_until_next_hook", &BridgeSolver::solve_until_next_hook,
           py::arg("K") = -1, py::arg("timeout_ms") = 500)
      .def("apply_action", &BridgeSolver::apply_action,
           py::arg("action_index"), py::arg("M") = -1)
      .def("get_metrics", &BridgeSolver::get_metrics)
      .def_static("version", &BridgeSolver::version);

  m.def("version", [](){ return std::string(StubSolver::version()); });
}

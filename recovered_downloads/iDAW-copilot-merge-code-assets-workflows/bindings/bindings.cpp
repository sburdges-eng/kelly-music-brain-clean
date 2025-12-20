#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declarations for submodule bindings
void bind_harmony(py::module_& m);
void bind_groove(py::module_& m);
void bind_diagnostics(py::module_& m);
void bind_osc(py::module_& m);

PYBIND11_MODULE(penta_core_native, m) {
    m.doc() = "Penta Core C++ engine with Python bindings";
    
    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("__cpp_standard__") = __cplusplus;
    
    // Create submodules
    auto harmony = m.def_submodule("harmony", "Harmony analysis module");
    auto groove = m.def_submodule("groove", "Groove analysis module");
    auto diagnostics = m.def_submodule("diagnostics", "Performance diagnostics module");
    auto osc = m.def_submodule("osc", "OSC communication module");
    
    // Bind submodules
    bind_harmony(harmony);
    bind_groove(groove);
    bind_diagnostics(diagnostics);
    bind_osc(osc);
}

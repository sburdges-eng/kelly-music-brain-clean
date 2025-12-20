#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <stdexcept>
#include "penta/osc/OSCHub.h"
#include "penta/osc/OSCMessage.h"
#include "penta/osc/RTMessageQueue.h"

namespace py = pybind11;
using namespace penta::osc;

void bind_osc(py::module_& m) {
    // OSCMessage structure
    py::class_<OSCMessage>(m, "OSCMessage")
        .def(py::init<>())
        .def_property("address",
            [](const OSCMessage& msg) { return msg.getAddress(); },
            [](OSCMessage& msg, const std::string& addr) { msg.setAddress(addr); })
        .def_property("timestamp",
            &OSCMessage::getTimestamp,
            &OSCMessage::setTimestamp)
        .def_property_readonly("argument_count", &OSCMessage::getArgumentCount)
        .def("add_int", &OSCMessage::addInt, py::arg("value"))
        .def("add_float", &OSCMessage::addFloat, py::arg("value"))
        .def("add_string",
            py::overload_cast<const std::string&>(&OSCMessage::addString),
            py::arg("value"))
        .def("add_string",
            [](OSCMessage& msg, const char* value) {
                // Delegate through std::string overload to avoid relying on the char* symbol export.
                msg.addString(std::string(value));
            },
            py::arg("value"))
        .def("clear", &OSCMessage::clear)
        .def("get_argument", [](const OSCMessage& msg, size_t index) -> py::object {
            try {
                const auto& arg = msg.getArgument(index);
                if (std::holds_alternative<int32_t>(arg)) {
                    return py::cast(std::get<int32_t>(arg));
                }
                if (std::holds_alternative<float>(arg)) {
                    return py::cast(std::get<float>(arg));
                }
                if (std::holds_alternative<std::string>(arg)) {
                    return py::cast(std::get<std::string>(arg));
                }
                if (std::holds_alternative<std::vector<uint8_t>>(arg)) {
                    const auto& blob = std::get<std::vector<uint8_t>>(arg);
                    return py::bytes(reinterpret_cast<const char*>(blob.data()), blob.size());
                }
            } catch (const std::out_of_range&) {
                throw py::index_error("Argument index out of range");
            }
            return py::none();
        }, py::arg("index"))
        .def("__repr__", [](const OSCMessage& msg) {
            return "OSCMessage(address='" + msg.getAddress() +
                   "', args=" + std::to_string(msg.getArgumentCount()) + ")";
        });
    
    // OSCHub configuration
    py::class_<OSCHub::Config>(m, "OSCConfig")
        .def(py::init<>())
        .def_readwrite("server_address", &OSCHub::Config::serverAddress)
        .def_readwrite("server_port", &OSCHub::Config::serverPort)
        .def_readwrite("client_address", &OSCHub::Config::clientAddress)
        .def_readwrite("client_port", &OSCHub::Config::clientPort)
        .def_readwrite("queue_size", &OSCHub::Config::queueSize);
    
    // OSCHub
    py::class_<OSCHub>(m, "OSCHub")
        .def(py::init<const OSCHub::Config&>(),
            py::arg("config") = OSCHub::Config{})
        .def("start", &OSCHub::start,
            "Start OSC server and client")
        .def("stop", &OSCHub::stop,
            "Stop OSC server and client")
        .def("send_message", &OSCHub::sendMessage,
            py::arg("message"),
            "Send OSC message (RT-safe)")
        .def("receive_message", [](OSCHub& self) -> py::object {
            OSCMessage msg;
            if (self.receiveMessage(msg)) {
                return py::cast(msg);
            }
            return py::none();
        }, "Receive OSC message (RT-safe, returns None if no message)")
        .def("register_callback", &OSCHub::registerCallback,
            py::arg("pattern"), py::arg("callback"),
            "Register callback for OSC address pattern")
        .def("update_config", &OSCHub::updateConfig,
            py::arg("config"),
            "Update OSC configuration");
    
    // Convenience functions for creating messages
    m.def("create_osc_message", [](const std::string& address) {
        OSCMessage msg(address);
        return msg;
    }, py::arg("address"), "Create a new OSC message");
}

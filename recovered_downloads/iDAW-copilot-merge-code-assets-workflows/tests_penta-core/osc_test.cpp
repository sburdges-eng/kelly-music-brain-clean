#include "penta/osc/OSCServer.h"
#include "penta/osc/OSCClient.h"
#include "penta/osc/OSCHub.h"
#include "penta/osc/RTMessageQueue.h"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <variant>

using namespace penta::osc;

// ========== RTMessageQueue Tests ==========

class RTMessageQueueTest : public ::testing::Test {
protected:
    RTMessageQueue<OSCMessage> queue{1024};
};

TEST_F(RTMessageQueueTest, PushAndPop) {
    OSCMessage msg;
    msg.setAddress("/test");
    msg.addFloat(42.0f);
    
    EXPECT_TRUE(queue.push(msg));
    
    OSCMessage retrieved;
    EXPECT_TRUE(queue.pop(retrieved));
    EXPECT_EQ(retrieved.getAddress(), "/test");
    EXPECT_EQ(retrieved.getArgumentCount(), 1u);
    EXPECT_FLOAT_EQ(std::get<float>(retrieved.getArgument(0)), 42.0f);
}

TEST_F(RTMessageQueueTest, FIFOOrder) {
    OSCMessage msg1, msg2, msg3;
    msg1.setAddress("/first");
    msg2.setAddress("/second");
    msg3.setAddress("/third");
    
    queue.push(msg1);
    queue.push(msg2);
    queue.push(msg3);
    
    OSCMessage retrieved;
    queue.pop(retrieved);
    EXPECT_EQ(retrieved.getAddress(), "/first");
    
    queue.pop(retrieved);
    EXPECT_EQ(retrieved.getAddress(), "/second");
    
    queue.pop(retrieved);
    EXPECT_EQ(retrieved.getAddress(), "/third");
}

TEST_F(RTMessageQueueTest, EmptyQueueReturnsFalse) {
    OSCMessage msg;
    EXPECT_FALSE(queue.pop(msg));
}

TEST_F(RTMessageQueueTest, ClearWorks) {
    OSCMessage msg;
    msg.setAddress("/test");
    
    queue.push(msg);
    queue.clear();
    
    EXPECT_FALSE(queue.pop(msg));
}

// ========== OSCServer Tests ==========

class OSCServerTest : public ::testing::Test {
protected:
    void SetUp() override {
        server = std::make_unique<OSCServer>("127.0.0.1", 9001);
    }
    
    void TearDown() override {
        if (server) {
            server->stop();
        }
    }
    
    std::unique_ptr<OSCServer> server;
};

TEST_F(OSCServerTest, StartsAndStops) {
    EXPECT_NO_THROW(server->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_NO_THROW(server->stop());
}

TEST_F(OSCServerTest, RegistersHandler) {
    bool handlerCalled = false;
    
    server->registerHandler("/test", [&handlerCalled](const OSCMessage& msg) {
        handlerCalled = true;
    });
    
    EXPECT_TRUE(server->hasHandler("/test"));
}

TEST_F(OSCServerTest, UnregistersHandler) {
    server->registerHandler("/test", [](const OSCMessage&) {});
    EXPECT_TRUE(server->hasHandler("/test"));
    
    server->unregisterHandler("/test");
    EXPECT_FALSE(server->hasHandler("/test"));
}

TEST_F(OSCServerTest, ReceivesMessage) {
    std::atomic<bool> received{false};
    
    server->registerHandler("/hello", [&received](const OSCMessage& msg) {
        received = true;
    });
    
    server->start();
    
    // Send message from client
    OSCClient client("127.0.0.1", 9001);
    client.start();
    
    OSCMessage msg;
    msg.setAddress("/hello");
    msg.addFloat(123.0f);
    client.send(msg);
    
    // Wait for message processing
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    client.stop();
    server->stop();
    
    EXPECT_TRUE(received);
}

// ========== OSCClient Tests ==========

class OSCClientTest : public ::testing::Test {
protected:
    void SetUp() override {
        client = std::make_unique<OSCClient>("127.0.0.1", 9002);
    }
    
    void TearDown() override {
        if (client) {
            client->stop();
        }
    }
    
    std::unique_ptr<OSCClient> client;
};

TEST_F(OSCClientTest, StartsAndStops) {
    EXPECT_NO_THROW(client->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_NO_THROW(client->stop());
}

TEST_F(OSCClientTest, SendsMessage) {
    OSCMessage msg;
    msg.setAddress("/test");
    msg.addFloat(42.0f);
    
    client->start();
    EXPECT_TRUE(client->send(msg));
    client->stop();
}

TEST_F(OSCClientTest, FailsWhenNotStarted) {
    OSCMessage msg;
    msg.setAddress("/test");
    
    EXPECT_FALSE(client->send(msg));
}

// ========== OSCHub Tests ==========

class OSCHubTest : public ::testing::Test {
protected:
    void SetUp() override {
        hub = std::make_unique<OSCHub>("127.0.0.1", 9003, "127.0.0.1", 9004);
    }
    
    void TearDown() override {
        if (hub) {
            hub->stop();
        }
    }
    
    std::unique_ptr<OSCHub> hub;
};

TEST_F(OSCHubTest, StartsAndStops) {
    EXPECT_NO_THROW(hub->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_NO_THROW(hub->stop());
}

TEST_F(OSCHubTest, BidirectionalCommunication) {
    hub->start();
    
    // Create counterpart: server on 9004, client to 9003
    OSCServer remoteServer("127.0.0.1", 9004);
    OSCClient remoteClient("127.0.0.1", 9003);
    
    std::atomic<bool> hubReceived{false};
    std::atomic<bool> remoteReceived{false};
    
    hub->registerHandler("/to_hub", [&hubReceived](const OSCMessage&) {
        hubReceived = true;
    });
    
    remoteServer.registerHandler("/to_remote", [&remoteReceived](const OSCMessage&) {
        remoteReceived = true;
    });
    
    remoteServer.start();
    remoteClient.start();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Hub sends to remote
    OSCMessage toRemote;
    toRemote.setAddress("/to_remote");
    hub->send(toRemote);
    
    // Remote sends to hub
    OSCMessage toHub;
    toHub.setAddress("/to_hub");
    remoteClient.send(toHub);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    remoteClient.stop();
    remoteServer.stop();
    hub->stop();
    
    EXPECT_TRUE(hubReceived);
    EXPECT_TRUE(remoteReceived);
}

// ========== Performance Benchmarks ==========

class OSCPerformanceBenchmark : public ::testing::Test {
protected:
    OSCClient client{"127.0.0.1", 9005};
    OSCMessage testMsg;
    
    void SetUp() override {
        testMsg.setAddress("/benchmark");
        testMsg.addFloat(1.0f);
        testMsg.addFloat(2.0f);
        testMsg.addFloat(3.0f);
        
        client.start();
    }
    
    void TearDown() override {
        client.stop();
    }
};

TEST_F(OSCPerformanceBenchmark, SendLatency) {
    constexpr int iterations = 1000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        client.send(testMsg);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgMicros = static_cast<double>(duration.count()) / iterations;
    
    std::cout << "Average OSC send time: " << avgMicros << " μs\n";
    
    EXPECT_LT(avgMicros, 100.0);  // Target: <100μs per send
}

TEST_F(OSCPerformanceBenchmark, MessageQueueThroughput) {
    RTMessageQueue<OSCMessage> queue{10000};
    constexpr int iterations = 10000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        queue.push(testMsg);
    }
    
    auto pushEnd = std::chrono::high_resolution_clock::now();
    
    OSCMessage retrieved;
    for (int i = 0; i < iterations; ++i) {
        queue.pop(retrieved);
    }
    
    auto popEnd = std::chrono::high_resolution_clock::now();
    
    auto pushDuration = std::chrono::duration_cast<std::chrono::microseconds>(pushEnd - start);
    auto popDuration = std::chrono::duration_cast<std::chrono::microseconds>(popEnd - pushEnd);
    
    double avgPushMicros = static_cast<double>(pushDuration.count()) / iterations;
    double avgPopMicros = static_cast<double>(popDuration.count()) / iterations;
    
    std::cout << "Average queue push: " << avgPushMicros << " μs\n";
    std::cout << "Average queue pop: " << avgPopMicros << " μs\n";
    
    EXPECT_LT(avgPushMicros, 1.0);  // Lock-free should be <1μs
    EXPECT_LT(avgPopMicros, 1.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

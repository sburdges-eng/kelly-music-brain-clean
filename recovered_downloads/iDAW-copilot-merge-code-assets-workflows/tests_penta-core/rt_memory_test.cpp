#include <gtest/gtest.h>
#include "penta/common/RTMemoryPool.h"
#include <thread>
#include <vector>

using namespace penta;

TEST(RTMemoryPoolTest, AllocateAndDeallocate) {
    RTMemoryPool pool(64, 10);
    
    void* ptr = pool.allocate();
    ASSERT_NE(ptr, nullptr);
    
    pool.deallocate(ptr);
    
    EXPECT_EQ(pool.getAvailableBlocks(), 10);
}

TEST(RTMemoryPoolTest, ExhaustsPool) {
    RTMemoryPool pool(64, 5);
    
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i) {
        void* ptr = pool.allocate();
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }
    
    // Should be exhausted
    void* ptr = pool.allocate();
    EXPECT_EQ(ptr, nullptr);
    
    // Free one
    pool.deallocate(ptrs[0]);
    
    // Should succeed now
    ptr = pool.allocate();
    EXPECT_NE(ptr, nullptr);
}

TEST(RTMemoryPoolTest, ThreadSafety) {
    RTMemoryPool pool(128, 1000);
    
    auto allocDealloc = [&pool]() {
        for (int i = 0; i < 100; ++i) {
            void* ptr = pool.allocate();
            if (ptr) {
                // Simulate some work
                std::this_thread::yield();
                pool.deallocate(ptr);
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(allocDealloc);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // All blocks should be freed
    EXPECT_EQ(pool.getAvailableBlocks(), 1000);
}

struct TestStruct {
    int value;
    TestStruct() : value(42) {}
    ~TestStruct() { value = -1; }
};

TEST(RTPoolPtrTest, RAII) {
    RTMemoryPool pool(sizeof(TestStruct), 10);
    
    {
        RTPoolPtr<TestStruct> ptr(pool);
        ASSERT_TRUE(ptr);
        EXPECT_EQ(ptr->value, 42);
    }  // Should deallocate here
    
    EXPECT_EQ(pool.getAvailableBlocks(), 10);
}

TEST(RTPoolPtrTest, Move) {
    RTMemoryPool pool(sizeof(TestStruct), 10);
    
    RTPoolPtr<TestStruct> ptr1(pool);
    ASSERT_TRUE(ptr1);
    
    RTPoolPtr<TestStruct> ptr2(std::move(ptr1));
    EXPECT_FALSE(static_cast<bool>(ptr1));
    EXPECT_TRUE(ptr2);
    EXPECT_EQ(ptr2->value, 42);
}

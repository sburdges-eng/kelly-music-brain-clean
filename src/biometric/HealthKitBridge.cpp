#include "biometric/HealthKitBridge.h"
#include <juce_core/juce_core.h>
#include <algorithm>
#include <numeric>
#include <cmath>

#if HEALTHKIT_AVAILABLE
// Objective-C bridge for HealthKit
#import <HealthKit/HealthKit.h>

@interface HealthKitBridgeImpl : NSObject
@property (nonatomic, strong) HKHealthStore* healthStore;
@property (nonatomic, assign) BOOL authorized;
@property (nonatomic, copy) void (^heartRateCallback)(float heartRate, float hrv, NSTimeInterval timestamp);
@end

@implementation HealthKitBridgeImpl

- (instancetype)init {
    self = [super init];
    if (self) {
        self.healthStore = [[HKHealthStore alloc] init];
        self.authorized = NO;
    }
    return self;
}

- (void)requestAuthorizationWithCompletion:(void (^)(BOOL success))completion {
    if (![HKHealthStore isHealthDataAvailable]) {
        if (completion) completion(NO);
        return;
    }

    NSSet* readTypes = [NSSet setWithObjects:
                       [HKObjectType quantityTypeForIdentifier:HKQuantityTypeIdentifierHeartRate],
                       [HKObjectType quantityTypeForIdentifier:HKQuantityTypeIdentifierHeartRateVariabilitySDNN],
                       [HKObjectType quantityTypeForIdentifier:HKQuantityTypeIdentifierRestingHeartRate],
                       nil];

    [self.healthStore requestAuthorizationToShareTypes:nil readTypes:readTypes completion:^(BOOL success, NSError* error) {
        self.authorized = success;
        if (completion) completion(success);
    }];
}

- (void)fetchLatestHeartRateWithCompletion:(void (^)(float heartRate, float hrv, NSTimeInterval timestamp))completion {
    HKSampleType* heartRateType = [HKQuantityType quantityTypeForIdentifier:HKQuantityTypeIdentifierHeartRate];
    NSSortDescriptor* sortDescriptor = [NSSortDescriptor sortDescriptorWithKey:HKSampleSortIdentifierEndDate ascending:NO];

    HKSampleQuery* query = [[HKSampleQuery alloc] initWithSampleType:heartRateType
                                                           predicate:nil
                                                               limit:1
                                                     sortDescriptors:@[sortDescriptor]
                                                      resultsHandler:^(HKSampleQuery* query, NSArray* results, NSError* error) {
        if (error || results.count == 0) {
            if (completion) completion(0.0, 0.0, 0.0);
            return;
        }

        HKQuantitySample* sample = (HKQuantitySample*)results.firstObject;
        HKQuantity* quantity = sample.quantity;
        float heartRate = [quantity doubleValueForUnit:[HKUnit unitFromString:@"count/min"]];
        NSTimeInterval timestamp = [sample.endDate timeIntervalSince1970];

        // Get HRV
        [self fetchHRVWithCompletion:^(float hrv) {
            if (completion) completion(heartRate, hrv, timestamp);
        }];
    }];

    [self.healthStore executeQuery:query];
}

- (void)fetchHRVWithCompletion:(void (^)(float hrv))completion {
    HKSampleType* hrvType = [HKQuantityType quantityTypeForIdentifier:HKQuantityTypeIdentifierHeartRateVariabilitySDNN];
    NSSortDescriptor* sortDescriptor = [NSSortDescriptor sortDescriptorWithKey:HKSampleSortIdentifierEndDate ascending:NO];

    HKSampleQuery* query = [[HKSampleQuery alloc] initWithSampleType:hrvType
                                                           predicate:nil
                                                               limit:1
                                                     sortDescriptors:@[sortDescriptor]
                                                      resultsHandler:^(HKSampleQuery* query, NSArray* results, NSError* error) {
        if (error || results.count == 0) {
            if (completion) completion(0.0);
            return;
        }

        HKQuantitySample* sample = (HKQuantitySample*)results.firstObject;
        HKQuantity* quantity = sample.quantity;
        float hrv = [quantity doubleValueForUnit:[HKUnit unitFromString:@"ms"]];
        if (completion) completion(hrv);
    }];

    [self.healthStore executeQuery:query];
}

- (void)startHeartRateObserverWithCallback:(void (^)(float, float, NSTimeInterval))callback {
    self.heartRateCallback = callback;

    HKQuantityType* heartRateType = [HKQuantityType quantityTypeForIdentifier:HKQuantityTypeIdentifierHeartRate];

    HKObserverQuery* observerQuery = [[HKObserverQuery alloc] initWithSampleType:heartRateType
                                                                         predicate:nil
                                                                     updateHandler:^(HKObserverQuery* query, HKObserverQueryCompletionHandler completionHandler, NSError* error) {
        if (error) {
            if (completionHandler) completionHandler();
            return;
        }

        [self fetchLatestHeartRateWithCompletion:^(float hr, float hrv, NSTimeInterval ts) {
            if (self.heartRateCallback) {
                self.heartRateCallback(hr, hrv, ts);
            }
            if (completionHandler) completionHandler();
        }];
    }];

    [self.healthStore executeQuery:observerQuery];
}

@end

#endif

namespace kelly {
namespace biometric {

HealthKitBridge::HealthKitBridge() {
#if HEALTHKIT_AVAILABLE
    impl_ = [[HealthKitBridgeImpl alloc] init];
#else
    impl_ = nullptr;
    juce::Logger::writeToLog("HealthKitBridge: HealthKit not available on this platform");
#endif
}

HealthKitBridge::~HealthKitBridge() {
    stopMonitoring();
#if HEALTHKIT_AVAILABLE
    impl_ = nil;
#endif
}

bool HealthKitBridge::requestAuthorization() {
#if HEALTHKIT_AVAILABLE
    if (!impl_) return false;

    __block bool authResult = false;
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

    [impl_ requestAuthorizationWithCompletion:^(BOOL success) {
        authResult = success;
        authorized_ = success;
        dispatch_semaphore_signal(semaphore);
    }];

    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
    return authResult;
#else
    return false;
#endif
}

bool HealthKitBridge::isAvailable() const {
#if HEALTHKIT_AVAILABLE
    return [HKHealthStore isHealthDataAvailable];
#else
    return false;
#endif
}

bool HealthKitBridge::isAuthorized() const {
    return authorized_;
}

HealthKitData HealthKitBridge::getLatestHeartRate() {
    HealthKitData data;

#if HEALTHKIT_AVAILABLE
    if (!impl_ || !authorized_) return data;

    __block HealthKitData result;
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

    [impl_ fetchLatestHeartRateWithCompletion:^(float hr, float hrv, NSTimeInterval ts) {
        result.heartRate = hr;
        result.heartRateVariability = hrv;
        result.timestamp = ts;
        dispatch_semaphore_signal(semaphore);
    }];

    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
    return result;
#else
    return data;
#endif
}

float HealthKitBridge::getHeartRateVariability() {
    auto data = getLatestHeartRate();
    return data.heartRateVariability;
}

float HealthKitBridge::getRestingHeartRate() {
    // Would query resting heart rate from HealthKit
    // For now, use historical average as baseline
    auto baseline = getHistoricalBaseline(7);
    return baseline.restingHeartRate > 0.0f ? baseline.restingHeartRate : 60.0f;
}

void HealthKitBridge::startMonitoring(std::function<void(const HealthKitData&)> callback) {
    if (monitoring_) return;

    dataCallback_ = callback;
    monitoring_ = true;

#if HEALTHKIT_AVAILABLE
    if (!impl_ || !authorized_) return;

    [impl_ startHeartRateObserverWithCallback:^(float hr, float hrv, NSTimeInterval ts) {
        HealthKitData data;
        data.heartRate = hr;
        data.heartRateVariability = hrv;
        data.timestamp = ts;

        if (dataCallback_) {
            dataCallback_(data);
        }

        // Update historical cache
        historicalData_.push_back(data);
        if (historicalData_.size() > 1000) {
            historicalData_.erase(historicalData_.begin());
        }
    }];
#endif
}

void HealthKitBridge::stopMonitoring() {
    monitoring_ = false;
    dataCallback_ = nullptr;
}

HealthKitData HealthKitBridge::getHistoricalBaseline(int days) {
    HealthKitData baseline;

    if (historicalData_.empty()) {
        // Return default if no data
        baseline.heartRate = 70.0f;
        baseline.heartRateVariability = 50.0f;
        baseline.restingHeartRate = 60.0f;
        return baseline;
    }

    // Calculate averages from historical data
    float sumHR = 0.0f;
    float sumHRV = 0.0f;
    int count = 0;

    // Filter by time window (simplified - would use actual timestamps)
    auto cutoff = historicalData_.end();
    if (historicalData_.size() > days * 24) {  // Assume hourly data
        cutoff = historicalData_.end() - (days * 24);
    }

    for (auto it = cutoff; it != historicalData_.end(); ++it) {
        if (it->isValid()) {
            sumHR += it->heartRate;
            sumHRV += it->heartRateVariability;
            count++;
        }
    }

    if (count > 0) {
        baseline.heartRate = sumHR / count;
        baseline.heartRateVariability = sumHRV / count;
        baseline.restingHeartRate = baseline.heartRate * 0.85f;  // Approximate
    }

    return baseline;
}

HealthKitBridge::NormalizationFactors HealthKitBridge::calculateNormalizationFactors() {
    NormalizationFactors factors;

    auto baseline = getHistoricalBaseline(7);

    if (baseline.heartRate > 0.0f) {
        // Normalize HR to 0-1 range based on typical range (40-200 BPM)
        factors.hrScale = 1.0f / 160.0f;  // 200 - 40 = 160
        factors.hrOffset = -40.0f;  // Shift to start at 0
    }

    if (baseline.heartRateVariability > 0.0f) {
        // Normalize HRV to 0-1 range based on typical range (20-100 ms)
        factors.hrvScale = 1.0f / 80.0f;  // 100 - 20 = 80
        factors.hrvOffset = -20.0f;
    }

    return factors;
}

void HealthKitBridge::updateHistoricalData() {
    // Periodic update of historical data cache
    // Would query HealthKit for historical samples
}

} // namespace biometric
} // namespace kelly

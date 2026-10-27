#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include "sesi_inference.h"
#include <cstring>
#include <string>

static thread_local std::string g_last_error;

static void set_error(char* out, size_t cap, const std::string& msg) {
    g_last_error = msg;
    if (out && cap > 0) {
        std::strncpy(out, msg.c_str(), cap - 1);
        out[cap - 1] = '\0';
    }
}

@interface SESIBridge : NSObject
@property(nonatomic, strong) MLModel *model;
@property(nonatomic, assign) int channels;
@property(nonatomic, assign) int height;
@property(nonatomic, assign) int width;
@end

@implementation SESIBridge
@end

extern "C" {

SESIModelOpaque* sesi_create(const char* path, int C, int H, int W,
                             int compute_units,
                             char* error_out, size_t error_capacity) {
    @autoreleasepool {
        if (!path || C <= 0 || H <= 0 || W <= 0) {
            set_error(error_out, error_capacity, "Invalid path or dimensions");
            return nullptr;
        }

        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];

        switch (compute_units) {
            case 1:  cfg.computeUnits = MLComputeUnitsCPUAndGPU; break;
            case 2:  cfg.computeUnits = MLComputeUnitsCPUOnly;   break;
            default: cfg.computeUnits = MLComputeUnitsAll;       break;
        }

        NSError *err = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:url
                                          configuration:cfg
                                                  error:&err];
        if (!model) {
            set_error(error_out, error_capacity,
                      err.localizedDescription.UTF8String ?: "MLModel load failed");
            return nullptr;
        }

        SESIBridge *bridge = [[SESIBridge alloc] init];
        bridge.model = model;
        bridge.channels = C;
        bridge.height = H;
        bridge.width = W;

        // +1 retain: ownership transferred to the C caller
        return (SESIModelOpaque*)CFBridgingRetain(bridge);
    }
}

int sesi_predict(SESIModelOpaque* handle,
                 const float* input, size_t input_count,
                 float* output, size_t output_capacity,
                 size_t* output_count,
                 char* error_out, size_t error_capacity) {
    @autoreleasepool {
        if (!handle || !input || !output || !output_count) {
            set_error(error_out, error_capacity, "Null argument");
            return -1;
        }

        SESIBridge *bridge = (__bridge SESIBridge*)handle;
        const size_t expected =
            (size_t)bridge.channels * bridge.height * bridge.width;

        if (input_count != expected || output_capacity < expected) {
            set_error(error_out, error_capacity, "Buffer size mismatch");
            return -2;
        }

        @synchronized (bridge) {
            NSError *err = nil;

            NSArray<NSNumber*> *shape = @[
                @1, @(bridge.channels), @(bridge.height), @(bridge.width)
            ];

            MLMultiArray *inArray =
                [[MLMultiArray alloc] initWithShape:shape
                                           dataType:MLMultiArrayDataTypeFloat32
                                              error:&err];
            if (!inArray) {
                set_error(error_out, error_capacity,
                          err.localizedDescription.UTF8String ?: "Input alloc failed");
                return -3;
            }

            std::memcpy(inArray.dataPointer, input, expected * sizeof(float));

            MLDictionaryFeatureProvider *provider =
                [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:@{@"input":
                        [MLFeatureValue featureValueWithMultiArray:inArray]}
                                   error:&err];
            if (!provider) {
                set_error(error_out, error_capacity,
                          err.localizedDescription.UTF8String ?: "Provider failed");
                return -4;
            }

            id<MLFeatureProvider> result =
                [bridge.model predictionFromFeatures:provider error:&err];
            if (!result) {
                set_error(error_out, error_capacity,
                          err.localizedDescription.UTF8String ?: "Prediction failed");
                return -5;
            }

            MLMultiArray *outArray =
                [result featureValueForName:@"output"].multiArrayValue;
            if (!outArray) {
                set_error(error_out, error_capacity, "Output 'output' missing/non-array");
                return -6;
            }

            const size_t n = (size_t)outArray.count;
            if (n > output_capacity) {
                set_error(error_out, error_capacity, "Output buffer too small");
                return -7;
            }

            std::memcpy(output, outArray.dataPointer, n * sizeof(float));
            *output_count = n;
            return 0;
        }
    }
}

void sesi_destroy(SESIModelOpaque* handle) {
    if (!handle) return;
    SESIBridge *bridge = (__bridge_transfer SESIBridge*)handle;
    bridge = nil; // ARC releases
}

const char* sesia_last_error(void) {
    return g_last_error.c_str();
}

} // extern "C"

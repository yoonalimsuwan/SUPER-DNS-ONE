// =============================================================================
// SESI Unreal Bridge — gRPC Client Header
// Developer: PAI, Yoon A Limsuwan / MSPS NETWORK
// License: MIT
// =============================================================================
#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "SESIBridge.generated.h"

USTRUCT(BlueprintType)
struct FArtParameters
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float LightingIntensity = 0.5f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FLinearColor LightColor = FLinearColor::White;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float ShadowSoftness = 0.5f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float ColorTemperature = 0.5f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString MentalStateTarget = TEXT("calm");
};

UCLASS(ClassGroup=(SESI), meta=(BlueprintSpawnableComponent))
class SESIUNREAL_API USESIBridge : public UActorComponent
{
    GENERATED_BODY()

public:
    USESIBridge();

    // Connection settings
    UPROPERTY(EditAnywhere, Category="SESI")
    FString Host = TEXT("localhost");

    UPROPERTY(EditAnywhere, Category="SESI")
    int32 Port = 50051;

    // Current art parameters
    UPROPERTY(BlueprintReadOnly, Category="SESI")
    FArtParameters CurrentArtParams;

    UFUNCTION(BlueprintCallable, Category="SESI")
    void ConnectToSESI();

    UFUNCTION(BlueprintCallable, Category="SESI")
    void RequestArtParameters(const FString& MentalState);

    UFUNCTION(BlueprintCallable, Category="SESI")
    void ApplyArtParameters(ALight* SceneLight);

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
    bool bConnected = false;
    // In production: grpc::Channel and stub here
};

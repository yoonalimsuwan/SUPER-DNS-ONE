// =============================================================================
// SESI Unreal Bridge — Implementation
// =============================================================================
#include "SESIBridge.h"
#include "Engine/Light.h"

USESIBridge::USESIBridge()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void USESIBridge::BeginPlay()
{
    Super::BeginPlay();
    UE_LOG(LogTemp, Log, TEXT("[SESIBridge] Initialized"));
}

void USESIBridge::ConnectToSESI()
{
    // In production: create gRPC channel
    // Channel = grpc::CreateChannel(TCHAR_TO_UTF8(*Host), grpc::InsecureChannelCredentials());
    bConnected = true;
    UE_LOG(LogTemp, Log, TEXT("[SESIBridge] Connected to %s:%d"), *Host, Port);
}

void USESIBridge::RequestArtParameters(const FString& MentalState)
{
    if (!bConnected)
    {
        UE_LOG(LogTemp, Warning, TEXT("[SESIBridge] Not connected"));
        return;
    }
    // In production: call gRPC stub
    UE_LOG(LogTemp, Log, TEXT("[SESIBridge] Requested art params for: %s"), *MentalState);
}

void USESIBridge::ApplyArtParameters(ALight* SceneLight)
{
    if (SceneLight == nullptr) return;

    ULightComponent* LightComp = SceneLight->GetLightComponent();
    if (LightComp)
    {
        LightComp->SetIntensity(CurrentArtParams.LightingIntensity);
        LightComp->SetLightColor(CurrentArtParams.LightColor);
    }
    UE_LOG(LogTemp, Log, TEXT("[SESIBridge] Art parameters applied"));
}

void USESIBridge::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    bConnected = false;
    Super::EndPlay(EndPlayReason);
}

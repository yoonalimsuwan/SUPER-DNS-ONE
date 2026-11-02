// =============================================================================
// SESI Unity Bridge — gRPC Client for SESI Art AGI
// Developer: PAI, Yoon A Limsuwan / MSPS NETWORK
// License: MIT
// =============================================================================
using UnityEngine;
using Grpc.Core;
using System.Threading.Tasks;

namespace SESI.Unity
{
    /// <summary>
    /// gRPC client bridge to SESI Art AGI backend.
    /// Place this script on a GameObject in your Unity scene.
    /// </summary>
    public class SESIBridge : MonoBehaviour
    {
        [Header("Connection")]
        [SerializeField] private string _host = "localhost";
        [SerializeField] private int _port = 50051;

        private Channel _channel;
        private bool _connected;

        // Art parameters received from SESI
        public ArtParameters CurrentArtParams { get; private set; }

        void Awake()
        {
            CurrentArtParams = new ArtParameters();
        }

        async void Start()
        {
            await ConnectAsync();
        }

        /// <summary>Establish gRPC channel to SESI backend.</summary>
        public async Task ConnectAsync()
        {
            try
            {
                _channel = new Channel($"{_host}:{_port}", ChannelCredentials.Insecure);
                _connected = true;
                Debug.Log($"[SESIBridge] Connected to {_host}:{_port}");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[SESIBridge] Connection failed: {e.Message}");
                _connected = false;
            }
            await Task.CompletedTask;
        }

        /// <summary>Request therapeutic art parameters from SESI.</summary>
        public async Task RequestArtParameters(string mentalState)
        {
            if (!_connected)
            {
                Debug.LogWarning("[SESIBridge] Not connected");
                return;
            }

            try
            {
                // In production: call generated gRPC stub
                // var client = new SESIService.SESIServiceClient(_channel);
                // var response = await client.GetArtParamsAsync(
                //     new ArtParamsRequest { MentalState = mentalState });
                // CurrentArtParams = MapToArtParameters(response);

                Debug.Log($"[SESIBridge] Requested art params for state: {mentalState}");
            }
            catch (RpcException e)
            {
                Debug.LogError($"[SESIBridge] RPC failed: {e.Status}");
            }
        }

        /// <summary>Apply art parameters to scene lighting/materials.</summary>
        public void ApplyArtParameters(Light sceneLight)
        {
            if (CurrentArtParams == null || sceneLight == null) return;

            sceneLight.intensity = CurrentArtParams.LightingIntensity;
            sceneLight.color = new Color(
                CurrentArtParams.LightColorR,
                CurrentArtParams.LightColorG,
                CurrentArtParams.LightColorB
            );

            Debug.Log("[SESIBridge] Art parameters applied to scene");
        }

        void OnDestroy()
        {
            if (_channel != null)
            {
                _channel.ShutdownAsync().Wait();
            }
        }
    }

    /// <summary>Data class for art parameters.</summary>
    [System.Serializable]
    public class ArtParameters
    {
        public float LightingIntensity = 0.5f;
        public float LightColorR = 1.0f;
        public float LightColorG = 1.0f;
        public float LightColorB = 1.0f;
        public float ShadowSoftness = 0.5f;
        public float ColorTemperature = 0.5f;
        public string MentalStateTarget = "calm";
    }
}

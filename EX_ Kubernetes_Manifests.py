# k8s/production/inference_service.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sesi-protein-fold
  namespace: sesi-serving
  annotations:
    autoscaling.knative.dev/target: "10"
    autoscaling.knative.dev/scaleToZeroPodAutoscaler: "enabled"
spec:
  predictor:
    model:
      modelFormat:
        name: pytorch
        version: "2"
      runtimeVersion: 0.11.0
      storageUri: s3://sesi-models/protein-fold/v1.0.0
      resources:
        requests:
          cpu: "2"
          memory: 8Gi
          nvidia.com/gpu: 1
        limits:
          cpu: "4"
          memory: 16Gi
          nvidia.com/gpu: 1
    minReplicas: 1
    maxReplicas: 10
---
# k8s/production/service_account.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: sesi-protein-fold-sa
  namespace: sesi-serving
---
# k8s/production/config_map.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sesi-protein-fold-config
  namespace: sesi-serving
data:
  model_config.json: |
    {
      "model_name": "sesi-protein-fold",
      "model_version": "1.0.0",
      "model_class": "ProteinFoldingOrchestrator",
      "config": {
        "latent_dim": 256,
        "sequence_dim": 128,
        "structure_dim": 256,
        "n_levels": 3
      }
    }
---
# k8s/production/prometheus_monitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sesi-protein-fold-monitor
  namespace: sesi-serving
spec:
  selector:
    matchLabels:
      serving.kserve.io/inferenceservice: sesi-protein-fold
  endpoints:
  - port: http
    path: /metrics
    interval: 30s

/**
 * NNAPI Profiling Utilities for SESI CFD inference.
 *
 * Provides:
 *  - Multi-backend comparison (CPU vs NNAPI variants)
 *  - ORT session profiling (JSON trace)
 *  - Device capability detection
 *  - Systrace integration
 *
 * HONEST SCOPE:
 *  - NNAPI performance is DEVICE-SPECIFIC; same model can be 10x faster
 *    on one device and slower than CPU on another
 *  - First-run compilation time is excluded from steady-state benchmarks
 *  - Results are relative, not absolute guarantees
 */

package org.mspsnetwork.sesi.inference

import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Device NNAPI capability summary.
 */
data class DeviceCapability(
    val manufacturer: String,
    val model: String,
    val sdkInt: Int,
    val nnapiAvailable: Boolean,
    val nnapiVersion: String,
    val cpuAbi: List<String>,
    val cores: Int,
    val estimatedClass: String,  // LOW / MID / HIGH based on heuristics
)

/**
 * Full NNAPI profiling session.
 */
class NnapiProfiler(private val context: Context) {

    companion object {
        private const val TAG = "NnapiProfiler"
    }

    private val model: SesiOnnxModel by lazy { SesiOnnxModel(context) }
    private val results = mutableListOf<BenchmarkResult>()

    /**
     * Detect device NNAPI capabilities.
     */
    fun detectCapabilities(): DeviceCapability {
        val nnapiAvailable = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1
        val nnapiVersion = when {
            Build.VERSION.SDK_INT >= 34 -> "1.3+"
            Build.VERSION.SDK_INT >= 31 -> "1.3"
            Build.VERSION.SDK_INT >= 30 -> "1.2"
            Build.VERSION.SDK_INT >= 29 -> "1.2"
            Build.VERSION.SDK_INT >= 28 -> "1.1"
            Build.VERSION.SDK_INT >= 27 -> "1.0"
            else -> "none"
        }

        val cores = Runtime.getRuntime().availableProcessors()
        val estimatedClass = when {
            Build.VERSION.SDK_INT < 28 -> "LOW"
            cores < 4 -> "LOW"
            cores < 8 -> "MID"
            else -> "HIGH"
        }

        return DeviceCapability(
            manufacturer = Build.MANUFACTURER,
            model = Build.MODEL,
            sdkInt = Build.VERSION.SDK_INT,
            nnapiAvailable = nnapiAvailable,
            nnapiVersion = nnapiVersion,
            cpuAbi = Build.SUPPORTED_ABIS.toList(),
            cores = cores,
            estimatedClass = estimatedClass,
        )
    }

    /**
     * Run full backend comparison benchmark.
     */
    fun runFullBenchmark(
        assetName: String,
        inputShape: IntArray = intArrayOf(1, 3, 64, 64),
        nRuns: Int = 20,
        nWarmup: Int = 5,
    ): List<BenchmarkResult> {
        results.clear()

        // Generate deterministic input
        val input = FloatArray(inputShape.reduce(Int::times)) { i ->
            // Simple deterministic pattern
            (i % 100) / 100f
        }

        val backends = listOf(
            BackendMode.CPU_ONLY,
            BackendMode.NNAPI_DEFAULT,
            BackendMode.NNAPI_FP16,
            BackendMode.NNAPI_NCHW,
        )

        for (backend in backends) {
            Log.i(TAG, "Benchmarking: ${backend.label}")
            try {
                model.load(assetName, backend)
                val result = model.benchmark(input, nRuns, nWarmup)
                results.add(result)
                Log.i(TAG, result.toString())
            } catch (e: Exception) {
                Log.e(TAG, "Failed for ${backend.label}: ${e.message}")
                results.add(
                    BenchmarkResult(
                        backend = backend.label,
                        nRuns = 0, meanMs = -1.0, medianMs = -1.0,
                        minMs = -1.0, maxMs = -1.0, p95Ms = -1.0,
                    )
                )
            }
        }

        return results
    }

    /**
     * Generate a formatted benchmark report.
     */
    fun generateReport(): String {
        val caps = detectCapabilities()
        val sb = StringBuilder()

        sb.appendLine("=" .repeat(70))
        sb.appendLine("SESI NNAPI BENCHMARK REPORT")
        sb.appendLine("=" .repeat(70))
        sb.appendLine("Device: ${caps.manufacturer} ${caps.model}")
        sb.appendLine("SDK: ${caps.sdkInt}, NNAPI: ${caps.nnapiVersion}")
        sb.appendLine("CPU cores: ${caps.cores}, ABI: ${caps.cpuAbi.joinToString()}")
        sb.appendLine("Estimated class: ${caps.estimatedClass}")
        sb.appendLine()
        sb.appendLine("%-30s %8s %8s %8s %8s %8s".format(
            "Backend", "Mean", "Median", "Min", "Max", "P95"
        ))
        sb.appendLine("-".repeat(70))

        for (r in results) {
            sb.appendLine(
                "%-30s %7.2f %7.2f %7.2f %7.2f %7.2f".format(
                    r.backend, r.meanMs, r.medianMs, r.minMs, r.maxMs, r.p95Ms
                )
            )
        }

        // Speedup summary
        val cpu = results.find { it.backend == BackendMode.CPU_ONLY.label }
        val best = results.filter { it.meanMs > 0 }.minByOrNull { it.meanMs }
        if (cpu != null && best != null && cpu.meanMs > 0) {
            sb.appendLine()
            sb.appendLine("Speedup vs CPU: ${"%.2f".format(cpu.meanMs / best.meanMs)}x " +
                "(best: ${best.backend})")
        }

        sb.appendLine("=" .repeat(70))
        return sb.toString()
    }

    /**
     * Save report to file.
     */
    fun saveReport(dir: File = context.filesDir): File {
        val report = generateReport()
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val file = File(dir, "nnapi_benchmark_$timestamp.txt")
        file.writeText(report)
        Log.i(TAG, "Report saved → ${file.absolutePath}")
        return file
    }

    /**
     * Run with systrace markers for Perfetto integration.
     */
    fun runWithSystrace(assetName: String, input: FloatArray, backend: BackendMode) {
        android.os.Trace.beginSection("SESI_NNAPI_${backend.name}")
        try {
            model.load(assetName, backend)
            android.os.Trace.beginSection("inference")
            model.run(input)
            android.os.Trace.endSection()
        } finally {
            android.os.Trace.endSection()
        }
    }

    fun close() {
        model.close()
    }
}

package org.mspsnetwork.sesi

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.mspsnetwork.sesi.inference.NnapiProfiler
import org.mspsnetwork.sesi.inference.SesiOnnxModel
import org.mspsnetwork.sesi.inference.BackendMode

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "SesiDemo"
        private const val MODEL_ASSET = "sesi_cfd.ort"
    }

    private lateinit var profiler: NnapiProfiler
    private lateinit var statusText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        val runButton = findViewById<Button>(R.id.runButton)

        profiler = NnapiProfiler(this)

        // Show device info
        val caps = profiler.detectCapabilities()
        statusText.text = buildString {
            appendLine("Device: ${caps.manufacturer} ${caps.model}")
            appendLine("SDK ${caps.sdkInt}, NNAPI ${caps.nnapiVersion}")
            appendLine("Class: ${caps.estimatedClass}")
        }

        runButton.setOnClickListener {
            lifecycleScope.launch {
                statusText.text = "Running benchmark..."
                val report = withContext(Dispatchers.Default) {
                    runBenchmark()
                }
                statusText.text = report
            }
        }
    }

    private fun runBenchmark(): String {
        return try {
            profiler.runFullBenchmark(MODEL_ASSET)
            val report = profiler.generateReport()
            val file = profiler.saveReport()
            Log.i(TAG, "Report saved to ${file.absolutePath}")
            report + "\nSaved to: ${file.absolutePath}"
        } catch (e: Exception) {
            Log.e(TAG, "Benchmark failed", e)
            "ERROR: ${e.message}"
        }
    }

    override fun onDestroy() {
        profiler.close()
        super.onDestroy()
    }
}

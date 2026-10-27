// Module-level build.gradle.kts
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "org.mspsnetwork.sesi"
    compileSdk = 34

    defaultConfig {
        applicationId = "org.mspsnetwork.sesi"
        minSdk = 27          // NNAPI 1.0 requires API 27+
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        // Native libs
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    // AAPT options for ML model files
    aaptOptions {
        noCompress.add("ort")
        noCompress.add("onnx")
    }
}

dependencies {
    // ONNX Runtime Mobile with NNAPI support
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")
    // Or, for latest:
    // implementation("com.microsoft.onnxruntime:onnxruntime-android:latest.release")

    // Optional: ONNX Runtime extensions for Kotlin
    implementation("com.microsoft.onnxruntime:onnxruntime-extensions-android:0.12.0")

    // AndroidX
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
}

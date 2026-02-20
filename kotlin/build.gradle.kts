import org.jlleitschuh.gradle.ktlint.reporter.ReporterType

plugins {
    kotlin("jvm") version "2.3.10"
    application
    id("com.gradleup.shadow") version "9.2.0"
    kotlin("plugin.serialization") version "2.3.10"
    id("org.jlleitschuh.gradle.ktlint") version "14.0.1"
    id("io.gitlab.arturbosch.detekt") version "1.23.6"
}

detekt {
    buildUponDefaultConfig = true
    config.setFrom(files("config/detekt.yml"))
}

group = "com.causalrag"
version = "0.0.1"

application {
    val requestedMain =
        if (project.hasProperty("mainClass")) {
            project.property("mainClass") as String
        } else {
            null
        }
    mainClass.set(requestedMain ?: "causalrag.examples.CliKt")
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("ch.qos.logback:logback-classic:1.5.13")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
    implementation("io.github.oshai:kotlin-logging-jvm:8.0.01")
    implementation("com.github.haifengl:smile-core:4.4.2")
    implementation("gg.jte:jte-kotlin:3.2.3")

    // LangChain4j dependencies
    implementation("dev.langchain4j:langchain4j:1.11.0")
    implementation("dev.langchain4j:langchain4j-open-ai:1.11.0")
    implementation("dev.langchain4j:langchain4j-azure-open-ai:1.11.0")
    implementation("dev.langchain4j:langchain4j-ollama:1.11.0")
    implementation("dev.langchain4j:langchain4j-community-neo4j:1.11.0-beta17")

    testImplementation("org.jetbrains.kotlin:kotlin-test-junit:2.3.10")
    testImplementation("io.mockk:mockk:1.14.9")
}

tasks {
    withType<Test> {
        jvmArgs("-XX:+EnableDynamicAgentLoading")
        testLogging {
            events("passed", "skipped", "failed", "standardOut", "standardError")
            showStandardStreams = true
        }
    }

    // Separate task for scriptable/CLI runs; keeps `run` intact for IDE defaults.
    val execute by registering(JavaExec::class) {
        group = "application"
        mainClass.set(application.mainClass)
        classpath = sourceSets.main.get().runtimeClasspath
    }
}

ktlint {
    version.set("1.8.0")
    verbose.set(true)
    outputToConsole.set(true)
    coloredOutput.set(true)
    reporters {
        reporter(ReporterType.CHECKSTYLE)
        reporter(ReporterType.JSON)
        reporter(ReporterType.HTML)
    }
    filter {
        exclude("**/style-violations.kt")
    }
}

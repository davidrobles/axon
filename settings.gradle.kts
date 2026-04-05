rootProject.name = "axon"

include(
    ":core",
    ":gridworld",
    ":examples",
    ":util",
)

project(":core").projectDir = file("modules/core")
project(":gridworld").projectDir = file("modules/gridworld")
project(":examples").projectDir = file("modules/examples")
project(":util").projectDir = file("modules/util")

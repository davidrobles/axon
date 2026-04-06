rootProject.name = "axon"

include(
    ":core",
    ":envs",
    ":examples",
    ":util",
)

project(":core").projectDir = file("modules/core")
project(":envs").projectDir = file("modules/envs")
project(":examples").projectDir = file("modules/examples")
project(":util").projectDir = file("modules/util")

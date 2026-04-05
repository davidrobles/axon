rootProject.name = "axon"

include(
    ":ReinforcementLearning",
    ":GridWorld",
    ":GridWorldPlay",
    ":Utils",
)

project(":ReinforcementLearning").projectDir = file("modules/ReinforcementLearning")
project(":GridWorld").projectDir = file("modules/GridWorld")
project(":GridWorldPlay").projectDir = file("modules/GridWorldPlay")
project(":Utils").projectDir = file("modules/Utils")

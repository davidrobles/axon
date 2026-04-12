package net.davidrobles.axon.envs.gridworld;

import java.util.List;
import java.util.Map;
import java.util.Random;
import net.davidrobles.axon.Environment;
import net.davidrobles.axon.StepResult;

public class GridWorldEnv implements Environment<GridWorldState, GridWorldAction> {
    private GridWorldMDP mdp;
    private GridWorldState currentState;
    private Random rng;

    public GridWorldEnv(GridWorldMDP mdp, Random rng) {
        this.mdp = mdp;
        this.rng = rng;
        reset();
    }

    //////////////////////////////
    // RL Environment Interface //
    //////////////////////////////

    @Override
    public GridWorldState getCurrentState() {
        return currentState;
    }

    @Override
    public List<GridWorldAction> getActions(GridWorldState state) {
        return mdp.getActions(state);
    }

    @Override
    public StepResult<GridWorldState> step(GridWorldAction action) {
        if (!mdp.getActions(currentState).contains(action))
            throw new IllegalArgumentException("Invalid action!");

        Map<GridWorldState, Double> stateDoubleMap = currentState.getActionNextStatesMap().get(action);
        currentState = stateDoubleMap.keySet().iterator().next();
        double reward = mdp.getReward(currentState, action, currentState);
        return new StepResult<>(currentState, reward, isTerminal());
    }

    @Override
    public GridWorldState reset() {
        currentState = mdp.getStartState();
        return currentState;
    }

    private boolean isTerminal() {
        return mdp.isTerminal(currentState);
    }
}

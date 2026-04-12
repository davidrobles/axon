package net.davidrobles.axon.envs.gridworld;

import java.util.*;
import net.davidrobles.axon.MDP;

/** A Grid World Markov Decision Process. */
public class GridWorldMDP implements MDP<GridWorldState, GridWorldAction> {
    private int cols;
    private int rows;
    private GridWorldState startState;
    private GridWorldState[][] states;
    private List<GridWorldState> allStates;
    private List<GridWorldState> terminalStates;
    private Map<GridWorldState, Double> rewards = new HashMap<GridWorldState, Double>();
    private final Random rng;

    public GridWorldMDP(int cols, int rows, Random rng) {
        this.cols = cols;
        this.rows = rows;
        this.rng = rng;
        this.states = new GridWorldState[cols][rows];
        this.allStates = new ArrayList<GridWorldState>();
        this.terminalStates = new ArrayList<GridWorldState>();

        createStates();
        createTransitions();
        initTerminalStates();
        setStartState();
    }

    private void setStartState() {
        do {
            startState = getRandomState();
        } while (isTerminal(startState));
    }

    private void initTerminalStates() {
        for (GridWorldState state : allStates) rewards.put(state, -1.0);

        GridWorldState t1;

        for (int i = 0; i < 1; i++) {
            t1 = getRandomState();
            terminalStates.add(t1);
        }

        for (GridWorldState terminalState : terminalStates) terminalState.getActionNextStatesMap().clear();
    }

    public void createStates() {
        for (int x = 0; x < cols; x++) {
            for (int y = 0; y < rows; y++) {
                GridWorldState newState = new GridWorldState(x, y);
                states[x][y] = newState;
                allStates.add(newState);
            }
        }
    }

    private void createTransitions() {
        for (GridWorldState state : allStates) {
            Map<GridWorldAction, Map<GridWorldState, Double>> map = new HashMap<GridWorldAction, Map<GridWorldState, Double>>();

            // UP
            Map<GridWorldState, Double> upTransProb = new HashMap<GridWorldState, Double>();

            if (state.getY() > 0 && state.getY() < rows)
                upTransProb.put(states[state.getX()][state.getY() - 1], 1.0);
            else upTransProb.put(states[state.getX()][state.getY()], 1.0);

            map.put(GridWorldAction.UP, upTransProb);

            // LEFT
            Map<GridWorldState, Double> leftTransProb = new HashMap<GridWorldState, Double>();

            if (state.getX() > 0 && state.getX() < rows)
                leftTransProb.put(states[state.getX() - 1][state.getY()], 1.0);
            else leftTransProb.put(states[state.getX()][state.getY()], 1.0);

            map.put(GridWorldAction.LEFT, leftTransProb);

            // RIGHT
            Map<GridWorldState, Double> rightTransProb = new HashMap<GridWorldState, Double>();

            if (state.getX() >= 0 && state.getX() < (rows - 1))
                rightTransProb.put(states[state.getX() + 1][state.getY()], 1.0);
            else rightTransProb.put(states[state.getX()][state.getY()], 1.0);

            map.put(GridWorldAction.RIGHT, rightTransProb);

            // DOWN
            Map<GridWorldState, Double> downTransProb = new HashMap<GridWorldState, Double>();

            if (state.getY() >= 0 && state.getY() < (rows - 1))
                downTransProb.put(states[state.getX()][state.getY() + 1], 1.0);
            else downTransProb.put(states[state.getX()][state.getY()], 1.0);

            map.put(GridWorldAction.DOWN, downTransProb);

            // Don't forget to set the map in the states!
            state.setActionNextStatesMap(map);
        }
    }

    protected GridWorldState getRandomState() {
        return allStates.get(rng.nextInt(allStates.size()));
    }

    public int getCols() {
        return cols;
    }

    public int getRows() {
        return rows;
    }

    public GridWorldState getState(int x, int y) {
        return states[x][y];
    }

    public List<GridWorldState> getTerminalStates() {
        return terminalStates;
    }

    // MDP Interface

    @Override
    public GridWorldState getStartState() {
        return startState;
    }

    @Override
    public List<GridWorldAction> getActions(GridWorldState state) {
        return new ArrayList<GridWorldAction>(state.getActionNextStatesMap().keySet());
    }

    @Override
    public List<GridWorldState> getStates() {
        return allStates;
    }

    @Override
    public Map<GridWorldState, Double> getTransitions(GridWorldState state, GridWorldAction action) {
        if (isTerminal(state)) return new HashMap<GridWorldState, Double>();

        return state.getActionNextStatesMap().get(action);
    }

    @Override
    public double getReward(GridWorldState state, GridWorldAction action, GridWorldState nextState) {
        return rewards.get(state);
    }

    @Override
    public boolean isTerminal(GridWorldState state) {
        return terminalStates.contains(state);
    }
}

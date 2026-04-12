package net.davidrobles.axon.envs.gridworld;

import java.util.Map;

public class GridWorldState {
    private int x;
    private int y;
    private Map<GridWorldAction, Map<GridWorldState, Double>> actionNextStatesMap;

    public GridWorldState(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public Map<GridWorldAction, Map<GridWorldState, Double>> getActionNextStatesMap() {
        return actionNextStatesMap;
    }

    public void setActionNextStatesMap(Map<GridWorldAction, Map<GridWorldState, Double>> actionNextStatesMap) {
        this.actionNextStatesMap = actionNextStatesMap;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        GridWorldState gwState = (GridWorldState) o;

        if (x != gwState.x) return false;
        if (y != gwState.y) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = x;
        result = 31 * result + y;
        return result;
    }

    @Override
    public String toString() {
        return "GridWorldState{" + "x=" + x + ", y=" + y + '}';
    }
}

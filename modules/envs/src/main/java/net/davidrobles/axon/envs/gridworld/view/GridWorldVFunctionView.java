package net.davidrobles.axon.envs.gridworld.view;

import java.awt.*;
import net.davidrobles.axon.envs.gridworld.GridWorldState;
import net.davidrobles.axon.envs.gridworld.GridWorldEnv;
import net.davidrobles.axon.envs.gridworld.GridWorldMDP;
import net.davidrobles.axon.util.color.ColorMap;
import net.davidrobles.axon.values.VFunction;
import net.davidrobles.axon.values.VFunctionObserver;

public class GridWorldVFunctionView extends GridWorldView implements VFunctionObserver<GridWorldState> {
    private VFunction<GridWorldState> vFunction;

    public GridWorldVFunctionView(
            GridWorldMDP gridWorld, int cellWidth, int cellHeight, GridWorldEnv env) {
        super(gridWorld, cellWidth, cellHeight, env);
    }

    @Override
    public void drawValues(Graphics g) {
        if (vFunction != null) {
            float max = Float.MIN_VALUE;
            float min = Float.MAX_VALUE;

            // calculate max and min color values
            for (GridWorldState state : gw.getStates()) {
                float colorValue = (float) vFunction.getValue(state);

                if (colorValue < min) min = colorValue;

                if (colorValue > max) max = colorValue;
            }

            ColorMap colorMap = new ColorMap(min, max, ColorMap.getJet());

            // draw states
            for (GridWorldState state : gw.getStates()) {
                g.setColor(colorMap.getColor(vFunction.getValue(state)));
                g.fillRect(
                        state.getX() * cellWidth, state.getY() * cellHeight, cellWidth, cellHeight);

                // draw values
                if (valuesEnabled) {
                    g.setColor(Color.WHITE);
                    String t = String.format("%.1f", (float) vFunction.getValue(state));
                    g.drawString(
                            t,
                            state.getX() * cellWidth,
                            state.getY() * cellHeight + cellHeight / 2);
                }
            }
        }

        // draw current state
        g.setColor(Color.RED);
        g.fillRect(
                env.getCurrentState().getX() * cellWidth,
                env.getCurrentState().getY() * cellHeight,
                cellWidth,
                cellHeight);
    }

    @Override
    public void valueFunctionUpdated(VFunction<GridWorldState> vFunction) {
        this.vFunction = vFunction;
        repaint();
    }
}

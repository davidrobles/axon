package net.davidrobles.axon.envs.gridworld.view;

import java.awt.*;
import net.davidrobles.axon.envs.gridworld.GWAction;
import net.davidrobles.axon.envs.gridworld.GWState;
import net.davidrobles.axon.envs.gridworld.GridWorldEnv;
import net.davidrobles.axon.envs.gridworld.GridWorldMDP;
import net.davidrobles.axon.util.color.ColorMap;
import net.davidrobles.axon.values.QFunction;
import net.davidrobles.axon.values.QFunctionObserver;

public class GWQFunctionView extends GridWorldView implements QFunctionObserver<GWState, GWAction> {
    private QFunction<GWState, GWAction> qFunction;

    public GWQFunctionView(
            GridWorldMDP gridWorld, int cellWidth, int cellHeight, GridWorldEnv environment) {
        super(gridWorld, cellWidth, cellHeight, environment);
    }

    @Override
    public void drawValues(Graphics g) {
        if (qFunction != null) {
            float max = Float.MIN_VALUE;
            float min = Float.MAX_VALUE;

            for (GWState state : gw.getStates()) {
                for (GWAction action : gw.getActions(state)) {
                    float colorValue = (float) qFunction.getValue(state, action);
                    if (colorValue < min) {
                        min = colorValue;
                    }
                    if (colorValue > max) {
                        max = colorValue;
                    }
                }
            }

            ColorMap colorMap = new ColorMap(min, max, ColorMap.getJet());

            for (GWState state : gw.getStates()) {
                for (GWAction action : gw.getActions(state)) {
                    g.setColor(colorMap.getColor(qFunction.getValue(state, action)));

                    if (action == GWAction.UP) {
                        // Top
                        g.fillRect(
                                state.getX() * cellWidth + cellWidth / 3,
                                state.getY() * cellHeight,
                                cellWidth / 3,
                                cellHeight / 3);
                    }

                    if (action == GWAction.DOWN) {
                        // Bottom
                        g.fillRect(
                                state.getX() * cellWidth + cellWidth / 3,
                                state.getY() * cellHeight + (cellHeight / 3) * 2,
                                cellWidth / 3,
                                cellHeight / 3);
                    }

                    if (action == GWAction.LEFT) {
                        // Left
                        g.fillRect(
                                state.getX() * cellWidth,
                                state.getY() * cellHeight + (cellHeight / 3),
                                cellWidth / 3,
                                cellHeight / 3);
                    }

                    if (action == GWAction.RIGHT) {
                        // Right
                        g.fillRect(
                                state.getX() * cellWidth + (cellWidth / 3) * 2,
                                state.getY() * cellHeight + (cellHeight / 3),
                                cellWidth / 3,
                                cellHeight / 3);
                    }
                }
            }
        }
    }

    @Override
    public void qFunctionUpdated(QFunction<GWState, GWAction> qFunction) {
        this.qFunction = qFunction;
        repaint();
    }
}

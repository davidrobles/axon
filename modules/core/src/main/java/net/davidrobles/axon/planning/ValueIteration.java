package net.davidrobles.axon.planning;

import java.util.Map;
import net.davidrobles.axon.MDP;
import net.davidrobles.axon.policies.TabularPolicy;
import net.davidrobles.axon.policies.Policy;
import net.davidrobles.axon.valuefunctions.AbstractVFunctionObservable;
import net.davidrobles.axon.valuefunctions.TabularVFunction;

public class ValueIteration<S, A> extends AbstractVFunctionObservable<S> implements Planner<S, A> {
    private MDP<S, A> mdp;
    private TabularVFunction<S> table = new TabularVFunction<S>();
    private double theta; // A small positive number used as a termination condition
    private double gamma; // Discount factor

    public ValueIteration(MDP<S, A> mdp, double theta, double gamma) {
        this.mdp = mdp;
        this.theta = theta;
        this.gamma = gamma;
    }

    @Override
    public Policy<S, A> solve() {
        double delta;

        do {
            delta = 0;

            for (S state : mdp.getStates()) {
                double oldStateValue = table.getValue(state);
                double newStateValue = Double.NEGATIVE_INFINITY;

                for (A action : mdp.getActions(state)) {
                    double tot = 0;
                    Map<S, Double> nextTransitions = mdp.getTransitions(state, action);

                    for (S nextState : nextTransitions.keySet()) {
                        double probability = nextTransitions.get(nextState);
                        double reward = mdp.getReward(state, action, nextState);
                        double nextStateValue = table.getValue(nextState);
                        tot += probability * (reward + (gamma * nextStateValue));
                    }

                    if (tot > newStateValue) newStateValue = tot;
                }

                table.setValue(state, newStateValue);
                delta = Math.max(delta, Math.abs(oldStateValue - table.getValue(state)));
                notifyVFunctionObservers(table);
            }

        } while (delta > theta);

        // Extract greedy policy via one-step lookahead
        TabularPolicy<S, A> policy = new TabularPolicy<>();
        for (S state : mdp.getStates()) {
            A bestAction = null;
            double bestValue = Double.NEGATIVE_INFINITY;
            for (A action : mdp.getActions(state)) {
                double tot = 0;
                for (Map.Entry<S, Double> e : mdp.getTransitions(state, action).entrySet()) {
                    tot +=
                            e.getValue()
                                    * (mdp.getReward(state, action, e.getKey())
                                            + gamma * table.getValue(e.getKey()));
                }
                if (tot > bestValue) {
                    bestValue = tot;
                    bestAction = action;
                }
            }
            if (bestAction != null) policy.setAction(state, bestAction);
        }
        return policy;
    }
}

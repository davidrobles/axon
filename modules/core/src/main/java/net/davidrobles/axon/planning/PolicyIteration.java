package net.davidrobles.axon.planning;

import java.util.*;
import net.davidrobles.axon.MDP;
import net.davidrobles.axon.policies.TabularPolicy;
import net.davidrobles.axon.policies.Policy;
import net.davidrobles.axon.values.AbstractVFunctionObservable;
import net.davidrobles.axon.values.TabularVFunction;

public class PolicyIteration<S, A> extends AbstractVFunctionObservable<S> implements Planner<S, A> {
    private MDP<S, A> mdp;
    private TabularVFunction<S> table = new TabularVFunction<S>();
    private boolean policyStable = false;
    private double theta; // A small positive number used as a termination condition
    private double gamma; // Discount factor
    private TabularPolicy<S, A> policy;

    public PolicyIteration(MDP<S, A> mdp, double theta, double gamma) {
        this.mdp = mdp;
        this.theta = theta;
        this.gamma = gamma;
    }

    private void policyEvaluation() {
        double delta;

        do {
            delta = 0;

            for (S state : mdp.getStates()) {
                double oldValue = table.getValue(state);
                Set<S> nextStates = new HashSet<>();
                for (A a : mdp.getActions(state))
                    nextStates.addAll(mdp.getTransitions(state, a).keySet());
                A action = policy.selectAction(state, new ArrayList<>(mdp.getActions(state)));
                double newValue = 0;

                for (S nextState : nextStates) {
                    Map<S, Double> tt = mdp.getTransitions(state, action);

                    if (tt.get(nextState) != null) {
                        double transition = mdp.getTransitions(state, action).get(nextState);
                        double reward = mdp.getReward(state, action, nextState);
                        double nextStateValue = table.getValue(nextState);
                        newValue += transition * (reward + (gamma * nextStateValue));
                    }
                }

                table.setValue(state, newValue);
                delta = Math.max(delta, Math.abs(oldValue - table.getValue(state)));
                notifyVFunctionObservers(table);
            }
        } while (delta > theta);
    }

    private void policyImprovement() {
        policyStable = true;

        for (S state : mdp.getStates()) {
            A oldAction = policy.selectAction(state, new ArrayList<>(mdp.getActions(state)));
            A bestAction = null;
            double bestScore = Double.NEGATIVE_INFINITY;

            for (A action : mdp.getActions(state)) {
                double tot = 0;
                Map<S, Double> transitions = mdp.getTransitions(state, action);

                for (S nextState : transitions.keySet()) {
                    double transition = transitions.get(nextState);
                    double reward = mdp.getReward(state, action, nextState);
                    double nextValue = table.getValue(nextState);
                    tot += transition * (reward + (gamma * nextValue));
                }

                if (tot > bestScore) {
                    bestAction = action;
                    bestScore = tot;
                }
            }

            policy.setAction(state, bestAction);

            if (!mdp.isTerminal(state)
                    && !oldAction.equals(
                            policy.selectAction(state, new ArrayList<>(mdp.getActions(state)))))
                policyStable = false;
        }
    }

    private final Random rng = new Random();

    @Override
    public Policy<S, A> solve() {
        policy = new TabularPolicy<>();

        // Initialize the policy arbitrarily
        for (S state : mdp.getStates()) {
            List<A> actions = new ArrayList<>(mdp.getActions(state));
            if (!actions.isEmpty()) {
                Collections.shuffle(actions);
                policy.setAction(state, actions.get(rng.nextInt(actions.size())));
            }
        }

        while (!policyStable) {
            policyEvaluation();
            policyImprovement();
        }

        return policy;
    }
}

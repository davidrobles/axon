package net.davidrobles.axon;

import java.util.Random;
import net.davidrobles.axon.envs.gridworld.GWAction;
import net.davidrobles.axon.envs.gridworld.GWState;
import net.davidrobles.axon.envs.gridworld.GridWorldEnv;
import net.davidrobles.axon.envs.gridworld.GridWorldMDP;
import net.davidrobles.axon.envs.gridworld.view.GWVView;
import net.davidrobles.axon.envs.gridworld.view.GWViewQValues;
import net.davidrobles.axon.agents.*;
import net.davidrobles.axon.planning.*;
import net.davidrobles.axon.policies.EpsilonGreedy;
import net.davidrobles.axon.policies.RandomPolicy;
import net.davidrobles.axon.prediction.*;
import net.davidrobles.axon.valuefunctions.TabularQFunction;
import net.davidrobles.axon.valuefunctions.TabularVFunction;
import net.davidrobles.axon.util.DRFrame;

public class GWRun {
    private static final Random RNG = new Random();

    private static void policyIteration() {
        double theta = 0.01;
        double gamma = 0.99;
        GridWorldMDP mdp = new GridWorldMDP(25, 25, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        GWVView view = new GWVView(mdp, 20, 20, env);
        new DRFrame(view);
        PolicyIteration<GWState, GWAction> learner = new PolicyIteration<>(mdp, theta, gamma);
        learner.addVFunctionObserver(view);
        learner.solve();
    }

    private static void valueIteration() {
        double theta = 0.1;
        double gamma = 0.99;
        GridWorldMDP mdp = new GridWorldMDP(25, 25, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        GWVView view = new GWVView(mdp, 20, 20, env);
        new DRFrame(view);
        ValueIteration<GWState, GWAction> learner = new ValueIteration<>(mdp, theta, gamma);
        learner.addVFunctionObserver(view);
        learner.solve();
    }

    private static void tabularMCPrediction() {
        double alpha = 0.01;
        double gamma = 0.99;
        int numEpisodes = 2000;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularVFunction<GWState> vTable = new TabularVFunction<>(alpha);
        RandomPolicy<GWState, GWAction> policy = new RandomPolicy<>(RNG);
        GWVView view = new GWVView(mdp, 20, 20, env);
        new DRFrame(view, "MC Prediction");
        MCPrediction<GWState, GWAction> agent = new MCPrediction<>(vTable, policy, gamma);
        agent.addVFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularMCControl() {
        double alpha = 0.01;
        double gamma = 0.99;
        int numEpisodes = 2000;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularQFunction<GWState, GWAction> qTable = new TabularQFunction<>(alpha);
        EpsilonGreedy<GWState, GWAction> policy = new EpsilonGreedy<>(qTable, 0.1, RNG);
        GWViewQValues view = new GWViewQValues(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new DRFrame(view, "MC Control");
        MCControl<GWState, GWAction> agent = new MCControl<>(qTable, policy, gamma);
        agent.addQFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularTD0() {
        double alpha = 0.01;
        double gamma = 0.99;
        int numEpisodes = 5000;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularVFunction<GWState> vTable = new TabularVFunction<>(alpha);
        RandomPolicy<GWState, GWAction> policy = new RandomPolicy<>(RNG);
        GWVView view = new GWVView(mdp, 20, 20, env);
        new DRFrame(view, "TD(0)");
        TD0<GWState, GWAction> agent = new TD0<>(vTable, policy, gamma);
        agent.addVFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularTDLambda() {
        double alpha = 0.001;
        double gamma = 0.99;
        double lambda = 0.1;
        int numEpisodes = 1000;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularVFunction<GWState> vTable = new TabularVFunction<>(alpha);
        RandomPolicy<GWState, GWAction> policy = new RandomPolicy<>(RNG);
        GWVView view = new GWVView(mdp, 20, 20, env);
        new DRFrame(view, "TD(λ)");
        TDLambda<GWState, GWAction> agent = new TDLambda<>(vTable, policy, gamma, lambda);
        agent.addVFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularSARSA() {
        double alpha = 0.1;
        double gamma = 0.99;
        int numEpisodes = 100;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularQFunction<GWState, GWAction> qTable = new TabularQFunction<>(alpha);
        RandomPolicy<GWState, GWAction> policy = new RandomPolicy<>(RNG);
        GWViewQValues view = new GWViewQValues(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new DRFrame(view, "SARSA");
        SARSA<GWState, GWAction> agent = new SARSA<>(qTable, policy, gamma);
        agent.addQFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularQLearning() {
        double alpha = 0.1;
        double gamma = 0.99;
        int numEpisodes = 300;
        GridWorldMDP mdp = new GridWorldMDP(25, 25, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularQFunction<GWState, GWAction> qTable = new TabularQFunction<>(alpha);
        EpsilonGreedy<GWState, GWAction> policy = new EpsilonGreedy<>(qTable, 0.1, RNG);
        GWViewQValues view = new GWViewQValues(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new DRFrame(view, "Q-Learning");
        QLearning<GWState, GWAction> agent = new QLearning<>(qTable, policy, gamma);
        agent.addQFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularSARSALambda() {
        double alpha = 0.1;
        double gamma = 0.99;
        double lambda = 0.9;
        int numEpisodes = 100;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularQFunction<GWState, GWAction> qTable = new TabularQFunction<>(alpha);
        RandomPolicy<GWState, GWAction> policy = new RandomPolicy<>(RNG);
        GWViewQValues view = new GWViewQValues(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new DRFrame(view, "SARSA(λ)");
        SARSALambda<GWState, GWAction> agent = new SARSALambda<>(qTable, policy, gamma, lambda);
        agent.addQFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    public static void main(String[] args) {
        //        tabularMCPrediction();
        //        tabularMCControl();
        //        tabularTD0();
        tabularSARSA();
        //        valueIteration();
        //        tabularQLearning();
        //        tabularTDLambda();
        //        tabularSARSALambda();
        //        policyIteration();
    }
}

package net.davidrobles.axon.examples.gridworld;

import java.util.Random;
import net.davidrobles.axon.RLLoop;
import net.davidrobles.axon.agents.*;
import net.davidrobles.axon.agents.ReplayBuffer;
import net.davidrobles.axon.envs.gridworld.GWAction;
import net.davidrobles.axon.envs.gridworld.GWState;
import net.davidrobles.axon.envs.gridworld.GridWorldEnv;
import net.davidrobles.axon.envs.gridworld.GridWorldMDP;
import net.davidrobles.axon.envs.gridworld.view.GWQFunctionView;
import net.davidrobles.axon.envs.gridworld.view.GWVFunctionView;
import net.davidrobles.axon.planning.*;
import net.davidrobles.axon.policies.EpsilonGreedy;
import net.davidrobles.axon.policies.RandomPolicy;
import net.davidrobles.axon.policies.SoftmaxPolicy;
import net.davidrobles.axon.policies.UCBPolicy;
import net.davidrobles.axon.prediction.*;
import net.davidrobles.axon.prediction.NStepTD;
import net.davidrobles.axon.util.AppFrame;
import net.davidrobles.axon.valuefunctions.TabularQFunction;
import net.davidrobles.axon.valuefunctions.TabularVFunction;

public class GWRun {
    private static final Random RNG = new Random();

    private static void policyIteration() {
        double theta = 0.01;
        double gamma = 0.99;
        GridWorldMDP mdp = new GridWorldMDP(25, 25, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        GWVFunctionView view = new GWVFunctionView(mdp, 20, 20, env);
        new AppFrame(view);
        PolicyIteration<GWState, GWAction> learner = new PolicyIteration<>(mdp, theta, gamma);
        learner.addVFunctionObserver(view);
        learner.solve();
    }

    private static void valueIteration() {
        double theta = 0.1;
        double gamma = 0.99;
        GridWorldMDP mdp = new GridWorldMDP(25, 25, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        GWVFunctionView view = new GWVFunctionView(mdp, 20, 20, env);
        new AppFrame(view);
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
        GWVFunctionView view = new GWVFunctionView(mdp, 20, 20, env);
        new AppFrame(view, "MC Prediction");
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
        GWQFunctionView view = new GWQFunctionView(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new AppFrame(view, "MC Control");
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
        GWVFunctionView view = new GWVFunctionView(mdp, 20, 20, env);
        new AppFrame(view, "TD(0)");
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
        GWVFunctionView view = new GWVFunctionView(mdp, 20, 20, env);
        new AppFrame(view, "TD(λ)");
        TDLambda<GWState, GWAction> agent = new TDLambda<>(vTable, policy, gamma, lambda);
        agent.addVFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularQLearningWithReplay() {
        double alpha = 0.1;
        double gamma = 0.99;
        int bufferCapacity = 5000;
        int batchSize = 32;
        int numEpisodes = 300;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularQFunction<GWState, GWAction> qTable = new TabularQFunction<>(alpha);
        EpsilonGreedy<GWState, GWAction> policy = new EpsilonGreedy<>(qTable, 0.1, RNG);
        GWQFunctionView view = new GWQFunctionView(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new AppFrame(view, "Q-Learning + Replay (batch=" + batchSize + ")");
        QLearningWithReplay<GWState, GWAction> agent =
                new QLearningWithReplay<>(
                        qTable, policy, gamma, new ReplayBuffer<>(bufferCapacity), batchSize, RNG);
        agent.addQFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularDynaQ() {
        double alpha = 0.1;
        double gamma = 0.99;
        int planningSteps = 50;
        int numEpisodes = 50;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularQFunction<GWState, GWAction> qTable = new TabularQFunction<>(alpha);
        EpsilonGreedy<GWState, GWAction> policy = new EpsilonGreedy<>(qTable, 0.1, RNG);
        GWQFunctionView view = new GWQFunctionView(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new AppFrame(view, "Dyna-Q (n=" + planningSteps + ")");
        DynaQ<GWState, GWAction> agent = new DynaQ<>(qTable, policy, gamma, planningSteps, RNG);
        agent.addQFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularNStepTD() {
        double alpha = 0.01;
        double gamma = 0.99;
        int n = 5;
        int numEpisodes = 1000;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularVFunction<GWState> vTable = new TabularVFunction<>(alpha);
        RandomPolicy<GWState, GWAction> policy = new RandomPolicy<>(RNG);
        GWVFunctionView view = new GWVFunctionView(mdp, 20, 20, env);
        new AppFrame(view, "n-step TD (n=" + n + ")");
        NStepTD<GWState, GWAction> agent = new NStepTD<>(vTable, policy, n, gamma);
        agent.addVFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularNStepSARSA() {
        double alpha = 0.1;
        double gamma = 0.99;
        int n = 5;
        int numEpisodes = 200;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularQFunction<GWState, GWAction> qTable = new TabularQFunction<>(alpha);
        EpsilonGreedy<GWState, GWAction> policy = new EpsilonGreedy<>(qTable, 0.1, RNG);
        GWQFunctionView view = new GWQFunctionView(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new AppFrame(view, "n-step SARSA (n=" + n + ")");
        NStepSARSA<GWState, GWAction> agent = new NStepSARSA<>(qTable, policy, n, gamma);
        agent.addQFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularExpectedSARSA() {
        double alpha = 0.1;
        double gamma = 0.99;
        int numEpisodes = 100;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularQFunction<GWState, GWAction> qTable = new TabularQFunction<>(alpha);
        EpsilonGreedy<GWState, GWAction> policy = new EpsilonGreedy<>(qTable, 0.1, RNG);
        GWQFunctionView view = new GWQFunctionView(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new AppFrame(view, "Expected SARSA");
        ExpectedSARSA<GWState, GWAction> agent = new ExpectedSARSA<>(qTable, policy, gamma);
        agent.addQFunctionObserver(view);
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
        GWQFunctionView view = new GWQFunctionView(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new AppFrame(view, "SARSA");
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
        GWQFunctionView view = new GWQFunctionView(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new AppFrame(view, "Q-Learning");
        QLearning<GWState, GWAction> agent = new QLearning<>(qTable, policy, gamma);
        agent.addQFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularUCB() {
        double alpha = 0.1;
        double gamma = 0.99;
        double c = 1.0;
        int numEpisodes = 300;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularQFunction<GWState, GWAction> qTable = new TabularQFunction<>(alpha);
        UCBPolicy<GWState, GWAction> policy = new UCBPolicy<>(qTable, c);
        GWQFunctionView view = new GWQFunctionView(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new AppFrame(view, "UCB (c=" + c + ")");
        QLearning<GWState, GWAction> agent = new QLearning<>(qTable, policy, gamma);
        agent.addQFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularSoftmax() {
        double alpha = 0.1;
        double gamma = 0.99;
        double temperature = 0.5;
        int numEpisodes = 300;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularQFunction<GWState, GWAction> qTable = new TabularQFunction<>(alpha);
        SoftmaxPolicy<GWState, GWAction> policy = new SoftmaxPolicy<>(qTable, temperature, RNG);
        GWQFunctionView view = new GWQFunctionView(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new AppFrame(view, "Softmax (τ=" + temperature + ")");
        QLearning<GWState, GWAction> agent = new QLearning<>(qTable, policy, gamma);
        agent.addQFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularDoubleQLearning() {
        double alpha = 0.1;
        double gamma = 0.99;
        int numEpisodes = 300;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularQFunction<GWState, GWAction> qA = new TabularQFunction<>(alpha);
        TabularQFunction<GWState, GWAction> qB = new TabularQFunction<>(alpha);
        EpsilonGreedy<GWState, GWAction> policy =
                new EpsilonGreedy<>(
                        (s, a) -> (qA.getValue(s, a) + qB.getValue(s, a)) / 2.0, 0.1, RNG);
        GWQFunctionView view = new GWQFunctionView(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new AppFrame(view, "Double Q-Learning");
        DoubleQLearning<GWState, GWAction> agent =
                new DoubleQLearning<>(qA, qB, policy, gamma, RNG);
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
        GWQFunctionView view = new GWQFunctionView(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new AppFrame(view, "SARSA(λ)");
        SARSALambda<GWState, GWAction> agent = new SARSALambda<>(qTable, policy, gamma, lambda);
        agent.addQFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    public static void main(String[] args) {
        //        tabularMCPrediction();
        //        tabularMCControl();
        //        tabularQLearningWithReplay();
        //        tabularDynaQ();
        //        tabularNStepTD();
        //        tabularNStepSARSA();
        //        tabularTD0();
        tabularSARSA();
        //        tabularExpectedSARSA();
        //        valueIteration();
        //        tabularQLearning();
        //        tabularTDLambda();
        //        tabularSARSALambda();
        //        policyIteration();
        //        tabularUCB();
        //        tabularSoftmax();
        //        tabularDoubleQLearning();
    }
}

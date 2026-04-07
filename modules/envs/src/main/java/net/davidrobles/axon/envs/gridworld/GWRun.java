package net.davidrobles.axon.envs.gridworld;

import java.util.Random;
import net.davidrobles.axon.envs.gridworld.view.GWVView;
import net.davidrobles.axon.envs.gridworld.view.GWViewQValues;
import net.davidrobles.axon.RLLoop;
import net.davidrobles.axon.ReplayBuffer;
import net.davidrobles.axon.agents.DynaQ;
import net.davidrobles.axon.agents.ExpectedSARSA;
import net.davidrobles.axon.agents.MCControl;
import net.davidrobles.axon.agents.NStepSARSA;
import net.davidrobles.axon.agents.QLearning;
import net.davidrobles.axon.agents.ReplayQLearning;
import net.davidrobles.axon.agents.SARSA;
import net.davidrobles.axon.agents.SARSALambda;
import net.davidrobles.axon.policies.EpsilonGreedy;
import net.davidrobles.axon.policies.RandomPolicy;
import net.davidrobles.axon.prediction.MCPrediction;
import net.davidrobles.axon.prediction.NStepTD;
import net.davidrobles.axon.prediction.TD0;
import net.davidrobles.axon.prediction.TDLambda;
import net.davidrobles.axon.valuefunctions.TabularQFunction;
import net.davidrobles.axon.valuefunctions.TabularVFunction;
import net.davidrobles.axon.util.DRFrame;

public class GWRun {
    private static final Random RNG = new Random();

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

    private static void tabularExpectedSARSA() {
        double alpha = 0.1;
        double gamma = 0.99;
        int numEpisodes = 100;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularQFunction<GWState, GWAction> qTable = new TabularQFunction<>(alpha);
        EpsilonGreedy<GWState, GWAction> policy = new EpsilonGreedy<>(qTable, 0.1, RNG);
        GWViewQValues view = new GWViewQValues(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new DRFrame(view, "Expected SARSA");
        ExpectedSARSA<GWState, GWAction> agent = new ExpectedSARSA<>(qTable, policy, gamma);
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
        GWVView view = new GWVView(mdp, 20, 20, env);
        new DRFrame(view, "n-step TD (n=" + n + ")");
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
        GWViewQValues view = new GWViewQValues(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new DRFrame(view, "n-step SARSA (n=" + n + ")");
        NStepSARSA<GWState, GWAction> agent = new NStepSARSA<>(qTable, policy, n, gamma);
        agent.addQFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularReplayQLearning() {
        double alpha = 0.1;
        double gamma = 0.99;
        int bufferCapacity = 5000;
        int batchSize = 32;
        int numEpisodes = 300;
        GridWorldMDP mdp = new GridWorldMDP(20, 20, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularQFunction<GWState, GWAction> qTable = new TabularQFunction<>(alpha);
        EpsilonGreedy<GWState, GWAction> policy = new EpsilonGreedy<>(qTable, 0.1, RNG);
        GWViewQValues view = new GWViewQValues(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new DRFrame(view, "Q-Learning + Replay (batch=" + batchSize + ")");
        ReplayQLearning<GWState, GWAction> agent =
                new ReplayQLearning<>(qTable, policy, gamma, new ReplayBuffer<>(bufferCapacity), batchSize, RNG);
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
        GWViewQValues view = new GWViewQValues(mdp, 20, 20, env);
        view.setGridEnabled(true);
        new DRFrame(view, "Dyna-Q (n=" + planningSteps + ")");
        DynaQ<GWState, GWAction> agent = new DynaQ<>(qTable, policy, gamma, planningSteps, RNG);
        agent.addQFunctionObserver(view);
        RLLoop.run(env, agent, policy, numEpisodes);
    }

    private static void tabularTD0() {
        double alpha = 0.01;
        double gamma = 0.99;
        int numEpisodes = 500;
        GridWorldMDP mdp = new GridWorldMDP(50, 50, RNG);
        GridWorldEnv env = new GridWorldEnv(mdp, RNG);
        TabularVFunction<GWState> vTable = new TabularVFunction<>(alpha);
        RandomPolicy<GWState, GWAction> policy = new RandomPolicy<>(RNG);
        GWVView view = new GWVView(mdp, 10, 10, env);
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
        //        tabularNStepTD();
        //        tabularNStepSARSA();
        //        tabularDynaQ();
        //        tabularReplayQLearning();
        //        tabularTD0();
        tabularSARSA();
        //        tabularExpectedSARSA();
        //        tabularQLearning();
        //        tabularTDLambda();
        //        tabularSARSALambda();
    }
}

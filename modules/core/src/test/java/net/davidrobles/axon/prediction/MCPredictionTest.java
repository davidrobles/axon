package net.davidrobles.axon.prediction;

import static org.junit.Assert.*;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import net.davidrobles.axon.StepResult;
import net.davidrobles.axon.policies.RandomPolicy;
import net.davidrobles.axon.valuefunctions.TabularVFunction;
import net.davidrobles.axon.valuefunctions.VFunction;
import org.junit.Before;
import org.junit.Test;

public class MCPredictionTest {

    private static final double EPS = 1e-9;

    private TabularVFunction<String> table;
    private MCPrediction<String, String> agent;

    @Before
    public void setUp() {
        table = new TabularVFunction<>(0.5);
        agent = new MCPrediction<>(table, new RandomPolicy<>(new java.util.Random(0)), 1.0);
    }

    // -------------------------------------------------------------------------
    // Return computation: G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
    // -------------------------------------------------------------------------

    @Test
    public void singleStepEpisodeUpdatesWithReward() {
        // Episode: s0 -r=2.0-> done
        // G = 2.0;  new V(s0) = 0 + 0.5*(2.0-0) = 1.0
        agent.observe("s0", new StepResult<>("s1", 2.0, true));
        assertEquals(1.0, table.getValue("s0"), EPS);
    }

    @Test
    public void noUpdateBeforeEpisodeEnds() {
        agent.observe("s0", new StepResult<>("s1", 5.0, false));
        assertEquals(0.0, table.getValue("s0"), EPS);
    }

    @Test
    public void multiStepEpisodeComputesReturnBackwards() {
        // Episode: s0 -r=0-> s1 -r=0-> s2 -r=1-> done; gamma=1
        // G(s2) = 1, G(s1) = 0 + 1*1 = 1, G(s0) = 0 + 1*1 = 1
        // new V(s0) = 0 + 0.5*1 = 0.5
        // new V(s1) = 0 + 0.5*1 = 0.5
        // new V(s2) = 0 + 0.5*1 = 0.5
        agent.observe("s0", new StepResult<>("s1", 0.0, false));
        agent.observe("s1", new StepResult<>("s2", 0.0, false));
        agent.observe("s2", new StepResult<>("s3", 1.0, true));
        assertEquals(0.5, table.getValue("s0"), EPS);
        assertEquals(0.5, table.getValue("s1"), EPS);
        assertEquals(0.5, table.getValue("s2"), EPS);
    }

    @Test
    public void discountingReducesEarlierReturns() {
        // gamma=0.5; episode: s0 -r=0-> s1 -r=4-> done
        MCPrediction<String, String> discounted =
                new MCPrediction<>(table, new RandomPolicy<>(new java.util.Random(0)), 0.5);
        // G(s1) = 4, G(s0) = 0 + 0.5*4 = 2
        // new V(s1) = 0 + 0.5*4 = 2.0
        // new V(s0) = 0 + 0.5*2 = 1.0
        discounted.observe("s0", new StepResult<>("s1", 0.0, false));
        discounted.observe("s1", new StepResult<>("s2", 4.0, true));
        assertEquals(1.0, table.getValue("s0"), EPS);
        assertEquals(2.0, table.getValue("s1"), EPS);
    }

    // -------------------------------------------------------------------------
    // First-visit: only the first visit to a state per episode is updated
    // -------------------------------------------------------------------------

    @Test
    public void firstVisitSkipsRevisitedState() {
        // Episode: s0 -r=0-> s1 -r=0-> s0 -r=4-> done; gamma=1
        // Returns: G(t=2,s0)=4, G(t=1,s1)=4, G(t=0,s0)=4
        // First visit to s0 is t=0 with G=4 → new V(s0) = 0 + 0.5*4 = 2.0
        // s0 at t=2 is skipped (already visited)
        agent.observe("s0", new StepResult<>("s1", 0.0, false));
        agent.observe("s1", new StepResult<>("s0", 0.0, false));
        agent.observe("s0", new StepResult<>("s3", 4.0, true));
        assertEquals(2.0, table.getValue("s0"), EPS);
    }

    // -------------------------------------------------------------------------
    // Buffer is cleared between episodes
    // -------------------------------------------------------------------------

    @Test
    public void bufferClearedBetweenEpisodes() {
        // Episode 1: s0 -r=2-> done → V(s0) = 1.0
        agent.observe("s0", new StepResult<>("s1", 2.0, true));
        assertEquals(1.0, table.getValue("s0"), EPS);

        // Episode 2: s0 -r=4-> done → update from 1.0: 1.0 + 0.5*(4-1.0) = 2.5
        agent.observe("s0", new StepResult<>("s1", 4.0, true));
        assertEquals(2.5, table.getValue("s0"), EPS);
    }

    // -------------------------------------------------------------------------
    // update() delegates to observe()
    // -------------------------------------------------------------------------

    @Test
    public void updateDelegatesToObserve() {
        // Same as singleStepEpisodeUpdatesWithReward but via update()
        agent.update("s0", "a0", new StepResult<>("s1", 2.0, true), List.of());
        assertEquals(1.0, table.getValue("s0"), EPS);
    }

    // -------------------------------------------------------------------------
    // Observer
    // -------------------------------------------------------------------------

    @Test
    public void observerNotifiedOncePerStateUpdateAtEpisodeEnd() {
        AtomicInteger count = new AtomicInteger();
        agent.addVFunctionObserver(vf -> count.incrementAndGet());

        // 2-step episode visits 2 distinct states → 2 notifications
        agent.observe("s0", new StepResult<>("s1", 0.0, false));
        agent.observe("s1", new StepResult<>("s2", 1.0, true));
        assertEquals(2, count.get());
    }

    @Test
    public void observerNotNotifiedBeforeEpisodeEnds() {
        AtomicInteger count = new AtomicInteger();
        agent.addVFunctionObserver(vf -> count.incrementAndGet());

        agent.observe("s0", new StepResult<>("s1", 1.0, false));
        assertEquals(0, count.get());
    }

    @Test
    public void observerReceivesUpdatedVFunction() {
        VFunction<String>[] captured = new VFunction[1];
        agent.addVFunctionObserver(vf -> captured[0] = vf);

        agent.observe("s0", new StepResult<>("s1", 2.0, true));

        assertNotNull(captured[0]);
        assertEquals(1.0, captured[0].getValue("s0"), EPS);
    }

    @Test
    public void duplicateObserverIsRegisteredOnce() {
        AtomicInteger count = new AtomicInteger();
        net.davidrobles.axon.valuefunctions.VFunctionObserver<String> o =
                vf -> count.incrementAndGet();
        agent.addVFunctionObserver(o);
        agent.addVFunctionObserver(o);

        agent.observe("s0", new StepResult<>("s1", 1.0, true));
        assertEquals(1, count.get());
    }

    // -------------------------------------------------------------------------
    // Construction validation
    // -------------------------------------------------------------------------

    @Test(expected = IllegalArgumentException.class)
    public void gammaBelowZeroIsRejected() {
        new MCPrediction<>(table, new RandomPolicy<>(new java.util.Random(0)), -0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void gammaAboveOneIsRejected() {
        new MCPrediction<>(table, new RandomPolicy<>(new java.util.Random(0)), 1.1);
    }

    @Test(expected = NullPointerException.class)
    public void nullTableIsRejected() {
        new MCPrediction<>(null, new RandomPolicy<>(new java.util.Random(0)), 0.9);
    }

    @Test(expected = NullPointerException.class)
    public void nullPolicyIsRejected() {
        new MCPrediction<>(table, null, 0.9);
    }
}

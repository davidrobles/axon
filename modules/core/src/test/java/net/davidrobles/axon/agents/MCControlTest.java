package net.davidrobles.axon.agents;

import static org.junit.Assert.*;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import net.davidrobles.axon.Experience;
import net.davidrobles.axon.policies.GreedyPolicy;
import net.davidrobles.axon.valuefunctions.QFunction;
import net.davidrobles.axon.valuefunctions.TabularQFunction;
import org.junit.Before;
import org.junit.Test;

public class MCControlTest {

    private static final double EPS = 1e-9;

    private TabularQFunction<String, String> table;
    private MCControl<String, String> agent;

    @Before
    public void setUp() {
        table = new TabularQFunction<>(0.5);
        agent = new MCControl<>(table, new GreedyPolicy<>(table), 1.0);
    }

    // -------------------------------------------------------------------------
    // Return computation: G_t = r_t + γ*r_{t+1} + ...
    // -------------------------------------------------------------------------

    @Test
    public void singleStepEpisodeUpdatesWithReward() {
        // Episode: (s0,a0) -r=2-> done; G = 2.0
        // new Q(s0,a0) = 0 + 0.5*(2.0-0) = 1.0
        agent.update(new Experience<>("s0", "a0", 2.0, "s1", true, List.of()));
        assertEquals(1.0, table.getValue("s0", "a0"), EPS);
    }

    @Test
    public void noUpdateBeforeEpisodeEnds() {
        agent.update(new Experience<>("s0", "a0", 5.0, "s1", false, List.of("a0")));
        assertEquals(0.0, table.getValue("s0", "a0"), EPS);
    }

    @Test
    public void multiStepEpisodeComputesReturnBackwards() {
        // Episode: (s0,a0) -r=0-> (s1,a0) -r=0-> (s2,a0) -r=1-> done; gamma=1
        // G(s2,a0)=1, G(s1,a0)=1, G(s0,a0)=1
        // new Q = 0 + 0.5*1 = 0.5 for all
        agent.update(new Experience<>("s0", "a0", 0.0, "s1", false, List.of("a0")));
        agent.update(new Experience<>("s1", "a0", 0.0, "s2", false, List.of("a0")));
        agent.update(new Experience<>("s2", "a0", 1.0, "s3", true, List.of()));
        assertEquals(0.5, table.getValue("s0", "a0"), EPS);
        assertEquals(0.5, table.getValue("s1", "a0"), EPS);
        assertEquals(0.5, table.getValue("s2", "a0"), EPS);
    }

    @Test
    public void discountingReducesEarlierReturns() {
        // gamma=0.5; episode: (s0,a0) -r=0-> (s1,a0) -r=4-> done
        MCControl<String, String> discounted =
                new MCControl<>(table, new GreedyPolicy<>(table), 0.5);
        // G(s1,a0)=4, G(s0,a0)=0+0.5*4=2
        // new Q(s1,a0) = 0 + 0.5*4 = 2.0
        // new Q(s0,a0) = 0 + 0.5*2 = 1.0
        discounted.update(new Experience<>("s0", "a0", 0.0, "s1", false, List.of("a0")));
        discounted.update(new Experience<>("s1", "a0", 4.0, "s2", true, List.of()));
        assertEquals(1.0, table.getValue("s0", "a0"), EPS);
        assertEquals(2.0, table.getValue("s1", "a0"), EPS);
    }

    // -------------------------------------------------------------------------
    // First-visit: only the first visit to a (s,a) pair per episode is updated
    // -------------------------------------------------------------------------

    @Test
    public void firstVisitSkipsRevisitedStatActionPair() {
        // Episode: (s0,a0) -r=0-> (s1,a0) -r=0-> (s0,a0) -r=4-> done; gamma=1
        // Returns: G(t=2)=4, G(t=1)=4, G(t=0)=4
        // First visit to (s0,a0) is t=0 with G=4 → new Q = 0.5*4 = 2.0
        agent.update(new Experience<>("s0", "a0", 0.0, "s1", false, List.of("a0")));
        agent.update(new Experience<>("s1", "a0", 0.0, "s0", false, List.of("a0")));
        agent.update(new Experience<>("s0", "a0", 4.0, "s3", true, List.of()));
        assertEquals(2.0, table.getValue("s0", "a0"), EPS);
    }

    @Test
    public void differentActionsAtSameStateAreTrackedSeparately() {
        // (s0,a0) and (s0,a1) are different pairs — both should be updated
        agent.update(new Experience<>("s0", "a0", 0.0, "s1", false, List.of("a1")));
        agent.update(new Experience<>("s0", "a1", 2.0, "s2", true, List.of()));
        // G(s0,a1)=2, G(s0,a0)=2
        assertEquals(1.0, table.getValue("s0", "a0"), EPS);
        assertEquals(1.0, table.getValue("s0", "a1"), EPS);
    }

    // -------------------------------------------------------------------------
    // Buffer is cleared between episodes
    // -------------------------------------------------------------------------

    @Test
    public void bufferClearedBetweenEpisodes() {
        // Episode 1: Q(s0,a0) = 1.0
        agent.update(new Experience<>("s0", "a0", 2.0, "s1", true, List.of()));
        assertEquals(1.0, table.getValue("s0", "a0"), EPS);

        // Episode 2: update from 1.0: 1.0 + 0.5*(4-1.0) = 2.5
        agent.update(new Experience<>("s0", "a0", 4.0, "s1", true, List.of()));
        assertEquals(2.5, table.getValue("s0", "a0"), EPS);
    }

    // -------------------------------------------------------------------------
    // selectAction delegates to the policy
    // -------------------------------------------------------------------------

    @Test
    public void selectActionDelegatesToPolicy() {
        table.setValue("s0", "a1", 10.0);
        assertEquals("a1", agent.selectAction("s0", List.of("a0", "a1")));
    }

    // -------------------------------------------------------------------------
    // Observer
    // -------------------------------------------------------------------------

    @Test
    public void observerNotifiedOncePerPairUpdateAtEpisodeEnd() {
        AtomicInteger count = new AtomicInteger();
        agent.addQFunctionObserver(qf -> count.incrementAndGet());

        // 2-step episode with 2 distinct (s,a) pairs → 2 notifications
        agent.update(new Experience<>("s0", "a0", 0.0, "s1", false, List.of("a0")));
        agent.update(new Experience<>("s1", "a0", 1.0, "s2", true, List.of()));
        assertEquals(2, count.get());
    }

    @Test
    public void observerNotNotifiedBeforeEpisodeEnds() {
        AtomicInteger count = new AtomicInteger();
        agent.addQFunctionObserver(qf -> count.incrementAndGet());

        agent.update(new Experience<>("s0", "a0", 1.0, "s1", false, List.of("a0")));
        assertEquals(0, count.get());
    }

    @Test
    public void observerReceivesUpdatedQFunction() {
        QFunction<String, String>[] captured = new QFunction[1];
        agent.addQFunctionObserver(qf -> captured[0] = qf);

        agent.update(new Experience<>("s0", "a0", 2.0, "s1", true, List.of()));

        assertNotNull(captured[0]);
        assertEquals(1.0, captured[0].getValue("s0", "a0"), EPS);
    }

    @Test
    public void duplicateObserverIsRegisteredOnce() {
        AtomicInteger count = new AtomicInteger();
        net.davidrobles.axon.valuefunctions.QFunctionObserver<String, String> o =
                qf -> count.incrementAndGet();
        agent.addQFunctionObserver(o);
        agent.addQFunctionObserver(o);

        agent.update(new Experience<>("s0", "a0", 1.0, "s1", true, List.of()));
        assertEquals(1, count.get());
    }

    // -------------------------------------------------------------------------
    // Construction validation
    // -------------------------------------------------------------------------

    @Test(expected = IllegalArgumentException.class)
    public void gammaBelowZeroIsRejected() {
        new MCControl<>(table, new GreedyPolicy<>(table), -0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void gammaAboveOneIsRejected() {
        new MCControl<>(table, new GreedyPolicy<>(table), 1.1);
    }

    @Test(expected = NullPointerException.class)
    public void nullTableIsRejected() {
        new MCControl<>(null, new GreedyPolicy<>(table), 0.9);
    }

    @Test(expected = NullPointerException.class)
    public void nullPolicyIsRejected() {
        new MCControl<>(table, null, 0.9);
    }
}

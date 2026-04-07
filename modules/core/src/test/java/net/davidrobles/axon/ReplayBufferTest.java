package net.davidrobles.axon;

import static org.junit.Assert.*;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;

public class ReplayBufferTest {

    private ReplayBuffer<String, String> buffer;

    private static Transition<String, String> t(String state) {
        return new Transition<>(state, "a0", new StepResult<>("s_next", 0.0, false), List.of("a0"));
    }

    @Before
    public void setUp() {
        buffer = new ReplayBuffer<>(5);
    }

    // -------------------------------------------------------------------------
    // Basic add / size / isFull
    // -------------------------------------------------------------------------

    @Test
    public void emptyBufferHasSizeZero() {
        assertEquals(0, buffer.size());
    }

    @Test
    public void sizeIncreasesWithEachAdd() {
        buffer.add(t("s0"));
        assertEquals(1, buffer.size());
        buffer.add(t("s1"));
        assertEquals(2, buffer.size());
    }

    @Test
    public void sizeDoesNotExceedCapacity() {
        for (int i = 0; i < 10; i++) buffer.add(t("s" + i));
        assertEquals(5, buffer.size());
    }

    @Test
    public void isFullAfterCapacityReached() {
        assertFalse(buffer.isFull());
        for (int i = 0; i < 5; i++) buffer.add(t("s" + i));
        assertTrue(buffer.isFull());
    }

    // -------------------------------------------------------------------------
    // Circular overwrite
    // -------------------------------------------------------------------------

    @Test
    public void oldestTransitionOverwrittenWhenFull() {
        ReplayBuffer<String, String> buf = new ReplayBuffer<>(3);
        Transition<String, String> t0 = t("s0");
        Transition<String, String> t1 = t("s1");
        Transition<String, String> t2 = t("s2");
        Transition<String, String> t3 = t("s3"); // should overwrite t0

        buf.add(t0);
        buf.add(t1);
        buf.add(t2);
        buf.add(t3);

        // capacity=3, so only t1, t2, t3 remain
        List<Transition<String, String>> all = buf.sample(3, new Random(0));
        Set<String> states = new HashSet<>();
        for (Transition<String, String> t : all) states.add(t.state());

        assertFalse(states.contains("s0"));
        assertTrue(states.contains("s1"));
        assertTrue(states.contains("s2"));
        assertTrue(states.contains("s3"));
    }

    // -------------------------------------------------------------------------
    // Sampling
    // -------------------------------------------------------------------------

    @Test
    public void sampleReturnsBatchOfCorrectSize() {
        for (int i = 0; i < 5; i++) buffer.add(t("s" + i));
        assertEquals(3, buffer.sample(3, new Random(0)).size());
    }

    @Test
    public void sampleWithoutReplacement() {
        for (int i = 0; i < 5; i++) buffer.add(t("s" + i));
        List<Transition<String, String>> batch = buffer.sample(5, new Random(0));
        Set<String> states = new HashSet<>();
        for (Transition<String, String> t : batch) states.add(t.state());
        assertEquals(5, states.size()); // all distinct
    }

    @Test
    public void sampleSizeOneAlwaysReturnsOneTransition() {
        buffer.add(t("s0"));
        assertEquals(1, buffer.sample(1, new Random(0)).size());
    }

    @Test(expected = IllegalArgumentException.class)
    public void sampleLargerThanBufferSizeThrows() {
        buffer.add(t("s0"));
        buffer.sample(2, new Random(0)); // only 1 in buffer
    }

    // -------------------------------------------------------------------------
    // Construction validation
    // -------------------------------------------------------------------------

    @Test(expected = IllegalArgumentException.class)
    public void capacityZeroIsRejected() {
        new ReplayBuffer<>(0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void negativeCapacityIsRejected() {
        new ReplayBuffer<>(-1);
    }

    @Test(expected = NullPointerException.class)
    public void addNullTransitionIsRejected() {
        buffer.add(null);
    }
}

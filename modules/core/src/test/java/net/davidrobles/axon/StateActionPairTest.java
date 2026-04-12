package net.davidrobles.axon;

import static org.junit.Assert.*;

import org.junit.Test;

public class StateActionPairTest {

    @Test
    public void accessors() {
        StateActionPair<String, Integer> pair = new StateActionPair<>("s0", 1);
        assertEquals("s0", pair.state());
        assertEquals(Integer.valueOf(1), pair.action());
    }

    @Test
    public void equalPairsAreEqual() {
        StateActionPair<String, String> a = new StateActionPair<>("s0", "up");
        StateActionPair<String, String> b = new StateActionPair<>("s0", "up");
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
    }

    @Test
    public void differentStateMeansNotEqual() {
        assertNotEquals(new StateActionPair<>("s0", "up"), new StateActionPair<>("s1", "up"));
    }

    @Test
    public void differentActionMeansNotEqual() {
        assertNotEquals(
                new StateActionPair<>("s0", "up"), new StateActionPair<>("s0", "down"));
    }

    @Test
    public void nullComponentsAreSupported() {
        StateActionPair<String, String> pair = new StateActionPair<>(null, null);
        assertNull(pair.state());
        assertNull(pair.action());
        assertEquals(pair, new StateActionPair<>(null, null));
    }

    @Test
    public void toStringContainsBothComponents() {
        String s = new StateActionPair<>("s0", "up").toString();
        assertTrue(s.contains("s0"));
        assertTrue(s.contains("up"));
    }
}
